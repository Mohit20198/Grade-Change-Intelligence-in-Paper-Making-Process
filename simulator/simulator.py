"""
simulator.py
------------
Core simulation engine for the Honeywell Paper-Making Process Simulator.

Runs a 1-second resolution time-loop that:
  1. Reads setpoint trajectories from the EventScheduler
  2. Applies disturbances when scheduled
  3. Steps all FOLPD transfer functions
  4. Adds realistic sensor noise
  5. Accumulates the full time-series DataFrame

Outputs
-------
  - process_data.csv      : full 1-second time-series of all tags
  - grade_change_log.csv  : event windows with off-spec flags
  - operator_action_log.csv : synthetic operator actions

Usage
-----
    python simulator.py
    python simulator.py --days 3 --seed 42 --output_dir ./output
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_functions import (
    make_stock_to_bw,
    make_speed_to_bw,
    make_steam_to_moisture,
    make_speed_to_moisture,
    make_filler_to_ash,
    REF_MACHINE_SPEED,
    REF_BW, REF_STOCK_FLOW, REF_STEAM_PRESSURE, REF_MOISTURE, REF_ASH,
)
from event_scheduler import EventScheduler, GradeChangeEvent, GRADE_LIBRARY
from disturbances import make_disturbance


# ---------------------------------------------------------------------------
# Noise parameters (all Gaussian, small magnitude)
# ---------------------------------------------------------------------------
NOISE_STD = {
    "basis_weight": 0.15,   # gsm
    "moisture":     0.08,   # %
    "ash":          0.05,   # %
    "stock_flow":   1.5,    # L/min
    "steam_pressure":0.012, # bar
    "machine_speed": 0.3,   # m/min
    "filler_flow":  0.8,    # kg/min
    "caliper":      0.2,    # µm
}

# Caliper: approximately 1.2 µm per gsm (rough empirical for copy paper)
CALIPER_PER_GSM = 1.2

# Off-spec threshold
BW_OFFSPEC_THRESHOLD = 0.025  # 2.5% deviation


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class PaperMachineSimulator:
    """
    Runs the full simulation and produces output DataFrames.

    Parameters
    ----------
    total_days          : simulation duration in days
    seed                : global random seed
    output_dir          : where to save CSVs
    disturbance_prob    : probability of disturbance per grade change
    validation_fraction : fraction of events held out
    """

    def __init__(self,
                 total_days: float = 8.0,
                 seed: int = 42,
                 output_dir: str = "./output",
                 disturbance_prob: float = 0.35,
                 validation_fraction: float = 0.20):

        self.total_seconds = int(total_days * 24 * 3600)
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(seed)

        # Build event schedule
        self.scheduler = EventScheduler(
            total_seconds=self.total_seconds,
            seed=seed,
            disturbance_prob=disturbance_prob,
            validation_fraction=validation_fraction,
        )
        self.events = self.scheduler.build_events()

        print(f"[Scheduler] {len(self.events)} grade-change events scheduled.")
        n_val = sum(1 for e in self.events if e.is_validation)
        n_dist = sum(1 for e in self.events if e.disturbance)
        print(f"  Validation events : {n_val}")
        print(f"  Events with disturbance : {n_dist}")

    # ------------------------------------------------------------------
    def _init_transfer_functions(self, init_spec):
        """
        Instantiate FOLPD blocks at initial steady state.

        BW model (absolute inputs):
          BW = TF_stock(stock_flow) + TF_speed(machine_speed - REF_SPEED)
          where TF_stock gain = 0.06 gsm/(L/min) and TF_speed gain = -0.068 gsm/(m/min)

          At G60 SS: TF_stock(1000) = 60 gsm, TF_speed(0 deviation) = 0  → BW = 60 ✓
          Across grades: BW tracks proportionally to stock_flow.

        Moisture model:
          moisture = TF_steam(steam_pressure) + TF_speed_moist(speed_deviation)
          TF_steam gain = -1.333 %/bar; at G60: TF_steam(4.5) ≈ -6.0  ← biased by init_out=+6.0
          Actually: moisture is POSITIVE so init_out = target = 6.0,
          and gain*input at SS = 6.0 → this works because FOLPD reaches K*u in SS:
          K * steam_pressure = -1.333 * 4.5 = -6.0 → sign problem.

          Fix: moisture TF is actually an inverse model where increase in steam
          REDUCES moisture. We model it as:
            moisture_raw = moist_target_SS  - |K_steam| * (steam - steam_SS_ref)
          i.e., steam deviation from nominal causes moisture deviation.
          But to keep FOLPD structure, we set:
            init_out = moist_target  (FOLPD bias toward positive moisture)
            input = steam_SP         → TF(steam_SP) → output in %
          with gain calibrated as NEGATIVE so that at ref steam, output = ref_moisture.
        """
        # Calibrate initial states for each TF
        bw_init = init_spec.bw_setpoint
        sf_init = init_spec.stock_flow
        ff_init = init_spec.filler_flow

        # Stock TF: absolute values. K=0.06 ensures K*flow = bw for all grades
        # (all grades in GRADE_LIBRARY are designed so bw_sp = 0.06 * stock_flow)
        self.tf_stock_bw = make_stock_to_bw(
            initial_out=bw_init,
            initial_in=sf_init,
        )
        # Speed BW TF: deviation-based (input = ms_meas - ms_sp), init at zero
        self.tf_speed_bw = make_speed_to_bw(
            initial_out=0.0,
            initial_in=0.0,
        )
        # Steam moisture TF: deviation-based (input = sp_meas - sp_sp), init at zero
        self.tf_steam_moisture = make_steam_to_moisture(
            initial_out=0.0,
            initial_in=0.0,
        )
        # Speed moisture TF (HIDDEN): deviation-based, init at zero
        self.tf_speed_moisture = make_speed_to_moisture(
            initial_out=0.0,
            initial_in=0.0,
        )
        # Filler → Ash: absolute values
        self.tf_filler_ash = make_filler_to_ash(
            initial_out=init_spec.ash_target,
            initial_in=ff_init,
        )

    # ------------------------------------------------------------------
    def _get_active_disturbances(self, t: int):
        """
        Return dict of active disturbance objects keyed by type for time t.
        """
        active = {}
        for evt in self.events:
            if evt.disturbance and evt.disturbance_start is not None:
                dt_local = t - evt.disturbance_start
                if 0 <= dt_local < 700:  # max disturbance window
                    if evt.event_id not in self._dist_objects:
                        np.random.seed(self.seed + evt.event_id + 9999)
                        self._dist_objects[evt.event_id] = make_disturbance(
                            evt.disturbance, rng_seed=self.seed + evt.event_id
                        )
                    active[evt.event_id] = (
                        evt.disturbance,
                        self._dist_objects[evt.event_id],
                        dt_local,
                        evt,
                    )
        return active

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full simulation loop. Returns the time-series DataFrame."""

        print(f"\n[Simulator] Starting {self.total_seconds:,}s simulation "
              f"({self.total_seconds/3600:.1f} hours)...")
        t0 = time.time()

        self._dist_objects = {}  # event_id -> disturbance object

        # --- Initial spec ---
        init_spec = GRADE_LIBRARY[0]
        # Find first event and use its old_spec as initial
        if self.events:
            init_spec = self.events[0].old_spec
        self._init_transfer_functions(init_spec)

        rng = np.random.default_rng(self.seed + 1)

        # Pre-allocate arrays for speed
        N = self.total_seconds
        cols = [
            "timestamp", "elapsed_s",
            "stock_flow_sp", "stock_flow",
            "steam_pressure_sp", "steam_pressure",
            "machine_speed_sp", "machine_speed",
            "filler_flow_sp", "filler_flow",
            "bw_setpoint", "basis_weight",
            "moisture_target", "moisture",
            "ash_target", "ash",
            "caliper",
            "event_id", "in_grade_change", "is_disturbance_active",
            "disturbance_type",
        ]
        records = []

        # Baseline timestamp (start of simulation = "Day 1, 00:00:00")
        import datetime
        base_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)

        print("[Simulator] Running time-loop... (this may take ~30s for 3 days)")

        for t in range(N):
            # --- Get setpoints from scheduler ---
            spec, active_evt = self.scheduler.get_setpoints_at(t)

            # --- Compute MV signals with noise ---
            sf_sp  = spec.stock_flow
            sp_sp  = spec.steam_pressure
            ms_sp  = spec.machine_speed
            ff_sp  = spec.filler_flow
            bw_sp  = spec.bw_setpoint
            moist_sp = spec.moisture_target

            # --- Apply Overrides ---
            if hasattr(self, 'overrides') and self.overrides:
                for ov in self.overrides:
                    if ov['start_t'] <= t <= ov['end_t']:
                        ramp_len = ov['ramp_dur']
                        elapsed = t - ov['start_t']
                        frac = min(1.0, elapsed / ramp_len)
                        sf_sp += ov['sf_diff'] * frac
                        sp_sp += ov['sp_diff'] * frac
                        ms_sp += ov['ms_diff'] * frac

            # Measured MVs (setpoint + sensor noise)
            sf_meas = sf_sp  + rng.normal(0, NOISE_STD["stock_flow"])
            sp_meas = sp_sp  + rng.normal(0, NOISE_STD["steam_pressure"])
            ms_meas = ms_sp  + rng.normal(0, NOISE_STD["machine_speed"])
            ff_meas = ff_sp  + rng.normal(0, NOISE_STD["filler_flow"])

            dist_type_str = ""
            is_dist = False

            # --- Apply disturbances ---
            active_dists = self._get_active_disturbances(t)
            for eid, (dtype, dist_obj, dt_local, evt) in active_dists.items():
                is_dist = True
                dist_type_str = dtype
                if dtype == "steam_sag":
                    sp_meas += dist_obj.delta(dt_local)  # measured steam deviates from SP
                elif dtype == "speed_hunting":
                    ms_meas += dist_obj.delta(dt_local)
                elif dtype == "stock_surge":
                    sf_meas += dist_obj.delta(dt_local)  # stock flow perturbation → BW impact
                # sensor_spike applied after TF step (below)

            # ---------------------------------------------------------------
            # Physical model (FOLPD on absolute MV signals):
            #
            #   BW(t)       = TF_stock(stock_flow_meas)
            #                 + TF_speed(machine_speed_meas)   [inverse, corrective]
            #
            #   moisture(t) = TF_steam(steam_pressure_meas)    [inverse: more steam=drier]
            #                 + TF_speed_moist(machine_speed_meas)  [HIDDEN 90s lag]
            #
            # TF gains are calibrated so at steady state the CV = target:
            #   gain_stock = bw_sp_SS / stock_flow_SS        (positive)
            #   gain_speed = 0  at nominal                   (corrective only)
            #
            # During a grade-change ramp, sp values change continuously
            # but the TFs lag behind → produces realistic BW undershoot/overshoot.
            # ---------------------------------------------------------------

            bw_from_stock = self.tf_stock_bw.step(sf_meas)

            # Speed BW correction: deviation of measured speed from CURRENT SP.
            # At SS: ms_meas ≈ ms_sp + noise → delta ≈ 0 → tf_speed_bw → 0
            # During disturbance (hunting): ms_meas deviates from ms_sp → BW correction
            # During ramp: ms_sp and ms_meas move together → delta ≈ noise (small)
            # BUT: speed TF also captures the dynamic lag of the grade ramp via the
            # fact that stock TF lags behind sf_meas while speed TF captures the
            # instantaneous speed effect. Since BW = K*flow/speed physics:
            # when speed ramps faster than stock TF responds, BW transiently dips.
            ms_dev_bw = ms_meas - ms_sp    # deviation from current speed SP
            bw_from_speed = self.tf_speed_bw.step(ms_dev_bw)

            # Total BW: absolute stock contribution + speed perturbation
            bw_raw = bw_from_stock + bw_from_speed

            # Moisture: steam deviation drives the primary response
            # At SS: sp_meas ≈ sp_sp → delta ≈ 0 → tf_steam outputs 0 → moisture = init_out
            # But init_out was set to -moist_target, and we negate → gives +moist_target ✓
            sp_dev = sp_meas - sp_sp
            moist_from_steam_dev = self.tf_steam_moisture.step(sp_dev)  # negative output when steam rises
            # steam increase (sp_dev > 0) → TF output < 0 → moisture decreases
            moist_from_steam = moist_sp + moist_from_steam_dev

            # Hidden speed-moisture coupling (deviation)
            ms_dev_moist = ms_meas - ms_sp
            moist_from_speed = self.tf_speed_moisture.step(ms_dev_moist)  # HIDDEN 90s lag
            moisture_raw = moist_from_steam + moist_from_speed

            ash_raw = self.tf_filler_ash.step(ff_meas)

            # --- Add measurement noise to CVs ---
            bw_noise = rng.normal(0, NOISE_STD["basis_weight"])
            # Sensor spike applied to BW only
            for eid, (dtype, dist_obj, dt_local, evt) in active_dists.items():
                if dtype == "sensor_spike":
                    bw_noise += dist_obj.delta(dt_local)

            bw_meas      = bw_raw + bw_noise
            moisture_meas = moisture_raw + rng.normal(0, NOISE_STD["moisture"])
            ash_meas      = ash_raw + rng.normal(0, NOISE_STD["ash"])
            caliper_meas  = bw_meas * CALIPER_PER_GSM + rng.normal(0, NOISE_STD["caliper"])

            # --- Check Override end ---
            if hasattr(self, 'overrides') and self.overrides:
                stop_sim = False
                for ov in self.overrides:
                    if t > ov['end_t']:
                        stop_sim = True
                if stop_sim:
                    break

            # --- Record ---
            if t >= getattr(self, 'start_recording', 0):
                timestamp_str = (base_dt + datetime.timedelta(seconds=t)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                records.append({
                    "timestamp":         timestamp_str,
                    "elapsed_s":         t,
                    "stock_flow_sp":     round(sf_sp, 2),
                    "stock_flow":        round(max(0, sf_meas), 2),
                    "steam_pressure_sp": round(sp_sp, 4),
                    "steam_pressure":    round(max(0, sp_meas), 4),
                    "machine_speed_sp":  round(ms_sp, 2),
                    "machine_speed":     round(max(0, ms_meas), 2),
                    "filler_flow_sp":    round(ff_sp, 2),
                    "filler_flow":       round(max(0, ff_meas), 2),
                    "bw_setpoint":       round(bw_sp, 3),
                    "basis_weight":      round(max(0, bw_meas), 3),
                    "moisture_target":   round(spec.moisture_target, 3),
                    "moisture":          round(max(0, moisture_meas), 3),
                    "ash_target":        round(spec.ash_target, 3),
                    "ash":               round(max(0, ash_meas), 3),
                    "caliper":           round(max(0, caliper_meas), 2),
                    "event_id":          active_evt.event_id if active_evt else -1,
                    "in_grade_change":   int(active_evt is not None and t <= active_evt.end_ts),
                    "is_disturbance_active": int(is_dist),
                    "disturbance_type":  dist_type_str,
                })

            if t % 50000 == 0 and t > 0:
                pct = 100 * t / N
                elapsed = time.time() - t0
                eta = elapsed / (t / N) - elapsed
                print(f"  {pct:.1f}%  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        print(f"[Simulator] Time-loop done in {time.time()-t0:.1f}s. Building DataFrame...")
        df = pd.DataFrame(records)

        # --- Mark off-spec in events ---
        self._mark_off_spec(df)

        print("[Simulator] Simulation complete.")
        return df

    # ------------------------------------------------------------------
    def _mark_off_spec(self, df: pd.DataFrame):
        """
        For each grade-change event, check if BW deviated more than ±2.5% from
        the new setpoint during the stabilization phase.
        
        Since the TF (tau=45, deadtime=15) takes ~150s to reach 95% of its final
        value after the ramp ends, we start checking 180s after the ramp end.
        Any deviation >2.5% after this grace period means the machine produced
        excessive off-spec paper (took too long to settle or had a disturbance).
        """
        for evt in self.events:
            grace_period = 180
            check_start = evt.end_ts + grace_period

            window = df[
                (df["elapsed_s"] >= check_start) &
                (df["elapsed_s"] <= evt.settle_ts)
            ]
            if window.empty:
                evt.went_off_spec = False
                continue

            bw_sp = evt.new_spec.bw_setpoint
            max_dev = (window["basis_weight"] - bw_sp).abs().max() / bw_sp
            
            evt.went_off_spec = bool(max_dev > BW_OFFSPEC_THRESHOLD)

    # ------------------------------------------------------------------
    def build_event_log(self) -> pd.DataFrame:
        """Build the grade-change event summary table."""
        import datetime
        base_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)

        rows = []
        for evt in self.events:
            rows.append({
                "event_id":        evt.event_id,
                "start_timestamp": (base_dt + datetime.timedelta(seconds=evt.start_ts)).strftime("%Y-%m-%d %H:%M:%S"),
                "end_timestamp":   (base_dt + datetime.timedelta(seconds=evt.end_ts)).strftime("%Y-%m-%d %H:%M:%S"),
                "settle_timestamp":(base_dt + datetime.timedelta(seconds=evt.settle_ts)).strftime("%Y-%m-%d %H:%M:%S"),
                "start_elapsed_s": evt.start_ts,
                "end_elapsed_s":   evt.end_ts,
                "settle_elapsed_s":evt.settle_ts,
                "old_grade":       evt.old_grade,
                "new_grade":       evt.new_grade,
                "old_bw_sp":       evt.old_spec.bw_setpoint,
                "new_bw_sp":       evt.new_spec.bw_setpoint,
                "ramp_duration_s": evt.ramp_duration,
                "disturbance":     evt.disturbance or "none",
                "went_off_spec":   evt.went_off_spec,
                "is_validation":   evt.is_validation,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def build_operator_log(self) -> pd.DataFrame:
        """Build the operator action log DataFrame."""
        import datetime
        base_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)

        rows = []
        for action in sorted(self.scheduler.operator_log, key=lambda x: x.timestamp):
            rows.append({
                "timestamp":  (base_dt + datetime.timedelta(seconds=action.timestamp)).strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s":  action.timestamp,
                "event_id":   action.event_id,
                "tag":        action.tag,
                "old_value":  action.old_value,
                "new_value":  action.new_value,
                "note":       action.note,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def save_outputs(self, df: pd.DataFrame):
        """Save all three CSVs."""
        ts_path  = self.output_dir / "process_data.csv"
        evt_path = self.output_dir / "grade_change_log.csv"
        ops_path = self.output_dir / "operator_action_log.csv"

        print(f"\n[Output] Saving process_data.csv ({len(df):,} rows)...")
        df.to_csv(ts_path, index=False)

        event_log = self.build_event_log()
        print(f"[Output] Saving grade_change_log.csv ({len(event_log)} events)...")
        event_log.to_csv(evt_path, index=False)

        op_log = self.build_operator_log()
        print(f"[Output] Saving operator_action_log.csv ({len(op_log)} entries)...")
        op_log.to_csv(ops_path, index=False)

        # --- Summary stats ---
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"  Duration          : {self.total_seconds/3600:.1f} hours")
        print(f"  Total samples     : {len(df):,}")
        print(f"  Grade changes     : {len(self.events)}")
        val_events = event_log[event_log["is_validation"]]
        train_events = event_log[~event_log["is_validation"]]
        print(f"  Training events   : {len(train_events)}")
        print(f"  Validation events : {len(val_events)}")
        n_offspec = event_log["went_off_spec"].sum()
        print(f"  Off-spec events   : {n_offspec} ({100*n_offspec/len(event_log):.1f}%)")
        n_dist = (event_log["disturbance"] != "none").sum()
        print(f"  Disturbance events: {n_dist} ({100*n_dist/len(event_log):.1f}%)")
        print("="*60)
        print(f"\nFiles written to: {self.output_dir.resolve()}")
        print(f"  {ts_path.name}")
        print(f"  {evt_path.name}")
        print(f"  {ops_path.name}")

        return {
            "process_df":    df,
            "event_log_df":  event_log,
            "operator_log_df": op_log,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Honeywell Paper-Making Process Simulator"
    )
    parser.add_argument("--days",    type=float, default=8.0,
                        help="Simulation duration in days (default: 8)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--disturbance_prob", type=float, default=0.35,
                        help="Probability of disturbance per transition (0-1)")
    parser.add_argument("--validation_fraction", type=float, default=0.20,
                        help="Fraction of events held out for validation")
    args = parser.parse_args()

    sim = PaperMachineSimulator(
        total_days=args.days,
        seed=args.seed,
        output_dir=args.output_dir,
        disturbance_prob=args.disturbance_prob,
        validation_fraction=args.validation_fraction,
    )
    df = sim.run()
    sim.save_outputs(df)


if __name__ == "__main__":
    main()
