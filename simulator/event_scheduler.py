"""
event_scheduler.py
------------------
Grade-change event scheduler and disturbance injector for the paper-making
process simulator.

Responsibilities
----------------
1. Schedule random grade-change events across the simulation horizon.
2. Generate coordinated ramp trajectories for all manipulated variables.
3. Inject disturbances on ~30-40% of transitions.
4. Produce an operator action log.
5. Mark ~20% of events as validation-set (held out).
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GradeSpec:
    """Steady-state operating recipe for one paper grade."""
    name: str
    bw_setpoint: float          # Basis Weight [gsm]
    stock_flow: float           # [L/min]
    steam_pressure: float       # [bar]
    machine_speed: float        # [m/min]
    filler_flow: float          # [kg/min]
    moisture_target: float      # [%]
    ash_target: float           # [%]


@dataclass
class GradeChangeEvent:
    """One grade-change transition."""
    event_id: int
    start_ts: int               # seconds from simulation start
    end_ts: int                 # end of ramp window
    settle_ts: int              # expected settle time
    old_grade: str
    new_grade: str
    old_spec: GradeSpec
    new_spec: GradeSpec
    ramp_duration: int          # [s]
    disturbance: Optional[str]  # None | 'steam_sag' | 'speed_hunting' | 'sensor_spike'
    disturbance_start: Optional[int]
    is_validation: bool = False
    went_off_spec: bool = False # filled in post-simulation


@dataclass
class OperatorAction:
    """One row in the operator action log."""
    timestamp: int              # seconds from simulation start
    tag: str
    old_value: float
    new_value: float
    note: str
    event_id: int


# ---------------------------------------------------------------------------
# Grade library
# ---------------------------------------------------------------------------

GRADE_LIBRARY: List[GradeSpec] = [
    GradeSpec("G45",  45.0,  750.0, 4.0, 950.0, 140.0, 5.5, 10.0),
    GradeSpec("G52",  52.0,  870.0, 4.2, 920.0, 155.0, 5.8, 11.0),
    GradeSpec("G60",  60.0, 1000.0, 4.5, 880.0, 175.0, 6.0, 12.0),
    GradeSpec("G70",  70.0, 1160.0, 4.8, 840.0, 195.0, 6.3, 13.0),
    GradeSpec("G80",  80.0, 1320.0, 5.1, 800.0, 215.0, 6.6, 14.0),
    GradeSpec("G90",  90.0, 1500.0, 5.4, 760.0, 235.0, 7.0, 15.0),
]

DISTURBANCE_TYPES = ["steam_sag", "speed_hunting", "sensor_spike", "stock_surge"]

# Operator note templates keyed by action tag
NOTE_TEMPLATES = {
    "stock_flow": [
        "Ramped stock flow from {old:.0f} to {new:.0f} L/min for grade change to {grade}",
        "Adjusted stock flow to match new basis weight target {bw:.1f} gsm",
    ],
    "steam_pressure": [
        "Increased steam to counter moisture dip during grade change",
        "Ramped steam pressure from {old:.2f} to {new:.2f} bar",
        "Steam pressure adjustment for dryer section — targeting {grade}",
    ],
    "machine_speed": [
        "Speed ramp from {old:.0f} to {new:.0f} m/min for grade change",
        "Adjusted machine speed to maintain reel tension on grade {grade}",
    ],
    "filler_flow": [
        "Filler flow adjusted from {old:.0f} to {new:.0f} kg/min",
        "Ash target change — filler flow set for grade {grade}",
    ],
    "steam_pressure_disturbance": [
        "Steam pressure sag detected — investigating header pressure",
        "Compensating for steam pressure drop, boosting setpoint",
        "Alerted DCS operator: steam sag during grade transition",
    ],
    "speed_hunting": [
        "Speed oscillation observed — checking draw section",
        "Notified mechanical: speed hunting on machine",
        "Reduced speed ramp rate to dampen oscillation",
    ],
    "sensor_spike": [
        "Basis weight scanner spike — cross-checking with caliper",
        "Noise spike on BW sensor, possible scan head issue",
        "Flagged scanner anomaly, verifying with lab sample",
    ],
    "stock_surge": [
        "Stock flow surge detected — checking headbox consistency valve",
        "BW deviation observed: stock flow instability, investigating pulp pump",
        "Reducing stock flow target temporarily to stabilize basis weight",
    ],
}


def _pick_note(tag: str, old_val: float, new_val: float,
               grade_name: str, bw_sp: float) -> str:
    templates = NOTE_TEMPLATES.get(tag, ["Manual adjustment on {tag}"])
    tmpl = random.choice(templates)
    return tmpl.format(
        old=old_val, new=new_val,
        grade=grade_name, bw=bw_sp, tag=tag
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class EventScheduler:
    """
    Schedules grade-change events and disturbances over a simulation horizon.

    Parameters
    ----------
    total_seconds      : total simulation duration [s]
    seed               : random seed for reproducibility
    min_gap_hours      : minimum gap between grade changes [h]
    max_gap_hours      : maximum gap between grade changes [h]
    disturbance_prob   : probability of disturbance per transition
    validation_fraction: fraction of events held out for validation
    settle_window_s    : how long after ramp end we expect stabilization [s]
    """

    def __init__(self,
                 total_seconds: int = 3 * 24 * 3600,
                 seed: int = 42,
                 min_gap_hours: float = 2.5,
                 max_gap_hours: float = 5.0,
                 disturbance_prob: float = 0.35,
                 validation_fraction: float = 0.20,
                 settle_window_s: int = 1200):

        self.total_seconds = total_seconds
        self.seed = seed
        self.min_gap = int(min_gap_hours * 3600)
        self.max_gap = int(max_gap_hours * 3600)
        self.disturbance_prob = disturbance_prob
        self.validation_fraction = validation_fraction
        self.settle_window_s = settle_window_s

        random.seed(seed)
        np.random.seed(seed)

        self.events: List[GradeChangeEvent] = []
        self.operator_log: List[OperatorAction] = []

    # ------------------------------------------------------------------
    def build_events(self) -> List[GradeChangeEvent]:
        """Generate the full event schedule."""
        current_spec = random.choice(GRADE_LIBRARY)
        t = self.min_gap  # first event not at t=0

        event_id = 0
        while t < self.total_seconds - self.max_gap:
            # Choose a different grade
            candidates = [g for g in GRADE_LIBRARY if g.name != current_spec.name]
            new_spec = random.choice(candidates)

            ramp_duration = random.randint(60, 180)   # 1–3 minutes
            end_ts = t + ramp_duration
            settle_ts = end_ts + self.settle_window_s

            # Disturbance?
            do_disturb = random.random() < self.disturbance_prob
            dist_type = random.choice(DISTURBANCE_TYPES) if do_disturb else None
            # Strike during the stabilization phase
            dist_start = (end_ts + random.randint(60, 300)
                          if do_disturb else None)

            evt = GradeChangeEvent(
                event_id=event_id,
                start_ts=t,
                end_ts=end_ts,
                settle_ts=settle_ts,
                old_grade=current_spec.name,
                new_grade=new_spec.name,
                old_spec=current_spec,
                new_spec=new_spec,
                ramp_duration=ramp_duration,
                disturbance=dist_type,
                disturbance_start=dist_start,
            )
            self.events.append(evt)

            # Generate operator actions for this event
            self._log_operator_actions(evt, event_id)

            current_spec = new_spec
            gap = random.randint(self.min_gap, self.max_gap)
            t = end_ts + gap
            event_id += 1

        # Mark validation set (stratified: every Nth)
        if self.events:
            n_val = max(1, int(len(self.events) * self.validation_fraction))
            val_indices = set(
                np.linspace(0, len(self.events) - 1, n_val, dtype=int).tolist()
            )
            for idx in val_indices:
                self.events[idx].is_validation = True

        return self.events

    # ------------------------------------------------------------------
    def _log_operator_actions(self, evt: GradeChangeEvent, eid: int):
        """Record operator actions for a grade-change event."""
        t = evt.start_ts
        tags = {
            "stock_flow":    (evt.old_spec.stock_flow, evt.new_spec.stock_flow),
            "steam_pressure":(evt.old_spec.steam_pressure, evt.new_spec.steam_pressure),
            "machine_speed": (evt.old_spec.machine_speed, evt.new_spec.machine_speed),
            "filler_flow":   (evt.old_spec.filler_flow, evt.new_spec.filler_flow),
        }
        for i, (tag, (old_v, new_v)) in enumerate(tags.items()):
            if abs(old_v - new_v) < 1e-6:
                continue
            note = _pick_note(tag, old_v, new_v,
                              evt.new_grade, evt.new_spec.bw_setpoint)
            self.operator_log.append(OperatorAction(
                timestamp=t + i * 2,   # slight stagger for realism
                tag=tag,
                old_value=round(old_v, 3),
                new_value=round(new_v, 3),
                note=note,
                event_id=eid,
            ))

        # Disturbance action
        if evt.disturbance and evt.disturbance_start:
            tag_key = (evt.disturbance if evt.disturbance in NOTE_TEMPLATES
                       else evt.disturbance)
            note = random.choice(NOTE_TEMPLATES.get(evt.disturbance, ["Disturbance noted"]))
            self.operator_log.append(OperatorAction(
                timestamp=evt.disturbance_start + random.randint(30, 90),
                tag=f"ALERT_{evt.disturbance.upper()}",
                old_value=0.0,
                new_value=1.0,
                note=note,
                event_id=eid,
            ))

    # ------------------------------------------------------------------
    def get_setpoints_at(self, t: int) -> Tuple[GradeSpec, Optional[GradeChangeEvent]]:
        """
        Return (active_spec, active_event_or_None) at time t.
        During a ramp, spec values are linearly interpolated.
        """
        active_event = None
        for evt in self.events:
            if evt.start_ts <= t <= evt.settle_ts + self.settle_window_s:
                active_event = evt
                break

        if active_event is None:
            # Find the most recent completed event's new_spec.
            # Before the first event, return that event's old_spec (the initial grade).
            if not self.events:
                return GRADE_LIBRARY[0], None

            if t < self.events[0].start_ts:
                return self.events[0].old_spec, None

            spec = self.events[0].old_spec
            for evt in self.events:
                if evt.end_ts <= t:
                    spec = evt.new_spec
                else:
                    break
            return spec, None

        evt = active_event
        if t < evt.start_ts:
            return evt.old_spec, None
        elif t <= evt.end_ts:
            # Linear ramp
            alpha = (t - evt.start_ts) / evt.ramp_duration
            ramped = _interpolate_specs(evt.old_spec, evt.new_spec, alpha)
            return ramped, evt
        else:
            return evt.new_spec, evt


# ------------------------------------------------------------------
def _interpolate_specs(old: GradeSpec, new: GradeSpec, alpha: float) -> GradeSpec:
    """Linear interpolation between two GradeSpecs (for ramp trajectory)."""
    def lerp(a, b):
        return a + alpha * (b - a)
    return GradeSpec(
        name=f"{old.name}->{new.name}",
        bw_setpoint=lerp(old.bw_setpoint, new.bw_setpoint),
        stock_flow=lerp(old.stock_flow, new.stock_flow),
        steam_pressure=lerp(old.steam_pressure, new.steam_pressure),
        machine_speed=lerp(old.machine_speed, new.machine_speed),
        filler_flow=lerp(old.filler_flow, new.filler_flow),
        moisture_target=lerp(old.moisture_target, new.moisture_target),
        ash_target=lerp(old.ash_target, new.ash_target),
    )
