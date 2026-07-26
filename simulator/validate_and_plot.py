"""
validate_and_plot.py
--------------------
Post-simulation validation and plotting script.

Generates diagnostic plots:
  1. Full time-series of all key tags (2-hour window)
  2. Correlation heatmap — reveals the hidden speed→moisture coupling
  3. Grade-change event summary (off-spec distribution)
  4. Disturbance impact analysis
  5. Validation vs Training event distribution

Run after simulator.py:
    python validate_and_plot.py --output_dir ./output
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
def load_outputs(output_dir: Path):
    df       = pd.read_csv(output_dir / "process_data.csv", parse_dates=["timestamp"])
    evt_log  = pd.read_csv(output_dir / "grade_change_log.csv")
    op_log   = pd.read_csv(output_dir / "operator_action_log.csv")
    return df, evt_log, op_log


# ---------------------------------------------------------------------------
def plot_timeseries(df: pd.DataFrame, output_dir: Path,
                    window_hours: float = 6.0):
    """Plot a representative time window of all key process tags."""
    w = int(window_hours * 3600)
    # Pick the window around the first grade change with a disturbance
    start = df["elapsed_s"].iloc[0]
    # Try to find an interesting window (first event with disturbance)
    dist_rows = df[df["is_disturbance_active"] == 1]
    if not dist_rows.empty:
        center = dist_rows["elapsed_s"].iloc[0]
        start = max(0, center - w // 4)

    sub = df[(df["elapsed_s"] >= start) & (df["elapsed_s"] < start + w)].copy()
    t = sub["elapsed_s"] - start  # relative seconds

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Paper Machine Simulator — Process Time-Series\n"
                 f"(6-hour window from elapsed {start//3600:.1f}h)",
                 fontsize=13, fontweight="bold")
    gs = GridSpec(5, 1, figure=fig, hspace=0.55)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, sub["basis_weight"], color="#2196F3", lw=0.6, label="Basis Weight (gsm)")
    ax1.plot(t, sub["bw_setpoint"],  color="#F44336", lw=1.2, ls="--", label="BW Setpoint")
    ax1.fill_between(t,
        sub["bw_setpoint"] * 0.975, sub["bw_setpoint"] * 1.025,
        alpha=0.12, color="#F44336", label="±2.5% band")
    _shade_events(ax1, sub, t)
    ax1.set_ylabel("Basis Weight [gsm]")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("Basis Weight", fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, sub["moisture"],        color="#009688", lw=0.6, label="Moisture (%)")
    ax2.plot(t, sub["moisture_target"], color="#FF5722", lw=1.2, ls="--", label="Target")
    _shade_events(ax2, sub, t)
    ax2.set_ylabel("Moisture [%]")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_title("Moisture  ← Steam pressure + HIDDEN speed coupling", fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, sub["machine_speed"],    color="#9C27B0", lw=0.7, label="Speed (m/min)")
    ax3.plot(t, sub["machine_speed_sp"], color="#673AB7", lw=1.2, ls="--", label="Speed SP")
    _shade_events(ax3, sub, t)
    ax3.set_ylabel("Machine Speed [m/min]")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_title("Machine Speed", fontsize=9)

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t, sub["steam_pressure"],    color="#FF9800", lw=0.7, label="Steam Press (bar)")
    ax4.plot(t, sub["steam_pressure_sp"], color="#E65100", lw=1.2, ls="--", label="SP")
    _shade_events(ax4, sub, t)
    ax4.set_ylabel("Steam Pressure [bar]")
    ax4.legend(fontsize=7, loc="upper right")
    ax4.set_title("Steam Pressure", fontsize=9)

    ax5 = fig.add_subplot(gs[4])
    ax5.plot(t, sub["ash"],        color="#795548", lw=0.6, label="Ash (%)")
    ax5.plot(t, sub["ash_target"], color="#4E342E", lw=1.2, ls="--", label="Target")
    _shade_events(ax5, sub, t)
    ax5.set_ylabel("Ash [%]")
    ax5.set_xlabel("Relative time [s]")
    ax5.legend(fontsize=7, loc="upper right")
    ax5.set_title("Ash Content", fontsize=9)

    plt.savefig(output_dir / "timeseries_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] timeseries_plot.png saved.")


def _shade_events(ax, sub, t):
    """Shade grade-change windows and disturbance windows."""
    gc_mask = sub["in_grade_change"].values == 1
    dist_mask = sub["is_disturbance_active"].values == 1
    _shade_mask(ax, t, gc_mask, "#BBDEFB", 0.25)
    _shade_mask(ax, t, dist_mask, "#FFCDD2", 0.35)


def _shade_mask(ax, t, mask, color, alpha):
    in_region = False
    start_t = None
    t_arr = t.values if hasattr(t, "values") else np.array(t)
    for i, (ti, mi) in enumerate(zip(t_arr, mask)):
        if mi and not in_region:
            start_t = ti
            in_region = True
        elif not mi and in_region:
            ax.axvspan(start_t, ti, color=color, alpha=alpha)
            in_region = False
    if in_region:
        ax.axvspan(start_t, t_arr[-1], color=color, alpha=alpha)


# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Correlation heatmap — speed→moisture hidden coupling should be visible."""
    numeric_cols = [
        "stock_flow", "steam_pressure", "machine_speed", "filler_flow",
        "basis_weight", "moisture", "ash", "caliper",
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(numeric_cols, fontsize=9)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")

    ax.set_title("Process Variable Correlation Matrix\n"
                 "(Note: machine_speed ↔ moisture — hidden coupling!)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] correlation_heatmap.png saved.")


# ---------------------------------------------------------------------------
def plot_event_summary(evt_log: pd.DataFrame, output_dir: Path):
    """Grade-change event summary with off-spec and validation flags."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Grade-Change Event Analysis", fontsize=13, fontweight="bold")

    # 1. Off-spec breakdown
    ax = axes[0]
    labels = ["On-Spec", "Off-Spec"]
    vals   = [
        (~evt_log["went_off_spec"]).sum(),
        evt_log["went_off_spec"].sum(),
    ]
    ax.pie(vals, labels=labels, autopct="%1.0f%%",
           colors=["#4CAF50", "#F44336"], startangle=90)
    ax.set_title("Transition Outcome")

    # 2. Disturbance type distribution
    ax = axes[1]
    dist_counts = evt_log["disturbance"].value_counts()
    ax.bar(dist_counts.index, dist_counts.values,
           color=["#9E9E9E", "#FF9800", "#9C27B0", "#F44336"])
    ax.set_title("Disturbance Types")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    # 3. BW setpoint range
    ax = axes[2]
    ax.scatter(
        evt_log["old_bw_sp"], evt_log["new_bw_sp"],
        c=evt_log["went_off_spec"].map({True: "#F44336", False: "#4CAF50"}),
        s=60, alpha=0.7, edgecolors="k", lw=0.4,
    )
    lim = [40, 95]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4, label="No change")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Old BW [gsm]"); ax.set_ylabel("New BW [gsm]")
    ax.set_title("Grade Transition Map")
    green_patch = mpatches.Patch(color="#4CAF50", label="On-spec")
    red_patch   = mpatches.Patch(color="#F44336", label="Off-spec")
    ax.legend(handles=[green_patch, red_patch], fontsize=8)

    # Mark validation events
    val = evt_log[evt_log["is_validation"]]
    ax.scatter(val["old_bw_sp"], val["new_bw_sp"],
               marker="D", s=80, facecolors="none",
               edgecolors="navy", lw=1.5, label="Validation set", zorder=5)
    ax.legend(handles=[green_patch, red_patch,
              mpatches.Patch(facecolor="none", edgecolor="navy", label="Validation")],
              fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "event_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] event_summary.png saved.")


# ---------------------------------------------------------------------------
def plot_hidden_coupling(df: pd.DataFrame, output_dir: Path):
    """
    Visualize the hidden speed→moisture coupling by lagged cross-correlation.
    At ~90s lag, the correlation should peak — this is the 'secret' relationship.
    """
    from scipy import signal

    # Downsample to every 10s for efficiency
    sub = df.iloc[::10].copy()
    spd = sub["machine_speed"].values - sub["machine_speed"].mean()
    moist = sub["moisture"].values - sub["moisture"].mean()

    max_lag_s = 200
    max_lag_samples = max_lag_s // 10

    xcorr = signal.correlate(moist, spd, mode="full")
    lags = signal.correlation_lags(len(moist), len(spd), mode="full")
    # Normalize
    xcorr /= (np.std(spd) * np.std(moist) * len(spd))
    center = len(lags) // 2

    lags_s = lags[center - max_lag_samples: center + max_lag_samples + 1] * 10
    xcorr_w = xcorr[center - max_lag_samples: center + max_lag_samples + 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags_s, xcorr_w, color="#9C27B0", lw=1.5)
    ax.axvline(90, color="#F44336", ls="--", lw=1.5, label="Expected peak at 90s")
    peak_lag = lags_s[np.argmax(xcorr_w)]
    ax.axvline(peak_lag, color="#4CAF50", ls=":", lw=1.5,
               label=f"Actual peak at {peak_lag:.0f}s")
    ax.set_xlabel("Lag [s]  (positive = speed leads moisture)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Hidden Coupling: Machine Speed → Moisture\n"
                 "Cross-correlation reveals the undocumented 90s lag",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hidden_coupling_xcorr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] hidden_coupling_xcorr.png saved.")


# ---------------------------------------------------------------------------
def print_validation_set(evt_log: pd.DataFrame):
    """Print held-out validation events."""
    val = evt_log[evt_log["is_validation"]]
    train = evt_log[~evt_log["is_validation"]]
    print("\n" + "="*60)
    print("VALIDATION SET (held out — do not use for training)")
    print("="*60)
    print(val[["event_id","start_timestamp","old_grade","new_grade",
               "disturbance","went_off_spec"]].to_string(index=False))
    print(f"\nTraining events: {len(train)}  |  Validation events: {len(val)}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()
    out = Path(args.output_dir)

    print("Loading simulation outputs...")
    df, evt_log, op_log = load_outputs(out)
    print(f"  Time-series: {len(df):,} rows")
    print(f"  Events     : {len(evt_log)} grade changes")
    print(f"  Op log     : {len(op_log)} actions")

    plot_timeseries(df, out)
    plot_correlation_heatmap(df, out)
    plot_event_summary(evt_log, out)
    plot_hidden_coupling(df, out)
    print_validation_set(evt_log)

    print("\nAll validation plots saved to:", out.resolve())


if __name__ == "__main__":
    main()
