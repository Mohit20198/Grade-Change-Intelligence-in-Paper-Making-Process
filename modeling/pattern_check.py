"""
pattern_check.py
----------------
Fix 1: Rename true_positive -> early_warning in lead_time CSV
Fix 2: Pattern check on 3 late-detection events (9, 12, 45)
       - "near-miss" = risk crossed >50% before final flag, then dropped, then re-crossed
       - "no signal"  = never exceeded 50% until at/after breach

Reports per-event risk score trace for events 9, 12, 45 and saves corrected CSV.
"""
import pandas as pd
import numpy as np
import joblib

print("Loading model and data...")
clf = joblib.load('./output/lgbm_classifier_ablated.pkl')
feature_cols = clf.feature_name()

df_raw = pd.read_csv('./output/process_data.csv', low_memory=False)
events = pd.read_csv('./output/grade_change_log.csv')

# Precompute rolling features
raw_tags = ['stock_flow', 'steam_pressure', 'machine_speed',
            'filler_flow', 'basis_weight', 'moisture', 'ash']
raw_cols = [c for c in (raw_tags + ['caliper']) if c in df_raw.columns]
feat = df_raw[raw_cols].copy()
for tag in raw_tags:
    s = df_raw[tag]
    feat[f'{tag}_lag_1s']        = s.shift(1)
    feat[f'{tag}_lag_5s']        = s.shift(5)
    feat[f'{tag}_lag_15s']       = s.shift(15)
    feat[f'{tag}_roll_mean_5m']  = s.rolling(300, min_periods=1).mean()
    feat[f'{tag}_roll_std_5m']   = s.rolling(300, min_periods=1).std().fillna(0)
    feat[f'{tag}_roll_mean_15m'] = s.rolling(900, min_periods=1).mean()
    feat[f'{tag}_roll_std_15m']  = s.rolling(900, min_periods=1).std().fillna(0)
    feat[f'{tag}_roc_1s']        = s.diff(1)
    feat[f'{tag}_roc_5s']        = s.diff(5)
for c in [x for x in feature_cols if x not in feat.columns]:
    feat[c] = 0.0

# -----------------------------------------------------------------------
# Fix 1: reload lead_time CSV and rename column
# -----------------------------------------------------------------------
lead_df = pd.read_csv('./output/lead_time_300s_results.csv')
if 'true_positive' in lead_df.columns:
    lead_df['early_warning'] = lead_df['true_positive']
    lead_df = lead_df.drop(columns=['true_positive'])
lead_df.to_csv('./output/lead_time_300s_results.csv', index=False)
print("Renamed 'true_positive' -> 'early_warning' in lead_time_300s_results.csv")

# -----------------------------------------------------------------------
# Fix 2: Pattern check for late-detection events 9, 12, 45
# -----------------------------------------------------------------------
late_events = [9, 12, 45]
off_spec = events[events['went_off_spec'] == True].copy()

for evt_id in late_events:
    evt_info = events[events['event_id'] == evt_id].iloc[0]
    bw_sp    = evt_info['new_bw_sp']
    start_s  = int(evt_info['start_elapsed_s'])
    settle_s = int(evt_info['settle_elapsed_s'])
    lower    = bw_sp * 0.975
    upper    = bw_sp * 1.025

    # Find actual breach time
    event_bw    = df_raw.loc[start_s:settle_s, 'basis_weight']
    out_of_band = event_bw[(event_bw < lower) | (event_bw > upper)]
    breach_s    = int(out_of_band.index[0]) if len(out_of_band) else settle_s

    # Scan every 30s from start_s-300 to settle_s — full risk trace
    scan_start = max(0, start_s - 300)
    risk_trace = []
    for t in range(scan_start, min(settle_s, breach_s + 120), 30):
        if t not in feat.index:
            continue
        row  = feat.loc[[t], feature_cols]
        risk = float(clf.predict(row)[0])
        risk_trace.append((t, risk))

    # Find crossings above 50%
    above_50 = [(t, r) for t, r in risk_trace if r > 0.5]
    before_breach = [(t, r) for t, r in above_50 if t < breach_s]
    at_or_after   = [(t, r) for t, r in above_50 if t >= breach_s]

    print(f"\n--- Event {evt_id} ({evt_info['disturbance']}) ---")
    print(f"  Breach at: {breach_s}s")
    print(f"  Risk trace (every 30s from {scan_start}s to breach+120s):")
    for t, r in risk_trace:
        marker = " <<< BREACH" if t == breach_s else ("  [>50%]" if r > 0.5 else "")
        print(f"    t={t:7d}s  risk={r:.3f}{marker}")

    if before_breach:
        # Check if there's a gap (risk dropped below 50% between first crossing and breach)
        first_cross_t = before_breach[0][0]
        between = [(t, r) for t, r in risk_trace
                   if first_cross_t < t < breach_s and r <= 0.5]
        if between:
            pattern = "NEAR-MISS: crossed >50% early, dropped back, re-triggered near breach"
        else:
            pattern = "SUSTAINED: stayed >50% from first crossing to breach"
        print(f"  Pattern: {pattern}")
        print(f"  First >50% crossing: t={first_cross_t}s ({(breach_s - first_cross_t)/60:.1f} min before breach)")
    else:
        pattern = "NO SIGNAL: risk never exceeded 50% before breach"
        print(f"  Pattern: {pattern}")
