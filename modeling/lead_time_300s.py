"""
lead_time_300s.py
-----------------
For each of the 9 off-spec validation events:
  1. Find the FIRST timestamp where the new 300s model's risk score > 50%
  2. Find the FIRST timestamp where basis_weight ACTUALLY breaches the ±2.5% band
  3. Lead time = breach_s - flag_s  (positive = early warning, negative = late)
  4. Report whether each flag was a TRUE or FALSE positive

Outputs per-event table + aggregate stats. Updates dashboard_data.json.
"""
import pandas as pd
import numpy as np
import json
import joblib

print("Loading 300s classifier...")
clf = joblib.load('./output/lgbm_classifier_ablated.pkl')
feature_cols = clf.feature_name()

print("Loading process data and events...")
df_raw = pd.read_csv('./output/process_data.csv', low_memory=False)
events = pd.read_csv('./output/grade_change_log.csv')
off_spec = events[events['went_off_spec'] == True].copy()

print(f"Precomputing rolling features on {len(df_raw):,} rows...")
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

missing = [c for c in feature_cols if c not in feat.columns]
if missing:
    for c in missing:
        feat[c] = 0.0

print("Scanning each off-spec event for first >50% flag...")
results = []

for _, evt in off_spec.iterrows():
    evt_id   = int(evt['event_id'])
    bw_sp    = evt['new_bw_sp']
    start_s  = int(evt['start_elapsed_s'])
    settle_s = int(evt['settle_elapsed_s'])

    lower = bw_sp * 0.975
    upper = bw_sp * 1.025

    # Extend search BEFORE the grade-change start by up to 300s
    scan_start = max(0, start_s - 300)

    # 1. Find first risk > 50% (scan every 30s for finer resolution)
    flag_s    = None
    flag_risk = None
    for t in range(scan_start, settle_s, 30):
        if t not in feat.index:
            continue
        row = feat.loc[[t], feature_cols]
        risk = clf.predict(row)[0]
        if risk > 0.5:
            flag_s = t
            flag_risk = round(float(risk), 3)
            break

    if flag_s is None:
        print(f"  Event {evt_id}: risk never exceeded 50%")
        continue

    # 2. Find first actual band breach within event window
    event_bw = df_raw.loc[start_s:settle_s, 'basis_weight']
    out_of_band = event_bw[(event_bw < lower) | (event_bw > upper)]

    if len(out_of_band) == 0:
        print(f"  Event {evt_id}: basis_weight never left band (shouldn't happen for off-spec events)")
        continue

    breach_s  = int(out_of_band.index[0])
    lead_time = breach_s - flag_s
    is_tp     = flag_s <= breach_s  # True positive: flag fires at or before breach

    print(f"  Event {evt_id:2d} ({evt['disturbance']:15s}): "
          f"flag={flag_s}s (risk={flag_risk}), breach={breach_s}s => "
          f"lead={lead_time}s ({lead_time/60:.1f}min)  {'TP' if is_tp else 'FP'}")

    results.append({
        'event_id':        evt_id,
        'disturbance':     evt['disturbance'],
        'flag_elapsed_s':  flag_s,
        'flag_risk':       flag_risk,
        'breach_elapsed_s':breach_s,
        'lead_time_s':     lead_time,
        'lead_time_min':   round(lead_time / 60, 1),
        'true_positive':   is_tp,
    })

res_df = pd.DataFrame(results)
print("\n--- Lead Time Distribution (300s model) ---")
print(res_df[['event_id','disturbance','lead_time_s','lead_time_min','true_positive']].to_string(index=False))

tp_df = res_df[res_df['true_positive']]
print(f"\n  True positives : {len(tp_df)}/{len(res_df)}")
print(f"  False positives: {len(res_df)-len(tp_df)}/{len(res_df)}")

if len(tp_df):
    print(f"\n  Lead time (TP only):")
    print(f"    Mean   : {tp_df['lead_time_s'].mean():.0f}s  ({tp_df['lead_time_s'].mean()/60:.1f} min)")
    print(f"    Median : {tp_df['lead_time_s'].median():.0f}s  ({tp_df['lead_time_s'].median()/60:.1f} min)")
    print(f"    Min    : {tp_df['lead_time_s'].min():.0f}s  ({tp_df['lead_time_s'].min()/60:.1f} min)")
    print(f"    Max    : {tp_df['lead_time_s'].max():.0f}s  ({tp_df['lead_time_s'].max()/60:.1f} min)")
else:
    print("\n  No true positives — all flags fire after breach.")

res_df.to_csv('./output/lead_time_300s_results.csv', index=False)

# Update dashboard_data.json
try:
    dash = json.load(open('./output/dashboard_data.json'))
except Exception:
    dash = {}

dash['lead_time_300s_mean_s']   = round(tp_df['lead_time_s'].mean(), 1) if len(tp_df) else 0
dash['lead_time_300s_median_s'] = round(tp_df['lead_time_s'].median(), 1) if len(tp_df) else 0
dash['lead_time_300s_mean_min'] = round(tp_df['lead_time_s'].mean() / 60, 1) if len(tp_df) else 0
dash['lead_time_300s_n_tp']     = len(tp_df)
dash['lead_time_300s_n_total']  = len(res_df)

with open('./output/dashboard_data.json', 'w') as f:
    json.dump(dash, f, indent=2)

print("\nSaved lead_time_300s_results.csv and updated dashboard_data.json")
