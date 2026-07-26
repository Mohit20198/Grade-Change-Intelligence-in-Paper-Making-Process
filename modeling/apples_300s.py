"""
apples_300s.py
--------------
Apples-to-apples stabilization time comparison using the new 300s model.
For each of the 9 off-spec events:
  - Loads the flag timestamp from lead_time_300s_results.csv (flag computed by new model)
  - Runs two simulator forks from that SAME timestamp:
      A) Baseline: no action (natural trajectory)
      B) Recommendation: setpoint trim applied (from recommendation engine)
  - Reports per-event and aggregate reduction

Uses start_recording to avoid memory overflow.
"""
import pandas as pd
import numpy as np
import json
import joblib
from simulator import PaperMachineSimulator

print("Loading models and data...")
clf = joblib.load('./output/lgbm_classifier_ablated.pkl')
reg = joblib.load('./output/lgbm_regressor_ablated.pkl')
feature_cols = clf.feature_name()

df_raw = pd.read_csv('./output/process_data.csv', low_memory=False)
events = pd.read_csv('./output/grade_change_log.csv')
lead_df = pd.read_csv('./output/lead_time_300s_results.csv')
lead_tp = lead_df[lead_df['true_positive'] == True].copy()
print(f"Using {len(lead_tp)} true-positive events for comparison")

# Precompute rolling features inline (same as lead_time_300s.py)
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
for c in missing:
    feat[c] = 0.0

def get_recommendation_trim(flag_s, bw_sp):
    """
    Compute the MV trim using the regressor:
    - Predict BW at t+300s from current features
    - If predicted BW is below setpoint: increase stock_flow by 0.2%
    - If predicted BW is above setpoint: decrease stock_flow by 0.2%
    - If within band: no trim
    """
    row = feat.loc[[flag_s], feature_cols]
    pred_bw = reg.predict(row)[0]
    diff = pred_bw - bw_sp
    cur_sf = df_raw.loc[flag_s, 'stock_flow']
    trim = cur_sf * 0.002  # 0.2% of current stock_flow
    if diff < -bw_sp * 0.010:   # predicted more than 1% below SP
        return trim, 0.0, 0.0, f"Increase stock_flow by 0.2% ({trim:.2f} L/min)"
    elif diff > bw_sp * 0.010:  # predicted more than 1% above SP
        return -trim, 0.0, 0.0, f"Decrease stock_flow by 0.2% ({trim:.2f} L/min)"
    else:
        return 0.0, 0.0, 0.0, "No trim (BW within 1% of SP)"

def get_settle_time(flag_s, lower, upper, overrides_list):
    """Run simulator from flag_s; return seconds until basis_weight stays in band for 30 consecutive readings."""
    sim = PaperMachineSimulator(seed=42)
    sim.overrides = overrides_list
    sim.start_recording = flag_s
    proj_df = sim.run()
    
    pred_bw = proj_df['basis_weight'].values
    in_band = (pred_bw >= lower) & (pred_bw <= upper)
    
    for i in range(len(in_band) - 30):
        if all(in_band[i:i+30]):
            return i   # seconds relative to flag_s
    return 2000

results = []

for _, evt_row in lead_tp.iterrows():
    evt_id  = int(evt_row['event_id'])
    flag_s  = int(evt_row['flag_elapsed_s'])
    
    # Look up the event's BW setpoint
    evt_info = events[events['event_id'] == evt_id].iloc[0]
    bw_sp    = evt_info['new_bw_sp']
    lower    = bw_sp * 0.975
    upper    = bw_sp * 1.025
    
    print(f"\nEvent {evt_id} ({evt_row['disturbance']}) — flag at {flag_s}s (lead={evt_row['lead_time_min']} min)")

    # Get recommendation at flag_s using inline regressor-based trim
    sf_diff, sp_diff, ms_diff, action_str = get_recommendation_trim(flag_s, bw_sp)
    print(f"  Recommendation: {action_str}")

    # Baseline fork
    print(f"  Running BASELINE fork...")
    baseline_time = get_settle_time(flag_s, lower, upper, [{
        'start_t': flag_s, 'end_t': flag_s + 2000, 'ramp_dur': 180,
        'sf_diff': 0.0, 'sp_diff': 0.0, 'ms_diff': 0.0
    }])
    
    # Recommendation fork
    print(f"  Running RECOMMENDATION fork...")
    rec_time = get_settle_time(flag_s, lower, upper, [{
        'start_t': flag_s, 'end_t': flag_s + 2000, 'ramp_dur': 180,
        'sf_diff': sf_diff, 'sp_diff': sp_diff, 'ms_diff': ms_diff
    }])
    
    absolute_reduction = baseline_time - rec_time
    pct_reduction = (absolute_reduction / baseline_time) * 100 if baseline_time > 0 else 0
    
    print(f"  Baseline: {baseline_time}s | Rec: {rec_time}s | Reduction: {absolute_reduction}s ({pct_reduction:.1f}%)")
    
    results.append({
        'event_id':           evt_id,
        'disturbance':        evt_row['disturbance'],
        'lead_time_min':      evt_row['lead_time_min'],
        'flag_elapsed_s':     flag_s,
        'baseline_time_s':    baseline_time,
        'rec_time_s':         rec_time,
        'absolute_reduction': absolute_reduction,
        'reduction_pct':      pct_reduction,
    })

res_df = pd.DataFrame(results)
print("\n--- Apples-to-Apples (300s model) ---")
print(res_df[['event_id','disturbance','lead_time_min','baseline_time_s','rec_time_s','reduction_pct']].to_string(index=False))

avg_red = res_df['reduction_pct'].mean()
med_red = res_df['reduction_pct'].median()
print(f"\nAverage reduction : {avg_red:.1f}%")
print(f"Median  reduction : {med_red:.1f}%")

res_df.to_csv('./output/apples_300s_results.csv', index=False)

try:
    dash = json.load(open('./output/dashboard_data.json'))
except Exception:
    dash = {}
dash['stab_reduction_300s_avg'] = round(avg_red, 1)
dash['stab_reduction_300s_med'] = round(med_red, 1)
with open('./output/dashboard_data.json', 'w') as f:
    json.dump(dash, f, indent=2)

print("Saved apples_300s_results.csv and updated dashboard_data.json")
