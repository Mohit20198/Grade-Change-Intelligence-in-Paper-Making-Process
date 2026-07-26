import pandas as pd
import numpy as np
import json
import joblib
import recommendation_engine as re
from simulator import PaperMachineSimulator

df_raw = pd.read_csv('output/process_data.csv', low_memory=False)
events = pd.read_csv('output/grade_change_log.csv')
off_spec = events[events['went_off_spec'] == True]

clf_model = re.clf_model
allowed_cols = clf_model.feature_name()

def get_features_for_index(idx):
    return re.get_features_for_index(idx)

results = []

for _, evt in off_spec.iterrows():
    evt_id = evt['event_id']
    target_bw = evt['new_bw_sp']
    start_s = evt['start_elapsed_s']
    settle_s = evt['settle_elapsed_s']
    
    # 1. Historical stabilization time
    lower_bound = target_bw * 0.975
    upper_bound = target_bw * 1.025
    
    hist_window = df_raw.loc[start_s:settle_s]
    exceeded = (hist_window['basis_weight'] < lower_bound) | (hist_window['basis_weight'] > upper_bound)
    
    if not exceeded.any():
        continue
        
    first_exceed_s = exceeded.idxmax()
    hist_stab_time = settle_s - first_exceed_s
    
    # 2. Find when risk was first flagged (>50%)
    flag_s = None
    for t in range(int(start_s), int(settle_s), 60):
        feat = get_features_for_index(t)
        if feat is None:
            continue
        risk = clf_model.predict(feat)[0]
        if risk > 0.5:
            flag_s = t
            break
            
    if flag_s is None:
        print(f"Event {evt_id}: Risk never flagged >50%")
        continue
        
    # 3. Generate recommendation at flag_s
    print(f"Event {evt_id}: risk flagged at {flag_s}. Generating recommendation...")
    feat_flag = get_features_for_index(flag_s)
    rec = re.generate_recommendation(feat_flag, target_bw, current_df_raw_index=flag_s)
    
    action_str = rec['recommended_action']
    parts = action_str.split(' ')
    
    sf_diff = 0.0
    sp_diff = 0.0
    ms_diff = 0.0
    
    if len(parts) >= 4:
        direction = parts[0]
        mv = parts[1]
        pct = float(parts[3].replace('%', ''))
        
        cur_val = df_raw.loc[flag_s, mv]
        diff_val = cur_val * (pct/100.0 if direction == 'Increase' else -pct/100.0)
        
        if mv == 'stock_flow': sf_diff = diff_val
        if mv == 'steam_pressure': sp_diff = diff_val
        if mv == 'machine_speed': ms_diff = diff_val

    # 4. Run Physical Simulator with Overrides
    print(f"Event {evt_id}: running physical simulator...")
    sim = PaperMachineSimulator(seed=42)
    sim.overrides = [{
        'start_t': flag_s,
        'end_t': flag_s + 1500,
        'ramp_dur': 180,
        'sf_diff': sf_diff,
        'sp_diff': sp_diff,
        'ms_diff': ms_diff
    }]
    
    # To save time, we can run simulator only up to flag_s + 1500
    # but we need it to run the event. Let's just run it! 
    # It takes 2 seconds for the whole 8 days.
    proj_df = sim.run()
    
    # Extract the projected window
    proj_window = proj_df.loc[flag_s:flag_s + 1500]
    
    # Find when it settles for 30s
    pred_bw = proj_window['basis_weight'].values
    in_band = (pred_bw >= lower_bound) & (pred_bw <= upper_bound)
    
    proj_settle_idx = -1
    for i in range(len(in_band)-30):
        if all(in_band[i:i+30]):
            proj_settle_idx = i
            break
            
    if proj_settle_idx == -1:
        proj_stab_time = 1500
    else:
        proj_stab_time = (flag_s - first_exceed_s) + proj_settle_idx
        
    pct_reduction = (hist_stab_time - proj_stab_time) / hist_stab_time * 100
    
    results.append({
        'event_id': evt_id,
        'disturbance': evt['disturbance'],
        'hist_stab_time': hist_stab_time,
        'proj_stab_time': proj_stab_time,
        'reduction_pct': pct_reduction
    })
    
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

avg_red = res_df['reduction_pct'].mean()
med_red = res_df['reduction_pct'].median()
print(f"\nAverage Stabilization Time Reduction: {avg_red:.1f}%")
print(f"Median Stabilization Time Reduction: {med_red:.1f}%")

res_df.to_csv('output/stabilization_results.csv', index=False)
with open('output/dashboard_data.json', 'w') as f:
    json.dump({"avg_reduction": avg_red, "median_reduction": med_red}, f)
