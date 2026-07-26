import pandas as pd
import numpy as np
import json
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
    
    lower_bound = target_bw * 0.975
    upper_bound = target_bw * 1.025
    
    # 1. Find flag_s (when risk > 50%)
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
        
    print(f"Event {evt_id}: risk flagged at {flag_s}.")
    
    # 2. Get recommendation at flag_s
    feat_flag = get_features_for_index(flag_s)
    rec = re.generate_recommendation(feat_flag, target_bw, current_df_raw_index=flag_s)
    
    action_str = rec['recommended_action']
    parts = action_str.split(' ')
    
    sf_diff, sp_diff, ms_diff = 0.0, 0.0, 0.0
    if len(parts) >= 4:
        direction = parts[0]
        mv = parts[1]
        pct = float(parts[3].replace('%', ''))
        
        cur_val = df_raw.loc[flag_s, mv]
        diff_val = cur_val * (pct/100.0 if direction == 'Increase' else -pct/100.0)
        
        if mv == 'stock_flow': sf_diff = diff_val
        if mv == 'steam_pressure': sp_diff = diff_val
        if mv == 'machine_speed': ms_diff = diff_val

    # Helper function to run simulator and find settle time relative to flag_s
    def get_settle_time(overrides_list):
        sim = PaperMachineSimulator(seed=42)
        sim.overrides = overrides_list
        sim.start_recording = flag_s
        proj_df = sim.run()
        
        # Look at window from flag_s to end of simulation
        # proj_df now only contains data from flag_s onwards.
        # But wait, proj_df index might not match elapsed_s directly if we just append to records.
        # It's better to just use the raw DataFrame values since we know it starts at flag_s.
        pred_bw = proj_df['basis_weight'].values
        in_band = (pred_bw >= lower_bound) & (pred_bw <= upper_bound)
        
        proj_settle_idx = -1
        for i in range(len(in_band)-30):
            if all(in_band[i:i+30]):
                proj_settle_idx = i
                break
        
        if proj_settle_idx == -1:
            return 2000 # Max window
        return proj_settle_idx

    # 3a. BASELINE FORK (no overrides)
    # Simulator will run exactly as it did historically. Disturbances apply naturally.
    print(f"Event {evt_id}: running BASELINE fork...")
    baseline_time = get_settle_time([{
        'start_t': flag_s,
        'end_t': flag_s + 2000,
        'ramp_dur': 180,
        'sf_diff': 0.0,
        'sp_diff': 0.0,
        'ms_diff': 0.0
    }])
    
    # 3b. WITH RECOMMENDATION FORK
    print(f"Event {evt_id}: running RECOMMENDATION fork...")
    rec_time = get_settle_time([{
        'start_t': flag_s,
        'end_t': flag_s + 2000,
        'ramp_dur': 180,
        'sf_diff': sf_diff,
        'sp_diff': sp_diff,
        'ms_diff': ms_diff
    }])
    
    absolute_reduction = baseline_time - rec_time
    pct_reduction = (absolute_reduction / baseline_time) * 100 if baseline_time > 0 else 0
    
    results.append({
        'event_id': evt_id,
        'disturbance': evt['disturbance'],
        'baseline_time': baseline_time,
        'rec_time': rec_time,
        'absolute_reduction': absolute_reduction,
        'reduction_pct': pct_reduction
    })
    
res_df = pd.DataFrame(results)
print("\n--- Apples-to-Apples Stabilization Time ---")
print(res_df.to_string(index=False))

avg_red = res_df['reduction_pct'].mean()
med_red = res_df['reduction_pct'].median()
print(f"\nAverage Stabilization Time Reduction: {avg_red:.1f}%")
print(f"Median Stabilization Time Reduction: {med_red:.1f}%")

res_df.to_csv('output/stabilization_results_apples.csv', index=False)
with open('output/dashboard_data.json', 'w') as f:
    json.dump({"avg_reduction": avg_red, "median_reduction": med_red}, f)
