import pandas as pd
import numpy as np
import joblib
import json

import recommendation_engine as re

df_raw = pd.read_csv('output/process_data.csv', low_memory=False)
events = pd.read_csv('output/grade_change_log.csv')
off_spec = events[events['went_off_spec'] == True]

clf_model = re.clf_model
reg_model = re.reg_model
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
    # Find when it first exceeded +/- 2.5%
    lower_bound = target_bw * 0.975
    upper_bound = target_bw * 1.025
    
    # scan from start_s to settle_s
    hist_window = df_raw.loc[start_s:settle_s]
    exceeded = (hist_window['basis_weight'] < lower_bound) | (hist_window['basis_weight'] > upper_bound)
    
    if not exceeded.any():
        continue
        
    first_exceed_s = exceeded.idxmax()
    hist_stab_time = settle_s - first_exceed_s
    
    # 2. Find when risk was first flagged (>50%)
    flag_s = None
    # We don't want to evaluate every second if it's slow, let's jump by 10s
    for t in range(int(start_s), int(settle_s), 10):
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
    feat_flag = get_features_for_index(flag_s)
    rec = re.generate_recommendation(feat_flag, target_bw, current_df_raw_index=flag_s)
    
    # The recommended action is extracted. But the recommendation_engine evaluates the 
    # forward simulation directly. Let's do the full forward simulation to find settling time.
    # We parse the recommendation or just run the optimizer again?
    # Wait, re.generate_recommendation returns predicted_new_risk, but how do we get the actual trajectory?
    # Let's extract the optimal setpoints. We can just use the K-NN best_change_hist if optimizer is constrained.
    # Actually, we can just read the rec['recommended_action'] string to see what it changed.
    action_str = rec['recommended_action']
    # Format: "Increase stock_flow by 0.1%" or "Reduce stock_flow by 0.1%"
    parts = action_str.split(' ')
    if len(parts) >= 4:
        direction = parts[0]
        mv = parts[1]
        pct = float(parts[3].replace('%', ''))
        
        cur_val = df_raw.loc[flag_s, mv]
        new_val = cur_val * (1 + (pct/100.0 if direction == 'Increase' else -pct/100.0))
        
        new_sf = df_raw.loc[flag_s, 'stock_flow']
        new_sp = df_raw.loc[flag_s, 'steam_pressure']
        new_ms = df_raw.loc[flag_s, 'machine_speed']
        
        if mv == 'stock_flow': new_sf = new_val
        if mv == 'steam_pressure': new_sp = new_val
        if mv == 'machine_speed': new_ms = new_val
    else:
        new_sf = df_raw.loc[flag_s, 'stock_flow']
        new_sp = df_raw.loc[flag_s, 'steam_pressure']
        new_ms = df_raw.loc[flag_s, 'machine_speed']
        
    # Simulate features 1500s forward
    hist_win = df_raw.loc[max(0, flag_s - 899) : flag_s].copy()
    future_rows = []
    last_row = hist_win.iloc[-1]
    
    ramp_dur = 180
    sf_ramp = np.linspace(last_row['stock_flow'], new_sf, ramp_dur)
    sp_ramp = np.linspace(last_row['steam_pressure'], new_sp, ramp_dur)
    ms_ramp = np.linspace(last_row['machine_speed'], new_ms, ramp_dur)
    
    for i in range(1, 1501):
        new_row = last_row.copy()
        new_row['stock_flow'] = sf_ramp[i-1] if i <= ramp_dur else new_sf
        new_row['steam_pressure'] = sp_ramp[i-1] if i <= ramp_dur else new_sp
        new_row['machine_speed'] = ms_ramp[i-1] if i <= ramp_dur else new_ms
        future_rows.append(new_row)
        
    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([hist_win, future_df], ignore_index=True)
    
    raw_tags = ['stock_flow', 'steam_pressure', 'machine_speed', 'filler_flow', 'basis_weight', 'moisture', 'ash']
    for tag in raw_tags:
        combined[f'{tag}_lag_1s'] = combined[tag].shift(1)
        combined[f'{tag}_lag_5s'] = combined[tag].shift(5)
        combined[f'{tag}_lag_15s'] = combined[tag].shift(15)
        combined[f'{tag}_roll_mean_5m'] = combined[tag].rolling(300, min_periods=1).mean()
        combined[f'{tag}_roll_std_5m'] = combined[tag].rolling(300, min_periods=1).std().fillna(0)
        combined[f'{tag}_roll_mean_15m'] = combined[tag].rolling(900, min_periods=1).mean()
        combined[f'{tag}_roll_std_15m'] = combined[tag].rolling(900, min_periods=1).std().fillna(0)
        combined[f'{tag}_roc_1s'] = combined[tag].diff(1)
        combined[f'{tag}_roc_5s'] = combined[tag].diff(5)
        
    proj_features = combined.iloc[-1500:].copy()
    proj_features = proj_features[allowed_cols]
    
    # Predict BW
    pred_bw = reg_model.predict(proj_features)
    
    # Find when it enters +/- 2.5% band AND STAYS THERE for the rest of the window
    # Actually, if it stays there for 30s we can consider it settled
    in_band = (pred_bw >= lower_bound) & (pred_bw <= upper_bound)
    
    proj_settle_idx = -1
    for i in range(len(in_band)-30):
        if all(in_band[i:i+30]):
            proj_settle_idx = i
            break
            
    if proj_settle_idx == -1:
        # Never settled in 1500s
        proj_stab_time = 1500
    else:
        # Time from first exceed to projected settle
        # wait, the recommendation is applied at flag_s
        # proj_settle_idx is relative to flag_s
        # total projected stabilization time = (flag_s - first_exceed_s) + proj_settle_idx
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
