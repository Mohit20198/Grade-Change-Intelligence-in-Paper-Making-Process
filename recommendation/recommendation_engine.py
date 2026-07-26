import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 1. Load Everything needed
print("Loading Recommendation Engine dependencies...")
clf_model = joblib.load('./output/lgbm_classifier_ablated.pkl')
reg_model = joblib.load('./output/lgbm_regressor_ablated.pkl')
explainer = joblib.load('./output/explainer.pkl')

with open('./output/feature_mapping.json', 'r') as f:
    FEATURE_MAPPING = json.load(f)

# Causal pathways from Phase 2
CAUSAL_PATHS = {
    'stock_flow': 'has a validated causal effect on basis_weight (Phase 2 correlation discovery, lag 5-60s).',
    'machine_speed': 'has a validated causal effect on basis_weight and moisture (Phase 2 correlation discovery, hidden coupling verified).',
    'steam_pressure': 'has a validated causal effect on moisture (Phase 2 correlation discovery).',
    'filler_flow': 'has a validated causal effect on ash (Phase 2 correlation discovery).'
}

print("Loading datasets and preparing K-NN pool...")
train_df = pd.read_csv('./output/train_features.csv')
df_raw = pd.read_csv('./output/process_data.csv', low_memory=False)

nan_mask = df_raw.index < 899
end_mask = df_raw.index >= (len(df_raw) - 60)
df_clean = df_raw[~(nan_mask | end_mask)].copy()

events = pd.read_csv('./output/grade_change_log.csv')
train_event_ids, val_event_ids = train_test_split(events['event_id'].values, test_size=9, stratify=events['went_off_spec'].values, random_state=42)
df_clean['is_validation_set'] = df_clean['event_id'].isin(val_event_ids)
buffer_mask = (df_clean['is_validation_set'].rolling(901, min_periods=1).max() > 0) & (~df_clean['is_validation_set'])
df_clean = df_clean[~buffer_mask].copy()

df_clean['split'] = df_clean['is_validation_set'].map({True: 'val', False: 'train'})
train_meta = df_clean[df_clean['split'] == 'train'].reset_index(drop=True)
val_meta = df_clean[df_clean['split'] == 'val'].reset_index(drop=True)

train_df['event_id'] = train_meta['event_id']

# Length may differ after 300s dropna relabeling — truncate to the shorter of the two
clean_train_idx = df_clean[df_clean['split'] == 'train'].index
n_align = min(len(train_df), len(clean_train_idx))
train_df = train_df.iloc[:n_align].copy()
train_df['event_id'] = train_meta['event_id'].values[:n_align]
train_df['df_raw_index'] = clean_train_idx[:n_align]

# We only want to search over off-spec training events where the future is ON-SPEC
off_spec_events_train = events[(events['went_off_spec'] == True) & (events['event_id'].isin(train_event_ids))]['event_id'].values
# Support both 60s and 300s target column names (use whichever is present)
target_col_knn = 'target_is_off_spec_300s_future' if 'target_is_off_spec_300s_future' in train_df.columns else 'target_is_off_spec_60s_future'
knn_pool_mask = (train_df['event_id'].isin(off_spec_events_train)) & (train_df[target_col_knn] == 0)
knn_pool_df = train_df[knn_pool_mask].copy()

# Drop all label columns (handle both 60s and 300s naming), plus meta columns
cols_to_drop = ['target_bw_60s_future', 'target_bw_300s_future',
                'target_is_off_spec_60s_future', 'target_is_off_spec_300s_future',
                'event_id', 'df_raw_index']
ablation_cols = [c for c in train_df.columns if c.endswith('_sp') or c.endswith('_target') or c == 'bw_setpoint']
# Also drop any non-numeric (object) columns that leaked from the feature CSV
obj_cols = [c for c in knn_pool_df.columns if knn_pool_df[c].dtype == object]
drop_for_knn = [c for c in set(cols_to_drop + ablation_cols + obj_cols) if c in knn_pool_df.columns]

X_knn_pool = knn_pool_df.drop(columns=drop_for_knn).select_dtypes(include=[np.number])
scaler = StandardScaler().fit(X_knn_pool)
X_knn_pool_scaled = scaler.transform(X_knn_pool)
knn_pool_raw_indices = knn_pool_df['df_raw_index'].values

def extract_shap_leading_indicator(row_features_df):
    shap_vals = explainer.shap_values(row_features_df)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]
    top_idx = np.argsort(sv)[::-1][0]
    feat_name = row_features_df.columns[top_idx]
    readable_name = FEATURE_MAPPING.get(feat_name, feat_name)
    return readable_name

# Expose a helper to extract features for a specific index for the dashboard
def get_features_for_index(idx):
    if idx < 899:
        return None
    hist_window = df_raw.loc[idx - 899 : idx].copy()
    raw_tags = ['stock_flow', 'steam_pressure', 'machine_speed', 'filler_flow', 'basis_weight', 'moisture', 'ash']
    for tag in raw_tags:
        hist_window[f'{tag}_lag_1s'] = hist_window[tag].shift(1)
        hist_window[f'{tag}_lag_5s'] = hist_window[tag].shift(5)
        hist_window[f'{tag}_lag_15s'] = hist_window[tag].shift(15)
        
        hist_window[f'{tag}_roll_mean_5m'] = hist_window[tag].rolling(300, min_periods=1).mean()
        hist_window[f'{tag}_roll_std_5m'] = hist_window[tag].rolling(300, min_periods=1).std().fillna(0)
        hist_window[f'{tag}_roll_mean_15m'] = hist_window[tag].rolling(900, min_periods=1).mean()
        hist_window[f'{tag}_roll_std_15m'] = hist_window[tag].rolling(900, min_periods=1).std().fillna(0)
        
        hist_window[f'{tag}_roc_1s'] = hist_window[tag].diff(1)
        hist_window[f'{tag}_roc_5s'] = hist_window[tag].diff(5)
        
    final_row = hist_window.iloc[[-1]].copy()
    allowed_cols = clf_model.feature_name()
    return final_row[allowed_cols]

def generate_recommendation(row_features_df, current_bw_sp, current_df_raw_index=None):
    # 1. Predict current risk
    risk_score = clf_model.predict(row_features_df)[0]
    
    # 2. Extract leading indicator (SHAP)
    leading_ind = extract_shap_leading_indicator(row_features_df)
    
    # 3. K-NN Historical Search
    x_scaled = scaler.transform(row_features_df)
    sims = cosine_similarity(x_scaled, X_knn_pool_scaled)[0]
    top_5_idx = np.argsort(sims)[::-1][:5]
    
    mv_changes = {'stock_flow': [], 'steam_pressure': [], 'machine_speed': []}
    for idx in top_5_idx:
        raw_idx = knn_pool_raw_indices[idx]
        for mv in mv_changes.keys():
            val_now = df_raw.loc[raw_idx, mv]
            if raw_idx + 90 in df_raw.index:
                val_future = df_raw.loc[raw_idx + 90, mv]
                pct_change = (val_future - val_now) / (val_now + 1e-9) * 100
                mv_changes[mv].append(pct_change)
                
    avg_changes = {k: np.mean(v) for k, v in mv_changes.items() if len(v) > 0}
    
    if len(avg_changes) == 0:
        best_mv_hist = 'stock_flow'
        best_change_hist = 0.0
    else:
        best_mv_hist = max(avg_changes, key=lambda k: abs(avg_changes[k]))
        best_change_hist = avg_changes[best_mv_hist]
    
    action_word = "increased" if best_change_hist > 0 else "reduced"
    historical_summary = f"{best_mv_hist} was {action_word} by ~{abs(best_change_hist):.1f}% over the next 90s in similar successful cases."
    
    # 4. Constrained Optimizer (using Regressor)
    initial_sf = row_features_df['stock_flow'].values[0]
    initial_sp = row_features_df['steam_pressure'].values[0]
    initial_ms = row_features_df['machine_speed'].values[0]
    
    def objective(x):
        temp_df = row_features_df.copy()
        temp_df['stock_flow'] = x[0]
        temp_df['steam_pressure'] = x[1]
        temp_df['machine_speed'] = x[2]
        pred_bw = reg_model.predict(temp_df)[0]
        return abs(pred_bw - current_bw_sp)
        
    # Rate-constrained bounds: +/- 0.2% maximum allowable instantaneous intervention
    # derived empirically from the K-NN historical successful cases, where operators
    # made trims of ~0.1% over 90s (equivalent to 0.2% over a full 180s ramp).
    bounds = [
        (initial_sf * 0.998, initial_sf * 1.002),
        (initial_sp * 0.998, initial_sp * 1.002),
        (initial_ms * 0.998, initial_ms * 1.002)
    ]
    
    res = differential_evolution(objective, bounds=bounds, seed=42)
    opt_sf, opt_sp, opt_ms = res.x
    
    opt_changes = {
        'stock_flow': (opt_sf - initial_sf)/initial_sf * 100,
        'steam_pressure': (opt_sp - initial_sp)/initial_sp * 100,
        'machine_speed': (opt_ms - initial_ms)/initial_ms * 100
    }
    
    best_mv_opt = max(opt_changes, key=lambda k: abs(opt_changes[k]))
    best_change_opt = opt_changes[best_mv_opt]
    
    def forward_simulate_features(raw_idx, new_sf, new_sp, new_ms, ramp_dur=180):
        # 1. Take the last 15 mins
        hist_window = df_raw.loc[max(0, raw_idx - 899) : raw_idx].copy()
        
        # 2. Create 300s future projection (5 mins)
        future_rows = []
        last_row = hist_window.iloc[-1]
        
        # Ramps over ramp_dur
        sf_ramp = np.linspace(last_row['stock_flow'], new_sf, ramp_dur)
        sp_ramp = np.linspace(last_row['steam_pressure'], new_sp, ramp_dur)
        ms_ramp = np.linspace(last_row['machine_speed'], new_ms, ramp_dur)
        
        for i in range(1, 301):
            new_row = last_row.copy()
            new_row['stock_flow'] = sf_ramp[i-1] if i <= ramp_dur else new_sf
            new_row['steam_pressure'] = sp_ramp[i-1] if i <= ramp_dur else new_sp
            new_row['machine_speed'] = ms_ramp[i-1] if i <= ramp_dur else new_ms
            # Keep CVs and filler_flow flat at current value
            future_rows.append(new_row)
            
        future_df = pd.DataFrame(future_rows)
        combined = pd.concat([hist_window, future_df], ignore_index=True)
        
        # 3. Recompute features
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
            
        # 4. Extract final row
        final_row = combined.iloc[[-1]].copy()
        
        # Drop ablated features
        cols_to_drop = [c for c in final_row.columns if c.endswith('_sp') or c.endswith('_target') or c == 'bw_setpoint']
        # Also drop raw columns not in the model input (timestamp, event_id, etc.)
        allowed_cols = row_features_df.columns.tolist()
        final_row = final_row[allowed_cols]
        
        return final_row

    # Check consistency
    if abs(best_change_opt) > 0.5:
        source = "optimizer"
        rec_mv = best_mv_opt
        rec_change = best_change_opt
        action_str = "Increase" if rec_change > 0 else "Reduce"
        rec_text = f"{action_str} {rec_mv} by {abs(rec_change):.1f}%"
        
        # Forward simulate with optimizer values (using 180s, matching simulator constraint)
        sim_features_df = forward_simulate_features(current_df_raw_index, opt_sf, opt_sp, opt_ms, ramp_dur=180)
        predicted_new_risk = clf_model.predict(sim_features_df)[0]
        
        print("\n[DEBUG] Optimizer forward simulation (5 min ahead, 180s ramp):")
        print(f"  Orig SF Roll Std (5m): {row_features_df['stock_flow_roll_std_5m'].values[0]:.4f} | Sim SF Roll Std: {sim_features_df['stock_flow_roll_std_5m'].values[0]:.4f}")
        print(f"  Orig SP Roll Std (15m): {row_features_df['steam_pressure_roll_std_15m'].values[0]:.4f} | Sim SP Roll Std: {sim_features_df['steam_pressure_roll_std_15m'].values[0]:.4f}")
        print(f"  Orig Risk: {risk_score:.2%} | Adj Risk: {predicted_new_risk:.2%}")
        
        if best_mv_opt == best_mv_hist:
            hist_str = f"Similar to 5 historical cases, most of which stabilized by adjusting {best_mv_opt}. {historical_summary}"
        else:
            hist_str = f"Note: historical precedent suggests {best_mv_hist} was the more common successful intervention in similar cases ({historical_summary}), while the optimizer's live search recommends {best_mv_opt} \u2014 presenting both for operator judgment."
    else:
        source = "historical_success"
        rec_mv = best_mv_hist
        rec_change = best_change_hist
        action_str = "Increase" if rec_change > 0 else "Reduce"
        rec_text = f"{action_str} {rec_mv} by {abs(rec_change):.1f}%"
        
        # Forward simulate with historical adjustment
        hist_sf = initial_sf * (1 + rec_change/100.0) if rec_mv == 'stock_flow' else initial_sf
        hist_sp = initial_sp * (1 + rec_change/100.0) if rec_mv == 'steam_pressure' else initial_sp
        hist_ms = initial_ms * (1 + rec_change/100.0) if rec_mv == 'machine_speed' else initial_ms
        
        sim_features_df = forward_simulate_features(current_df_raw_index, hist_sf, hist_sp, hist_ms, ramp_dur=180)
        predicted_new_risk = clf_model.predict(sim_features_df)[0]
        hist_str = f"Similar to 5 historical cases, most of which stabilized by adjusting {best_mv_hist}. {historical_summary}"


    causal_pathway = CAUSAL_PATHS.get(rec_mv, "No causal path documented.")
    
    rec_obj = {
        "current_risk_score": float(risk_score),
        "recommended_action": rec_text,
        "predicted_new_risk": float(predicted_new_risk),
        "supporting_evidence": {
            "historical_precedent": hist_str,
            "causal_pathway": f"{rec_mv} {causal_pathway}"
        },
        "source": source,
        "leading_indicator_context": f"{leading_ind} indicates rising instability."
    }
    
    return rec_obj

def test_pipeline():
    print("\nLoading Validation Set for testing...")
    val_df = pd.read_csv('./output/val_features.csv')
    val_df['event_id'] = val_meta['event_id']
    val_df['df_raw_index'] = df_clean[df_clean['split'] == 'val'].index
    
    y_val = val_df['target_is_off_spec_300s_future'] if 'target_is_off_spec_300s_future' in val_df.columns else val_df['target_is_off_spec_60s_future']
    event_ids = val_df['event_id']
    
    X_val = val_df.drop(columns=drop_for_knn)
    
    print("\n--- TEST: EVENT 16 (Start of Deviation) ---")
    event_16_indices = np.where((event_ids == 16) & (y_val == 1))[0]
    if len(event_16_indices) > 0:
        idx_16 = event_16_indices[0]
        row_df_16 = X_val.iloc[[idx_16]]
        raw_idx = val_df['df_raw_index'].iloc[idx_16]
        bw_sp = df_raw.loc[raw_idx, 'bw_setpoint']
        
        rec_16 = generate_recommendation(row_df_16, bw_sp, raw_idx)
        print(json.dumps(rec_16, indent=2))
        
    print("\n--- TEST: EVENT 34 (Start of Deviation) ---")
    event_34_indices = np.where((event_ids == 34) & (y_val == 1))[0]
    if len(event_34_indices) > 0:
        idx_34 = event_34_indices[0]
        row_df_34 = X_val.iloc[[idx_34]]
        raw_idx = val_df['df_raw_index'].iloc[idx_34]
        bw_sp = df_raw.loc[raw_idx, 'bw_setpoint']
        
        rec_34 = generate_recommendation(row_df_34, bw_sp, raw_idx)
        print(json.dumps(rec_34, indent=2))

if __name__ == '__main__':
    test_pipeline()
