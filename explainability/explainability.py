import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import shap
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

# Feature mapping
FEATURE_MAPPING = {
    "stock_flow_roll_std_5m": "Stock flow variability (last 5 min)",
    "machine_speed_roll_std_15m": "Machine speed variability (last 15 min)",
    "steam_pressure_roll_mean_15m": "Steam pressure trend (last 15 min)",
    "moisture_roll_std_15m": "Moisture variability (last 15 min)",
    "moisture_roll_mean_15m": "Moisture trend (last 15 min)",
    "machine_speed_roll_mean_5m": "Machine speed trend (last 5 min)",
    "basis_weight_roll_std_5m": "Basis weight variability (last 5 min)",
    "stock_flow_roll_std_15m": "Stock flow variability (last 15 min)",
    "steam_pressure_roll_std_15m": "Steam pressure variability (last 15 min)",
    "basis_weight_roll_std_15m": "Basis weight variability (last 15 min)",
    "ash_roll_std_15m": "Ash variability (last 15 min)",
    "moisture_roll_std_5m": "Moisture variability (last 5 min)",
    "steam_pressure_roll_std_5m": "Steam pressure variability (last 5 min)",
    "moisture_roll_mean_5m": "Moisture trend (last 5 min)",
    "basis_weight_roll_mean_15m": "Basis weight trend (last 15 min)"
}

def load_data():
    val_df = pd.read_csv('./output/val_features.csv')
    df_raw = pd.read_csv('./output/process_data.csv', low_memory=False)
    events = pd.read_csv('./output/grade_change_log.csv')
    
    nan_mask = df_raw.index < 899
    end_mask = df_raw.index >= (len(df_raw) - 60)
    df_clean = df_raw[~(nan_mask | end_mask)].copy()
    
    train_event_ids, val_event_ids = train_test_split(events['event_id'].values, test_size=9, stratify=events['went_off_spec'].values, random_state=42)
    df_clean['is_validation_set'] = df_clean['event_id'].isin(val_event_ids)
    buffer_mask = (df_clean['is_validation_set'].rolling(901, min_periods=1).max() > 0) & (~df_clean['is_validation_set'])
    df_clean = df_clean[~buffer_mask].copy()
    
    df_clean['split'] = df_clean['is_validation_set'].map({True: 'val', False: 'train'})
    val_meta = df_clean[df_clean['split'] == 'val'].reset_index(drop=True)
    
    val_df['event_id'] = val_meta['event_id']
    
    cols_to_drop = ['target_bw_60s_future', 'target_bw_300s_future', 'target_is_off_spec_60s_future', 'target_is_off_spec_300s_future', 'event_id']
    
    # Drop setpoint columns exactly as in ablated training
    ablation_cols = [c for c in val_df.columns if c.endswith('_sp') or c.endswith('_target') or c == 'bw_setpoint']
    cols_to_drop.extend(ablation_cols)
    
    drop_cols = [c for c in cols_to_drop if c in val_df.columns]
    X_val = val_df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    target_col = 'target_is_off_spec_300s_future' if 'target_is_off_spec_300s_future' in val_df.columns else 'target_is_off_spec_60s_future'
    y_val = val_df[target_col]
    event_ids = val_df['event_id']
    
    return X_val, y_val, event_ids

def main():
    print("Loading data and model...")
    X_val, y_val, event_ids = load_data()
    model = joblib.load('./output/lgbm_classifier_ablated.pkl')
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # The output of shap_values for LightGBM binary classifier is usually a list of two arrays [negative_class, positive_class]
    # or a single array representing the positive class log odds.
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    else:
        shap_vals_pos = shap_values
        
    # Save the explainer and mapping
    print("Saving explainer and mapping...")
    with open('./output/explainer.pkl', 'wb') as f:
        joblib.dump(explainer, f)
    with open('./output/feature_mapping.json', 'w') as f:
        json.dump(FEATURE_MAPPING, f)
        
    def explain_prediction(row_index):
        sv = shap_vals_pos[row_index]
        # Sort indices by absolute SHAP value
        top_indices = np.argsort(np.abs(sv))[::-1][:3]
        
        explanation = []
        for idx in top_indices:
            feat_name = X_val.columns[idx]
            contribution = sv[idx]
            readable_name = FEATURE_MAPPING.get(feat_name, feat_name)
            direction = "increasing risk" if contribution > 0 else "decreasing risk"
            
            explanation.append({
                "feature": feat_name,
                "contribution": float(round(contribution, 4)),
                "direction": direction,
                "readable_name": readable_name
            })
            
        return explanation

    print("\n--- SHAP EXPLANATIONS FOR EVENT 16 (OFF-SPEC ROWS) ---")
    event_16_indices = np.where((event_ids == 16) & (y_val == 1))[0]
    test_indices_16 = [event_16_indices[0], event_16_indices[len(event_16_indices)//2], event_16_indices[-1]]
    
    for idx in test_indices_16:
        pred_prob = model.predict(X_val.iloc[[idx]])[0]
        actual = y_val.iloc[idx]
        print(f"Row {idx} | Actual: {actual} | Predicted Risk: {pred_prob:.2%}")
        expl = explain_prediction(idx)
        for e in expl:
            print(f"  -> {e['readable_name']} ({e['feature']}): {e['contribution']:+.2f} ({e['direction']})")
            
    print("\n--- SHAP EXPLANATIONS FOR EVENT 34 (OFF-SPEC ROWS) [Disturbance: speed_hunting] ---")
    event_34_indices = np.where((event_ids == 34) & (y_val == 1))[0]
    test_indices_34 = [event_34_indices[0], event_34_indices[len(event_34_indices)//2], event_34_indices[-1]]
    
    for idx in test_indices_34:
        pred_prob = model.predict(X_val.iloc[[idx]])[0]
        actual = y_val.iloc[idx]
        print(f"Row {idx} | Actual: {actual} | Predicted Risk: {pred_prob:.2%}")
        expl = explain_prediction(idx)
        for e in expl:
            print(f"  -> {e['readable_name']} ({e['feature']}): {e['contribution']:+.2f} ({e['direction']})")
            
    print("\n--- SHAP EXPLANATIONS FOR EVENT 11 (OFF-SPEC ROWS, TRAIN SET) [Disturbance: stock_surge] ---")
    train_df = pd.read_csv('./output/train_features.csv')
    df_raw = pd.read_csv('./output/process_data.csv', low_memory=False)
    
    nan_mask = df_raw.index < 899
    end_mask = df_raw.index >= (len(df_raw) - 60)
    df_clean = df_raw[~(nan_mask | end_mask)].copy()
    
    events = pd.read_csv('./output/grade_change_log.csv')
    from sklearn.model_selection import train_test_split
    train_event_ids, val_event_ids = train_test_split(events['event_id'].values, test_size=9, stratify=events['went_off_spec'].values, random_state=42)
    df_clean['is_validation_set'] = df_clean['event_id'].isin(val_event_ids)
    buffer_mask = (df_clean['is_validation_set'].rolling(901, min_periods=1).max() > 0) & (~df_clean['is_validation_set'])
    df_clean = df_clean[~buffer_mask].copy()
    
    df_clean['split'] = df_clean['is_validation_set'].map({True: 'val', False: 'train'})
    train_meta = df_clean[df_clean['split'] == 'train'].reset_index(drop=True)
    train_df['event_id'] = train_meta['event_id']
    
    cols_to_drop_train = ['target_bw_60s_future', 'target_bw_300s_future', 'target_is_off_spec_60s_future', 'target_is_off_spec_300s_future', 'event_id']
    ablation_cols_train = [c for c in train_df.columns if c.endswith('_sp') or c.endswith('_target') or c == 'bw_setpoint']
    cols_to_drop_train.extend(ablation_cols_train)
    
    drop_cols_train = [c for c in cols_to_drop_train if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols_train).select_dtypes(include=[np.number])
    target_col_train = 'target_is_off_spec_300s_future' if 'target_is_off_spec_300s_future' in train_df.columns else 'target_is_off_spec_60s_future'
    y_train = train_df[target_col_train]
    train_event_ids_col = train_df['event_id']
    
    shap_values_train = explainer.shap_values(X_train)
    if isinstance(shap_values_train, list):
        shap_vals_pos_train = shap_values_train[1]
    else:
        shap_vals_pos_train = shap_values_train
        
    def explain_prediction_train(row_index):
        sv = shap_vals_pos_train[row_index]
        top_indices = np.argsort(np.abs(sv))[::-1][:3]
        explanation = []
        for idx in top_indices:
            feat_name = X_train.columns[idx]
            contribution = sv[idx]
            readable_name = FEATURE_MAPPING.get(feat_name, feat_name)
            direction = "increasing risk" if contribution > 0 else "decreasing risk"
            explanation.append({"feature": feat_name, "contribution": float(round(contribution, 4)), "direction": direction, "readable_name": readable_name})
        return explanation

    event_11_indices = np.where((train_event_ids_col == 11) & (y_train == 1))[0]
    if len(event_11_indices) > 0:
        test_indices_11 = [event_11_indices[0], event_11_indices[len(event_11_indices)//2], event_11_indices[-1]]
        for idx in test_indices_11:
            pred_prob = model.predict(X_train.iloc[[idx]])[0]
            actual = y_train.iloc[idx]
            print(f"Row {idx} | Actual: {actual} | Predicted Risk: {pred_prob:.2%}")
            expl = explain_prediction_train(idx)
            for e in expl:
                print(f"  -> {e['readable_name']} ({e['feature']}): {e['contribution']:+.2f} ({e['direction']})")

if __name__ == '__main__':
    main()

_explainer = None
_shap_mapping = None

def get_explainer():
    global _explainer, _shap_mapping
    if _explainer is None:
        _explainer = joblib.load('./output/explainer.pkl')
        with open('./output/feature_mapping.json', 'r') as f:
            import json
            _shap_mapping = json.load(f)
    return _explainer, _shap_mapping

def explain_prediction(row_features_df):
    explainer, mapping = get_explainer()
    shap_vals = explainer.shap_values(row_features_df)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]
        
    top_indices = np.argsort(np.abs(sv))[::-1][:3]
    explanation = []
    for idx in top_indices:
        feat_name = row_features_df.columns[idx]
        contribution = sv[idx]
        readable_name = mapping.get(feat_name, feat_name)
        direction = "increasing risk" if contribution > 0 else "decreasing risk"
        explanation.append({
            "feature": feat_name,
            "contribution": float(round(contribution, 4)),
            "direction": direction,
            "readable_name": readable_name
        })
    return explanation
