import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import joblib

def main():
    print("Loading data...")
    # 1. Load train and val features
    train_df = pd.read_csv('./output/train_features.csv')
    val_df = pd.read_csv('./output/val_features.csv')
    
    # 2. We need the event_ids. audit_features.py dropped them, so we recreate the mapping.
    df_raw = pd.read_csv('./output/process_data.csv', low_memory=False)
    events = pd.read_csv('./output/grade_change_log.csv')
    
    # Re-create exactly what audit_features did
    # Re-create exactly what audit_features did
    # min_periods=900 means the first 899 rows are NaN.
    # feature_engineering also drops the last 60 rows due to future target shift.
    nan_mask = df_raw.index < 899
    end_mask = df_raw.index >= (len(df_raw) - 60)
    df_clean = df_raw[~(nan_mask | end_mask)].copy()
    
    # Re-split train/val
    train_event_ids, val_event_ids = train_test_split(
        events['event_id'].values,
        test_size=9,
        stratify=events['went_off_spec'].values,
        random_state=42
    )
    df_clean['is_validation_set'] = df_clean['event_id'].isin(val_event_ids)
    buffer_mask = (df_clean['is_validation_set'].rolling(901, min_periods=1).max() > 0) & (~df_clean['is_validation_set'])
    df_clean = df_clean[~buffer_mask].copy()
    
    # Assign split
    df_clean['split'] = df_clean['is_validation_set'].map({True: 'val', False: 'train'})
    
    train_meta = df_clean[df_clean['split'] == 'train'].reset_index(drop=True)
    val_meta = df_clean[df_clean['split'] == 'val'].reset_index(drop=True)
    
    assert len(train_meta) == len(train_df), "Train length mismatch"
    assert len(val_meta) == len(val_df), "Val length mismatch"
    
    train_df['event_id'] = train_meta['event_id']
    train_df['split'] = train_meta['split']
    val_df['event_id'] = val_meta['event_id']
    val_df['split'] = val_meta['split']
    
    y_reg_train = train_df['target_bw_60s_future']
    y_reg_val = val_df['target_bw_60s_future']
    y_class_train = train_df['target_is_off_spec_60s_future']
    y_class_val = val_df['target_is_off_spec_60s_future']
    
    cols_to_drop = ['target_bw_60s_future', 'target_is_off_spec_60s_future', 'event_id', 'split']
    X_train = train_df.drop(columns=cols_to_drop)
    X_val = val_df.drop(columns=cols_to_drop)
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    # 2. TRAIN REGRESSION MODEL
    print("\nTraining Regression Model...")
    params_reg = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_samples': 50,
        'verbose': -1,
        'random_state': 42
    }
    train_data_reg = lgb.Dataset(X_train, label=y_reg_train)
    val_data_reg = lgb.Dataset(X_val, label=y_reg_val, reference=train_data_reg)
    model_reg = lgb.train(
        params_reg, train_data_reg, num_boost_round=1000, 
        valid_sets=[val_data_reg], callbacks=[lgb.early_stopping(50)]
    )
    
    # 3. TRAIN CLASSIFICATION MODEL
    print("\nTraining Classification Model...")
    pos_count = y_class_train.sum()
    neg_count = len(y_class_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    params_clf = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_samples': 50,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'random_state': 42
    }
    train_data_clf = lgb.Dataset(X_train, label=y_class_train)
    val_data_clf = lgb.Dataset(X_val, label=y_class_val, reference=train_data_clf)
    model_clf = lgb.train(
        params_clf, train_data_clf, num_boost_round=1000, 
        valid_sets=[val_data_clf], callbacks=[lgb.early_stopping(50)]
    )
    
    # 4. EVALUATE BOTH APPROACHES
    print("\nEvaluating Models...")
    pred_bw_val = model_reg.predict(X_val)
    pred_prob_val = model_clf.predict(X_val)
    pred_class_val = (pred_prob_val > 0.5).astype(int)
    
    # Derive risk flag for regression
    pred_risk_val = (np.abs(pred_bw_val - X_val['bw_setpoint']) / X_val['bw_setpoint'] > 0.025).astype(int)
    
    # Aggregate metrics
    reg_rmse = np.sqrt(mean_squared_error(y_reg_val, pred_bw_val))
    reg_mae = mean_absolute_error(y_reg_val, pred_bw_val)
    reg_prec = precision_score(y_class_val, pred_risk_val, zero_division=0)
    reg_rec = recall_score(y_class_val, pred_risk_val, zero_division=0)
    reg_f1 = f1_score(y_class_val, pred_risk_val, zero_division=0)
    
    clf_auc = roc_auc_score(y_class_val, pred_prob_val)
    clf_ap = average_precision_score(y_class_val, pred_prob_val)
    clf_prec = precision_score(y_class_val, pred_class_val, zero_division=0)
    clf_rec = recall_score(y_class_val, pred_class_val, zero_division=0)
    clf_f1 = f1_score(y_class_val, pred_class_val, zero_division=0)
    
    # 5. PER-EVENT BREAKDOWN
    print("\nPer-Event Breakdown (Off-spec events only):")
    off_spec_events = events[events['went_off_spec'] == True]['event_id'].tolist()
    
    # Add predictions to metadata
    val_meta['y_true'] = y_class_val.values
    val_meta['pred_reg'] = pred_risk_val
    val_meta['pred_clf'] = pred_class_val
    
    # We also need to evaluate training events. So we need predictions on train data.
    pred_bw_train = model_reg.predict(X_train)
    pred_class_train = (model_clf.predict(X_train) > 0.5).astype(int)
    pred_risk_train = (np.abs(pred_bw_train - X_train['bw_setpoint']) / X_train['bw_setpoint'] > 0.025).astype(int)
    
    train_meta['y_true'] = y_class_train.values
    train_meta['pred_reg'] = pred_risk_train
    train_meta['pred_clf'] = pred_class_train
    
    all_meta = pd.concat([train_meta, val_meta])
    
    records = []
    for eid in off_spec_events:
        event_df = all_meta[all_meta['event_id'] == eid]
        if len(event_df) == 0: continue
        
        y_t = event_df['y_true']
        if y_t.sum() == 0: continue # Should have some positive rows
        
        pr_reg = precision_score(y_t, event_df['pred_reg'], zero_division=0)
        re_reg = recall_score(y_t, event_df['pred_reg'], zero_division=0)
        
        pr_clf = precision_score(y_t, event_df['pred_clf'], zero_division=0)
        re_clf = recall_score(y_t, event_df['pred_clf'], zero_division=0)
        
        records.append({
            'event_id': eid,
            'split': event_df['split'].iloc[0],
            'true_pos_rows': y_t.sum(),
            'reg_precision': pr_reg,
            'reg_recall': re_reg,
            'clf_precision': pr_clf,
            'clf_recall': re_clf
        })
    
    rep_df = pd.DataFrame(records).sort_values(by=['split', 'event_id'])
    print(f"{'event_id':<10} {'split':<8} {'true_pos':<10} | {'Reg_Prec':<10} {'Reg_Rec':<10} | {'Clf_Prec':<10} {'Clf_Rec':<10}")
    print("-" * 80)
    for _, r in rep_df.iterrows():
        print(f"{r['event_id']:<10} {r['split']:<8} {r['true_pos_rows']:<10} | {r['reg_precision']:<10.4f} {r['reg_recall']:<10.4f} | {r['clf_precision']:<10.4f} {r['clf_recall']:<10.4f}")
        
    # 6. FEATURE IMPORTANCE
    print("\nTop 15 Features by Gain (Regression Model):")
    importance = model_reg.feature_importance(importance_type='gain')
    feats = model_reg.feature_name()
    imp_df = pd.DataFrame({'feature': feats, 'gain': importance}).sort_values('gain', ascending=False).head(15)
    print(imp_df.to_string(index=False))
    
    # 7. SAVE MODELS
    print("\nSaving models...")
    joblib.dump(model_reg, './output/lgbm_regressor.pkl')
    joblib.dump(model_clf, './output/lgbm_classifier.pkl')
    
    # 8. SUMMARY
    print("\n--- FINAL COMPARISON SUMMARY (Validation Set) ---")
    print(f"{'Metric':<20} | {'Regression (Derived)':<25} | {'Classification (Direct)':<25}")
    print("-" * 75)
    print(f"{'Primary Metric':<20} | {'RMSE: ' + str(round(reg_rmse,4)):<25} | {'AUC: ' + str(round(clf_auc,4)):<25}")
    print(f"{'Precision':<20} | {reg_prec:<25.4f} | {clf_prec:<25.4f}")
    print(f"{'Recall':<20} | {reg_rec:<25.4f} | {clf_rec:<25.4f}")
    print(f"{'F1 Score':<20} | {reg_f1:<25.4f} | {clf_f1:<25.4f}")
    
    val_off = rep_df[rep_df['split'] == 'val']
    print("\nValidation Off-Spec Events Breakdown:")
    for _, r in val_off.iterrows():
        print(f"Event {r['event_id']} ({r['true_pos_rows']} pos rows) -> Reg [P:{r['reg_precision']:.2f}, R:{r['reg_recall']:.2f}] | Clf [P:{r['clf_precision']:.2f}, R:{r['clf_recall']:.2f}]")

if __name__ == '__main__':
    main()
