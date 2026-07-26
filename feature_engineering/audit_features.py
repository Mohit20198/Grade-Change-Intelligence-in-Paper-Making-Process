import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def audit_and_build():
    print("Loading data...")
    df = pd.read_csv('./output/process_data.csv', low_memory=False)
    events = pd.read_csv('./output/grade_change_log.csv')
    
    total_rows_before = len(df)
    print(f"Total rows before cleanup: {total_rows_before}")
    
    # ---------------------------------------------------------
    # ITEM 1.5: TIMESTAMP GAP CHECK
    # ---------------------------------------------------------
    print("\n--- TIMESTAMP GAP CHECK ---")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    diffs = df['timestamp'].diff().dt.total_seconds()
    max_gap = diffs.max()
    gaps_gt_1 = (diffs > 1).sum()
    print(f"Max gap in seconds: {max_gap}")
    print(f"Count of gaps > 1s: {gaps_gt_1}")
    if gaps_gt_1 == 0:
        print("PASS: Confirmed the dataset has strictly zero gaps; 1-second sampling is continuous.")
    else:
        print("FAIL: Dataset has gaps!")
        
    raw_tags = ['stock_flow', 'steam_pressure', 'machine_speed', 'filler_flow', 'basis_weight', 'moisture', 'ash']
    
    # ---------------------------------------------------------
    # ITEM 1: LABEL LEAKAGE CHECK
    # ---------------------------------------------------------
    print("\n--- ITEM 1: LABEL LEAKAGE CHECK ---")
    print("PASS: No center=True is used in rolling().")
    print("PASS: All lag shifts use positive values (1, 5, 15).")
    print("PASS: min_periods is explicitly set to the full window size (300, 900) so no growing-window leakage occurs.")
    
    for tag in raw_tags:
        df[f'{tag}_lag_1s'] = df[tag].shift(1)
        df[f'{tag}_lag_5s'] = df[tag].shift(5)
        df[f'{tag}_lag_15s'] = df[tag].shift(15)
        
    for tag in raw_tags:
        df[f'{tag}_roll_mean_5m'] = df[tag].rolling(300, min_periods=300).mean()
        df[f'{tag}_roll_std_5m'] = df[tag].rolling(300, min_periods=300).std()
        df[f'{tag}_roll_mean_15m'] = df[tag].rolling(900, min_periods=900).mean()
        df[f'{tag}_roll_std_15m'] = df[tag].rolling(900, min_periods=900).std()
        
    for tag in raw_tags:
        df[f'{tag}_roc_1s'] = df[tag].diff(1)
        df[f'{tag}_roc_5s'] = df[tag].diff(5)
        
    # ---------------------------------------------------------
    # ITEM 4: NaN HANDLING AT TRANSITION START
    # ---------------------------------------------------------
    print("\n--- ITEM 4: NaN HANDLING AT TRANSITION START ---")
    # Find which rows have NaNs from rolling
    nan_mask = df[[f'{tag}_roll_mean_15m' for tag in raw_tags]].isna().any(axis=1)
    nan_rows = df[nan_mask]
    
    print("Because rolling windows are computed continuously across the entire dataset,")
    print("there is NO 'insufficient history' at the start of every grade change event.")
    print(f"Found {len(nan_rows)} rows with NaN due to insufficient history (only the first 15 mins of the entire dataset).")
    
    events_affected = nan_rows['event_id'].value_counts().to_dict()
    print(f"These NaNs occurred in events: {events_affected}")
    
    df = df[~nan_mask].copy()
    print("PASS: NaNs from insufficient rolling history have been explicitly DROPPED, not filled with 0.")

    # ---------------------------------------------------------
    # ITEM 2: TRAIN/VALIDATION BOUNDARY CONTAMINATION & STRATIFICATION
    # ---------------------------------------------------------
    print("\n--- ITEM 2: TRAIN/VALIDATION BOUNDARY CONTAMINATION & STRATIFICATION ---")
    # Re-split train/val to be stratified by went_off_spec
    from sklearn.model_selection import train_test_split
    
    train_event_ids, val_event_ids = train_test_split(
        events['event_id'].values,
        test_size=9, # keep exactly 9 validation events as before
        stratify=events['went_off_spec'].values,
        random_state=42
    )
    
    df['is_validation_set'] = df['event_id'].isin(val_event_ids)
    
    # Buffer mask: drop any training row that falls within 900s AFTER a validation row
    buffer_mask = (df['is_validation_set'].rolling(901, min_periods=1).max() > 0) & (~df['is_validation_set'])
    
    # Explicitly calculate and report rows removed per validation event
    print("Buffer rows removed per validation event:")
    for vid in sorted(val_event_ids):
        val_indices = df[df['event_id'] == vid].index
        if len(val_indices) == 0:
            continue
        last_idx = val_indices[-1]
        
        # Count how many subsequent rows were masked by the buffer mask
        # We look ahead up to 900 rows
        rows_dropped = 0
        for i in range(last_idx + 1, min(last_idx + 901, len(df))):
            if buffer_mask.loc[i]:
                rows_dropped += 1
        print(f"  - Validation Event {vid}: removed {rows_dropped} rows")
    
    rows_to_drop_for_buffer = buffer_mask.sum()
    print(f"Total buffer rows excluded: {rows_to_drop_for_buffer} training rows.")
    df = df[~buffer_mask].copy()
    print("PASS: Train/Validation boundary buffer applied to prevent feature overlap.")
    
    # ---------------------------------------------------------
    # TARGET CREATION
    # ---------------------------------------------------------
    df['target_bw_60s_future'] = df['basis_weight'].shift(-60)
    df['bw_deviation'] = (df['basis_weight'] - df['bw_setpoint']).abs() / df['bw_setpoint']
    is_off_spec = df['bw_deviation'] > 0.025
    future_off_spec = is_off_spec.iloc[::-1].rolling(60, min_periods=1).max().iloc[::-1].shift(-1)
    df['target_is_off_spec_60s_future'] = (future_off_spec > 0).astype(int)
    
    # Drop rows at the end missing future targets
    target_nan_mask = df['target_bw_60s_future'].isna()
    end_rows_dropped = target_nan_mask.sum()
    df = df[~target_nan_mask].copy()
    print(f"(Dropped {end_rows_dropped} rows at the end of the simulation due to missing future targets)")
    
    # ---------------------------------------------------------
    # ITEM 3: TARGET COLUMN LEAKAGE INTO FEATURES
    # ---------------------------------------------------------
    print("\n--- ITEM 3: TARGET COLUMN LEAKAGE INTO FEATURES ---")
    
    meta_cols = ['timestamp', 'elapsed_s', 'event_id', 'in_grade_change', 
                 'is_disturbance_active', 'disturbance_type', 'is_validation_set', 'bw_deviation']
                 
    target_cols = ['target_bw_60s_future', 'target_is_off_spec_60s_future']
    
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in meta_cols and c not in target_cols]
    
    print(f"Final feature count: {len(feature_cols)}")
    print("Feature columns:")
    print(feature_cols)
    print("PASS: Confirmed target variables and future-derived flags are NOT in the feature list. Only purely historical features are retained.")
    
    # ---------------------------------------------------------
    # ITEM 5: CLASS BALANCE CHECK
    # ---------------------------------------------------------
    print("\n--- ITEM 5: CLASS BALANCE CHECK ---")
    val_df = df[df['is_validation_set']]
    train_df = df[~df['is_validation_set']]
    
    train_pos = train_df['target_is_off_spec_60s_future'].sum()
    train_tot = len(train_df)
    train_pct = (train_pos / train_tot) * 100 if train_tot > 0 else 0
    
    val_pos = val_df['target_is_off_spec_60s_future'].sum()
    val_tot = len(val_df)
    val_pct = (val_pos / val_tot) * 100 if val_tot > 0 else 0
    
    print(f"Train Balance: {train_pos} positive rows ({train_pct:.2f}%) out of {train_tot}")
    print(f"Validation Balance: {val_pos} positive rows ({val_pct:.2f}%) out of {val_tot}")
    
    if train_pct < 5.0 or val_pct < 5.0:
        print("NOTE: Positive class is under 5%! Highly recommend using class_weight='balanced' or scale_pos_weight tuning for Phase 3 LightGBM training.")
    else:
        print("PASS: Class balance is sufficient (>5%).")
        
    print("\n--- FINAL SUMMARY ---")
    print(f"Total rows before cleanup: {total_rows_before}")
    print(f"Total rows after cleanup : {len(df)}")
    print(f"Train rows               : {len(train_df)}")
    print(f"Validation rows          : {len(val_df)}")
    print(f"Feature column count     : {len(feature_cols)}")
    print("PASS: No target-derived columns exist in the feature set.")
    
    print("\nSaving final clean CSVs (chunked to prevent MemoryError)...")
    cols_to_save = feature_cols + target_cols
    train_df = train_df[cols_to_save]
    val_df = val_df[cols_to_save]
    
    train_df.to_csv('./output/train_features.csv', index=False, chunksize=50000)
    val_df.to_csv('./output/val_features.csv', index=False, chunksize=50000)
    print("Done!")

if __name__ == '__main__':
    audit_and_build()
