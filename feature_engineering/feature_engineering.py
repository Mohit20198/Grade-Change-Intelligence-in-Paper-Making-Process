import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def build_features(input_data_path, event_log_path, output_dir):
    print(f"Loading data from {input_data_path}...")
    df = pd.read_csv(input_data_path, low_memory=False)
    events = pd.read_csv(event_log_path)
    
    raw_tags = ['stock_flow', 'steam_pressure', 'machine_speed', 'filler_flow', 'basis_weight', 'moisture', 'ash']
    
    print("Engineering features...")
    # 1. Lags
    for tag in raw_tags:
        df[f'{tag}_lag_1s'] = df[tag].shift(1)
        df[f'{tag}_lag_5s'] = df[tag].shift(5)
        df[f'{tag}_lag_15s'] = df[tag].shift(15)
        
    # 2. Rolling stats (5-min = 300s, 15-min = 900s)
    for tag in raw_tags:
        df[f'{tag}_roll_mean_5m'] = df[tag].rolling(300, min_periods=1).mean()
        df[f'{tag}_roll_std_5m'] = df[tag].rolling(300, min_periods=1).std()
        df[f'{tag}_roll_mean_15m'] = df[tag].rolling(900, min_periods=1).mean()
        df[f'{tag}_roll_std_15m'] = df[tag].rolling(900, min_periods=1).std()
        
    # 3. Rates of change (1-sample and 5-sample windows)
    for tag in raw_tags:
        df[f'{tag}_roc_1s'] = df[tag].diff(1)
        df[f'{tag}_roc_5s'] = df[tag].diff(5)
        
    print("Creating labels...")
    # Label 1: actual future basis_weight at t+60s and t+300s
    df['target_bw_60s_future'] = df['basis_weight'].shift(-60)
    df['target_bw_300s_future'] = df['basis_weight'].shift(-300)
    
    # Label 2: binary flag: does basis_weight deviate >2.5% from setpoint within next 60s / 300s
    df['bw_deviation'] = (df['basis_weight'] - df['bw_setpoint']).abs() / df['bw_setpoint']
    is_off_spec = df['bw_deviation'] > 0.025
    
    # Reversing the series allows rolling to look into the "future" (i.e. next rows)
    # rolling(60) on reversed series gives max over [t, t+59]. 
    # shift(-1) adjusts this to [t+1, t+60].
    future_off_spec_60s = is_off_spec.iloc[::-1].rolling(60, min_periods=1).max().iloc[::-1].shift(-1)
    df['target_is_off_spec_60s_future'] = (future_off_spec_60s > 0).astype(int)
    
    future_off_spec_300s = is_off_spec.iloc[::-1].rolling(300, min_periods=1).max().iloc[::-1].shift(-1)
    df['target_is_off_spec_300s_future'] = (future_off_spec_300s > 0).astype(int)
    
    # Drop rows where we don't have a future target (the last 300 seconds of the simulation)
    df = df.dropna(subset=['target_bw_300s_future'])
    
    print("Splitting data into train/validation based on event windows...")
    val_events = events[events['is_validation'] == True]['event_id'].tolist()
    
    df['is_validation_set'] = df['event_id'].isin(val_events)
    
    val_df = df[df['is_validation_set']]
    train_df = df[~df['is_validation_set']]
    
    cols_to_drop = ['bw_deviation', 'is_validation_set']
    val_df = val_df.drop(columns=cols_to_drop)
    train_df = train_df.drop(columns=cols_to_drop)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    train_path = out_dir / "train_features.csv"
    val_path = out_dir / "val_features.csv"
    
    print("Saving feature matrices...")
    train_df.to_csv(train_path, index=False, chunksize=50000)
    val_df.to_csv(val_path, index=False, chunksize=50000)
    print(f"Saved to {train_path} and {val_path}")
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./output/process_data.csv", help="Path to process_data.csv")
    parser.add_argument("--events", default="./output/grade_change_log.csv", help="Path to grade_change_log.csv")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    args = parser.parse_args()
    
    build_features(args.data, args.events, args.output_dir)
