import pandas as pd

def report_per_event_breakdown(df, event_log):
    """
    Computes a breakdown of off-spec events.
    
    df: DataFrame containing 'event_id', 'basis_weight', 'bw_setpoint', 
        'target_is_off_spec_60s_future', and 'split' (or 'is_validation_set').
    event_log: DataFrame from grade_change_log.csv
    """
    
    # We only care about events that actually went off spec
    off_spec_events = event_log[event_log['went_off_spec'] == True]['event_id'].tolist()
    
    # Map validation set to string if needed
    if 'split' not in df.columns and 'is_validation_set' in df.columns:
        df['split'] = df['is_validation_set'].map({True: 'val', False: 'train'})
        
    records = []
    
    for eid in off_spec_events:
        event_df = df[df['event_id'] == eid]
        if len(event_df) == 0:
            continue
            
        # compute actual off-spec duration
        if 'bw_deviation' not in event_df.columns:
            bw_dev = (event_df['basis_weight'] - event_df['bw_setpoint']).abs() / event_df['bw_setpoint']
        else:
            bw_dev = event_df['bw_deviation']
            
        actual_off_spec_s = (bw_dev > 0.025).sum()
        
        pos_rows = event_df['target_is_off_spec_60s_future'].sum()
        split_name = event_df['split'].iloc[0] if 'split' in event_df.columns else "unknown"
        
        records.append({
            'event_id': eid,
            'split': split_name,
            'off_spec_duration_s': actual_off_spec_s,
            'positive_rows_contributed': pos_rows
        })
        
    report_df = pd.DataFrame(records)
    report_df = report_df.sort_values(by=['split', 'event_id'])
    
    print(f"{'event_id':<10} | {'split':<10} | {'off_spec_duration_s':<20} | {'positive_rows_contributed':<25}")
    print("-" * 75)
    for _, row in report_df.iterrows():
        print(f"{row['event_id']:<10} | {row['split']:<10} | {row['off_spec_duration_s']:<20} | {row['positive_rows_contributed']:<25}")
        
    return report_df

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    
    print("Loading data for reporting...")
    df = pd.read_csv('./output/process_data.csv', low_memory=False)
    events = pd.read_csv('./output/grade_change_log.csv')
    
    # Re-create split logic exactly as in audit_features.py
    train_event_ids, val_event_ids = train_test_split(
        events['event_id'].values,
        test_size=9,
        stratify=events['went_off_spec'].values,
        random_state=42
    )
    df['split'] = df['event_id'].apply(lambda x: 'val' if x in val_event_ids else 'train')
    
    # Re-create targets for reporting
    df['bw_deviation'] = (df['basis_weight'] - df['bw_setpoint']).abs() / df['bw_setpoint']
    is_off_spec = df['bw_deviation'] > 0.025
    future_off_spec = is_off_spec.iloc[::-1].rolling(60, min_periods=1).max().iloc[::-1].shift(-1)
    df['target_is_off_spec_60s_future'] = (future_off_spec > 0).astype(int)
    
    print("\nPer-Event Breakdown (Off-spec events only):")
    report_per_event_breakdown(df, events)
