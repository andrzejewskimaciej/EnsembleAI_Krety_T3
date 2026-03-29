import os
import glob
import pandas as pd
import numpy as np
from datetime import time

# Configuration
INPUT_DIR = "data/engineered_devices"
OUTPUT_FILE = "data/daily_collapsed_data.csv"

def get_day_period(hour):
    if 0 <= hour < 6: return 'night'
    if 6 <= hour < 12: return 'morning'
    if 12 <= hour < 18: return 'noon'
    return 'evening'

def calculate_daily_dynamics(df):
    """
    Collapses 5-min data into daily features capturing intra-day dynamics.
    """
    if df.empty:
        return None
    
    # Ensure Timedate is datetime
    df['Timedate'] = pd.to_datetime(df['Timedate'])
    df['date_only'] = df['Timedate'].dt.date
    df['hour'] = df['Timedate'].dt.hour
    df['period'] = df['hour'].apply(get_day_period)
    
    # Group by date
    daily_groups = df.groupby('date_only')
    
    daily_features = []
    
    for date, group in daily_groups:
        res = {}
        res['deviceId'] = group['deviceId'].iloc[0]
        res['deviceType'] = group['deviceType'].iloc[0] if 'deviceType' in group.columns else 0
        res['period_info'] = group['period'].iloc[0] # Original split info from source if available
        
        # 1. Target and Basic
        res['x2_mean'] = group['x2'].mean()
        
        # 2. Period-specific temperatures (t1 is outdoor)
        for p in ['night', 'morning', 'noon', 'evening']:
            p_mask = group['period'] == p
            res[f't1_{p}_mean'] = group.loc[p_mask, 't1'].mean() if p_mask.any() else np.nan
            res[f't2_{p}_mean'] = group.loc[p_mask, 't2'].mean() if p_mask.any() else np.nan

        # 3. Slopes / Dynamics
        # Morning rise: slope of t1 between 6:00 and 10:00
        morning_mask = (group['hour'] >= 6) & (group['hour'] <= 10)
        if morning_mask.sum() > 1:
            m_data = group.loc[morning_mask].sort_values('Timedate')
            res['t1_morning_rise_slope'] = np.polyfit(np.arange(len(m_data)), m_data['t1'].values, 1)[0]
        else:
            res['t1_morning_rise_slope'] = 0

        # Evening fall: slope of t1 between 18:00 and 22:00
        evening_mask = (group['hour'] >= 18) & (group['hour'] <= 22)
        if evening_mask.sum() > 1:
            e_data = group.loc[evening_mask].sort_values('Timedate')
            res['t1_evening_fall_slope'] = np.polyfit(np.arange(len(e_data)), e_data['t1'].values, 1)[0]
        else:
            res['t1_evening_fall_slope'] = 0

        # Daily Change (Max - Min)
        res['t1_daily_range'] = group['t1'].max() - group['t1'].min()
        res['t2_daily_drift'] = group['t2'].iloc[-1] - group['t2'].iloc[0] # internal temp change

        # 4. Weather data (if exists) - take daily mean
        weather_cols = [c for c in group.columns if 'weather_' in c or c in ['temperature_2m_mean', 'precipitation_sum']]
        for col in weather_cols:
            res[col] = group[col].mean()

        daily_features.append(res)
        
    return pd.DataFrame(daily_features)

def run_collapse():
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(files)} files to collapse.")
    
    all_daily_data = []
    
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f, engine='python')
            daily_df = calculate_daily_dynamics(df)
            if daily_df is not None:
                all_daily_data.append(daily_df)
                
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"Processed {i+1}/{len(files)} devices", end='\r')
        except Exception as e:
            print(f"\nError processing {f}: {e}")

    if not all_daily_data:
        print("\nNo data processed.")
        return

    print("\nMerging all results into final DataFrame...")
    final_df = pd.concat(all_daily_data, ignore_index=True)
    
    # Save results
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Daily collapsed data saved to {OUTPUT_FILE}")
    print(f"Final shape: {final_df.shape}")

if __name__ == "__main__":
    run_collapse()
