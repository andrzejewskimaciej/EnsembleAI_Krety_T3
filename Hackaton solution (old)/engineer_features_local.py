import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from scipy.spatial import KDTree

def get_nearest_weather_stations(devices_df, weather_df):
    """
    Creates a mapping from each deviceId to its nearest weather station coordinates.
    Uses KDTree for efficient nearest neighbor search.
    """
    print("Calculating nearest weather stations for all devices...")
    
    # Get unique weather locations
    weather_locs = weather_df[['latitude', 'longitude']].drop_duplicates()
    
    # Build KDTree from weather locations
    tree = KDTree(weather_locs[['latitude', 'longitude']].values)
    
    # Map for each device
    mapping = {}
    for idx, row in devices_df.iterrows():
        dist, loc_idx = tree.query([row['latitude'], row['longitude']])
        nearest_loc = weather_locs.iloc[loc_idx]
        mapping[row['deviceId']] = {
            'w_lat': nearest_loc['latitude'],
            'w_lon': nearest_loc['longitude']
        }
        
    print(f"Mapping complete for {len(mapping)} devices.")
    return mapping

def engineer_single_device(df, device_meta, weather_df, mapping):
    """
    Calculates features for a single device, joining with the nearest weather data.
    """
    # 1. Coordinate normalization for this device
    device_id = df['deviceId'].iloc[0]
    w_coords = mapping.get(device_id)
    
    if not w_coords:
        return df # Should not happen if devices.csv is complete

    # 2. Join with weather
    df['Timedate'] = pd.to_datetime(df['Timedate'])
    df = df.sort_values('Timedate')
    df['date'] = df['Timedate'].dt.date.astype(str)
    
    # Filter weather for the nearest location
    loc_weather = weather_df[
        (weather_df['latitude'] == w_coords['w_lat']) & 
        (weather_df['longitude'] == w_coords['w_lon'])
    ].copy()
    
    if not loc_weather.empty:
        # Avoid column collisions on lat/lon
        df = df.merge(loc_weather, left_on='date', right_on='date', how='left', suffixes=('', '_weather'))
    
    # 3. Time-based features
    df['year'] = df['Timedate'].dt.year
    df['month'] = df['Timedate'].dt.month
    df['day'] = df['Timedate'].dt.day
    df['hour'] = df['Timedate'].dt.hour
    df['dayofweek'] = df['Timedate'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclic
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # 4. Rolling Features (Time Series)
    # Important columns from task: t1, t2, x1, x2 (if available)
    raw_cols = [c for c in ['t1', 't2', 'x1', 'x2'] if c in df.columns]
    for col in raw_cols:
        df[f'{col}_roll_mean_1h'] = df[col].rolling(window=12, min_periods=1).mean()
        df[f'{col}_lag_24h'] = df[col].shift(288) # 24h lag

    # 5. Weather Rolling
    weather_cols = ['temperature_2m_mean', 'precipitation_sum']
    for col in weather_cols:
        if col in df.columns:
            # 3. 3-day rolling weather to capture climate inertia
            df[f'weather_{col}_roll_3d'] = df[col].rolling(window=3*288, min_periods=1).mean()

    # 6. Keep all temporal columns (needed for model aggregation/submission)
    # Timedate is converted to string for CSV saving to ensure consistency
    df['Timedate'] = df['Timedate'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df

def run_engineering(input_dir, output_dir, devices_path, weather_path):
    """
    Main processing loop.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load auxiliary data
    print("Loading auxiliary data...")
    devices_df = pd.read_csv(devices_path)
    weather_df = pd.read_csv(weather_path)
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date.astype(str)
    
    # Pre-calculate spatial mapping
    mapping = get_nearest_weather_stations(devices_df, weather_df)
    
    # Get all per-device files
    device_files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(device_files)} devices to process.")
    
    for i, file_path in enumerate(device_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_dir, filename)
        
        # Robustness: skip already processed
        if os.path.exists(out_path):
            continue
            
        print(f"[{i+1}/{len(device_files)}] Engineering {filename}...")
        
        try:
            df = pd.read_csv(file_path)
            if df.empty: continue
            
            # Use columns mapping to handle names
            col_mapping = {'deviceid': 'deviceId', 'timedate': 'Timedate'}
            df = df.rename(columns={c: col_mapping.get(str(c).lower(), c) for c in df.columns})
            
            df_final = engineer_single_device(df, devices_df, weather_df, mapping)
            df_final.to_csv(out_path, index=False)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # Keep going for other devices, but log the error
            continue

    print("Offline Feature Engineering complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust per-device offline feature engineering")
    parser.add_argument("--input_dir", default="data/devices_raw", help="Dir with per-device CSVs")
    parser.add_argument("--output_dir", default="data/engineered_devices", help="Output directory")
    parser.add_argument("--devices_csv", default="data/devices.csv", help="Metadata file")
    parser.add_argument("--weather_csv", default="data/weather_daily_updates.csv", help="Weather file")
    
    args = parser.parse_args()
    
    run_engineering(args.input_dir, args.output_dir, args.devices_csv, args.weather_csv)
