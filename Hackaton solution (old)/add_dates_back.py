import pandas as pd
import os
import glob
import argparse

def add_dates(eng_dir, raw_dir):
    """
    Restores year and month columns to engineered files by pulling 
    them from the corresponding raw device files.
    """
    eng_files = glob.glob(os.path.join(eng_dir, "*.csv"))
    print(f"Found {len(eng_files)} engineered files to update.")

    for i, eng_file in enumerate(eng_files):
        filename = os.path.basename(eng_file)
        raw_file = os.path.join(raw_dir, filename)

        if not os.path.exists(raw_file):
            print(f"Warning: Raw file not found for {filename}. Skipping.")
            continue

        if i % 50 == 0:
            print(f"Updating {i}/{len(eng_files)}: {filename}")

        try:
            # Load both files
            df_eng = pd.read_csv(eng_file)
            df_raw = pd.read_csv(raw_file)

            # Ensure Timedate is present and parsed in raw
            col_mapping = {'timedate': 'Timedate'}
            df_raw = df_raw.rename(columns={c: col_mapping.get(str(c).lower(), c) for c in df_raw.columns})
            
            if 'Timedate' not in df_raw.columns:
                print(f"Error: No 'Timedate' in {filename} (raw). Skipping.")
                continue
            
            df_raw['Timedate'] = pd.to_datetime(df_raw['Timedate'])
            
            # Extract full date and temporal components
            df_eng['Timedate'] = df_raw['Timedate'].values
            df_eng['year'] = df_raw['Timedate'].dt.year.values
            df_eng['month'] = df_raw['Timedate'].dt.month.values
            df_eng['day'] = df_raw['Timedate'].dt.day.values
            df_eng['hour'] = df_raw['Timedate'].dt.hour.values
            
            # Save the updated engineered file
            df_eng.to_csv(eng_file, index=False)

        except Exception as e:
            print(f"Error updating {filename}: {e}")

    print("Temporal data restoration complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore date columns to engineered files")
    parser.add_argument("--eng_dir", default="data/engineered_devices", help="Dir with engineered CSVs")
    parser.add_argument("--raw_dir", default="data/devices_raw", help="Dir with raw CSVs")
    
    args = parser.parse_args()
    add_dates(args.eng_dir, args.raw_dir)
