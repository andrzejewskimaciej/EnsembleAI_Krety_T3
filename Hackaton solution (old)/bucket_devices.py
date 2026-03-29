import pandas as pd
import os
import glob
import argparse
from pathlib import Path

def bucket_data(input_dir, output_dir, state_file="bucketing_progress.txt"):
    """
    Reads split CSV chunks and appends rows to per-device CSV files.
    Maintains a state file to resume progress.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load progress
    processed_chunks = set()
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            processed_chunks = set(line.strip() for line in f)
        print(f"Resuming: {len(processed_chunks)} chunks already processed.")

    # Get all split files
    split_files = sorted(glob.glob(os.path.join(input_dir, 'data_chunk_*.csv')))
    if not split_files:
        print(f"No split files found in {input_dir}")
        return

    # To efficiently track if a file already has a header, 
    # we'll look at the current files in the output directory
    existing_files = {os.path.basename(f) for f in glob.glob(os.path.join(output_dir, '*.csv'))}

    for i, file_path in enumerate(split_files):
        filename = os.path.basename(file_path)
        
        if filename in processed_chunks:
            continue

        print(f"[{i+1}/{len(split_files)}] Processing {filename}...")
        
        try:
            # Read chunk
            df = pd.read_csv(file_path)
            
            # Normalize column names (case-insensitive)
            col_mapping = {'deviceid': 'deviceId', 'timedate': 'Timedate'}
            df = df.rename(columns={c: col_mapping.get(str(c).lower(), c) for c in df.columns})
            
            if 'deviceId' not in df.columns:
                print(f"Warning: No 'deviceId' column in {filename}. Skipping.")
                continue

            # Group by deviceId and append to individual files
            for device_id, group in df.groupby('deviceId'):
                # Sanitize device_id for filename
                safe_id = str(device_id).replace('/', '_').replace('\\', '_')
                out_path = os.path.join(output_dir, f"{safe_id}.csv")
                
                # Check if we need to write a header
                file_exists = os.path.exists(out_path)
                
                # Append to file
                group.to_csv(out_path, mode='a', index=False, header=not file_exists)

            # Record progress
            with open(state_file, 'a') as f:
                f.write(filename + '\n')
            processed_chunks.add(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            print("Stopping to prevent data corruption. Fix the issue and restart.")
            break

    print("Bucketing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket split data by deviceId")
    parser.add_argument("--input_dir", default="data/split", help="Dir with data_chunk_*.csv")
    parser.add_argument("--output_dir", default="data/devices_raw", help="Dir for per-device files")
    parser.add_argument("--state_file", default="bucketing_progress.txt", help="File to track progress")
    
    args = parser.parse_args()
    
    bucket_data(args.input_dir, args.output_dir, args.state_file)
