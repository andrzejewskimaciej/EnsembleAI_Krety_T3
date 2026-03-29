import pandas as pd
import os
import argparse

def split_csv(input_file, output_dir, chunksize=500000):
    """
    Split a large CSV file into smaller chunks to fit git limits (e.g. < 100MB).
    chunksize of 500000 usually results in ~50-80MB files depending on the number of columns.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading {input_file} in chunks of {chunksize} rows...")
    
    # Use an iterator to read the CSV in chunks
    chunk_iterator = pd.read_csv(input_file, chunksize=chunksize)
    
    for i, chunk in enumerate(chunk_iterator):
        output_file = os.path.join(output_dir, f"data_chunk_{i:04d}.csv")
        print(f"Writing chunk {i} to {output_file}...")
        
        # Save chunk to csv
        chunk.to_csv(output_file, index=False)
        
    print("Done splitting the dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large CSV into chunks")
    parser.add_argument("--input", default="data/data.csv", help="Path to input CSV file")
    parser.add_argument("--output_dir", default="data/split", help="Directory to save chunks")
    parser.add_argument("--chunksize", type=int, default=500000, help="Number of rows per chunk")
    
    args = parser.parse_args()
    
    split_csv(args.input, args.output_dir, args.chunksize)
