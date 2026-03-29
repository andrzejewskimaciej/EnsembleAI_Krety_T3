import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from model import FeedForwardNN

# Configuration
INPUT_DIR = "data/engineered_devices"
TARGET_COL = "x2"
DROP_COLS = ["deviceId", "period", "deviceType", "latitude", "longitude", "Timedate", "date"]

def load_and_preprocess_file(file_path, scaler=None, is_train=True):
    try:
        # Use engine='python' to be more robust against file corruption/formatting issues
        df = pd.read_csv(file_path, engine='python')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None
    
    # Identify periods
    train_df = df[df['period'] == 'train'].copy()
    val_df = df[df['period'] == 'val'].copy()
    test_df = df[df['period'] == 'test'].copy()
    
    selected_df = train_df if is_train else pd.concat([val_df, test_df])
    if selected_df.empty:
        return None, None, None
        
    # Drop columns not for training
    y = selected_df[TARGET_COL].values
    
    # Track available aggregation columns (year/month/day/hour might be available)
    agg_cols_available = [c for c in ["year", "month", "day", "hour"] if c in selected_df.columns]
    meta = selected_df[["deviceId"] + agg_cols_available].copy()
    
    # Drop all non-numeric/reserved columns
    X_df = selected_df.drop(columns=DROP_COLS + [TARGET_COL] + ["year", "month", "day", "hour"], errors='ignore')
    
    # Fill remaining NaNs (e.g. from lags)
    X_df = X_df.fillna(0)
    
    X = X_df.values.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    
    if is_train:
        if scaler is not None:
            X = scaler.fit_transform(X)
    else:
        if scaler is not None:
            X = scaler.transform(X)
            
    return torch.tensor(X), torch.tensor(y), meta

def train_and_evaluate(subset_n=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Using device: {device}")
    
    # 0. Get all files
    full_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not full_files:
        print(f"Error: No engineered files found in {INPUT_DIR}")
        return
    
    # 1. Define training files (subset or all)
    if subset_n:
        print(f"Mode: Subset training (first {subset_n} devices)")
        train_files = full_files[:subset_n]
    else:
        print(f"Mode: Full training ({len(full_files)} devices)")
        train_files = full_files

    # 1b. Fit scaler on a subset of training data
    print("\n--- Preprocessing ---")
    print("Fitting scaler on available training data...")
    scaler = StandardScaler()
    fitted_count = 0
    for f in train_files[:20]: # Fit on up to 20 files
        X, y, _ = load_and_preprocess_file(f, scaler=None, is_train=True)
        if X is not None:
            scaler.partial_fit(X.numpy())
            fitted_count += 1
    print(f"Scaler fitted on {fitted_count} files.")
            
    # 2. Initialize Model
    # Get input dimension from first valid training file
    input_dim = None
    for f in train_files:
        X_sample, _, _ = load_and_preprocess_file(f, scaler=scaler, is_train=True)
        if X_sample is not None:
            input_dim = X_sample.shape[1]
            break
            
    if input_dim is None:
        print("Error: Could not find any valid training data to determine input dimension.")
        return
        
    print(f"Model Input Dimension: {input_dim}")
    model = FeedForwardNN(input_dim=input_dim).to(device)
    
    criterion = nn.L1Loss() # MAE
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    print("\n--- Training ---")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        
        for i, f in enumerate(train_files):
            X, y, _ = load_and_preprocess_file(f, scaler=scaler, is_train=True)
            if X is None: continue
            
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
            if (i + 1) % 10 == 0 or (i + 1) == len(train_files):
                print(f"Epoch {epoch+1}/{epochs} | Progress: {i+1}/{len(train_files)} devices | Batch Loss: {loss.item():.6f}", end='\r')
            
        if count > 0:
            print(f"\nEpoch {epoch+1}/{epochs} Completed. Average MAE: {total_loss/count:.6f}")

    # 4. Evaluation and Aggregation (ALWAYS on full set of files)
    print("\n--- Evaluation & Aggregation (All Devices) ---")
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for i, f in enumerate(full_files):
            X, y, meta = load_and_preprocess_file(f, scaler=scaler, is_train=False)
            if X is None: continue
            
            X = X.to(device)
            preds = model(X).cpu().numpy().flatten()
            
            meta['prediction'] = preds
            all_results.append(meta)
            if (i + 1) % 10 == 0 or (i + 1) == len(full_files):
                print(f"Predicting: {i+1}/{len(full_files)} devices", end='\r')
            
    if not all_results:
        print("\nNo validation/test data found for aggregation.")
        return
        
    print("\nMerging results and calculating averages...")
    final_df = pd.concat(all_results)
    
    # Submission columns: deviceId, year, month, prediction
    agg_cols = [c for c in ["year", "month"] if c in final_df.columns]
    
    if "year" in agg_cols and "month" in agg_cols:
        submission = final_df.groupby(['deviceId', 'year', 'month'])['prediction'].mean().reset_index()
        # Ensure correct column order for submission
        submission = submission[['deviceId', 'year', 'month', 'prediction']]
        output_path = "data/submission.csv"
        submission.to_csv(output_path, index=False)
        print(f"SUCCESS: Submission file saved to {output_path}")
        print(f"Total rows in submission: {len(submission)}")
    else:
        print("Warning: 'year' or 'month' columns missing. Check if add_dates_back.py was run.")
        output_path = "data/preliminary_results.csv"
        final_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path} (full metadata included)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and Evaluate NN Model")
    parser.add_argument("--subset", type=int, default=None, help="Number of devices to use for training (subset)")
    args = parser.parse_args()
    
    train_and_evaluate(subset_n=args.subset)
