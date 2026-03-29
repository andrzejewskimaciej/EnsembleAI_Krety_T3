import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from model import FeedForwardNN

# Configuration
INPUT_FILE = "data/daily_collapsed_data.csv"
TARGET_COL = "x2_mean"
# Columns that are NOT features for the model
DROP_COLS = ["deviceId", "date", "year", "month", "split_group", "period_info"]
# Categorical columns to encode (e.g., period_type: day/night)
CAT_COLS = ["deviceType", "period_type"]

def train_and_evaluate(epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization (Daily NN) ---")
    print(f"Using device: {device}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run daily_collapse.py first.")
        return

    # Load All Data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # 0. Assign Splits based on Date (don't name it period_type)
    print("Assigning data splits based on dates...")
    df['date'] = pd.to_datetime(df['date'])
    
    # Train: Oct 2024 - Apr 2025
    # Val: May 2025 - Jun 2025
    # Test: Jul 2025 - Oct 2025
    def assign_split(dt):
        if dt < pd.Timestamp(2025, 5, 1): return 'train'
        if dt < pd.Timestamp(2025, 7, 1): return 'val'
        return 'test'
    
    df['split_group'] = df['date'].apply(assign_split)
    
    # 0b. Clean Data (Protect deviceId)
    print("Initial data cleanup...")
    initial_len = len(df)
    # 1. Drop rows with missing deviceId (critical for submission)
    df = df.dropna(subset=['deviceId'])
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with missing deviceId.")

    # 1. Handle Categorical Variables (One-Hot Encoding)
    print("Applying One-Hot encoding...")
    actual_cat_cols = [c for c in CAT_COLS if c in df.columns]
    if actual_cat_cols:
        df = pd.get_dummies(df, columns=actual_cat_cols, drop_first=True)
    
    # Split by split_group
    train_df = df[df['split_group'] == 'train'].copy()
    eval_df = df[df['split_group'].isin(['val', 'test'])].copy()
    
    # 1c. Clean TRAIN Data specifically
    print("Cleaning training data...")
    train_initial_len = len(train_df)
    # Important: Only drop NaNs from training. Eval set MUST have NaNs to be predicted.
    train_df = train_df.replace({TARGET_COL: [np.inf, -np.inf]}, np.nan).dropna(subset=[TARGET_COL])
    
    # Replace any remaining infs in features with 0
    train_df = train_df.replace([np.inf, -np.inf], 0)
    eval_df = eval_df.replace([np.inf, -np.inf], 0)
    
    print(f"Final training set size: {len(train_df)} (Dropped {train_initial_len - len(train_df)} missing target rows).")
    
    if train_df.empty:
        print("Error: No valid training data found after cleaning.")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    
    # Identify feature columns (all except target and drop_cols)
    feature_cols = [c for c in train_df.columns if c not in (DROP_COLS + [TARGET_COL])]
    
    # Fill remaining NaNs ONLY in features (to avoid zeroing deviceId if any survived)
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    eval_df[feature_cols] = eval_df[feature_cols].fillna(0)
    
    X_train_df = train_df[feature_cols]
    X_eval_df = eval_df[feature_cols]
    y_train = train_df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
    
    # Final safety fill
    X_train_df = X_train_df.fillna(0)
    X_eval_df = X_eval_df.fillna(0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values).astype(np.float32)
    X_eval = scaler.transform(X_eval_df.values).astype(np.float32)
    
    # Convert to Tensors
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    # 3. Model Setup
    input_dim = X_train.shape[1]
    print(f"Input Dimension: {input_dim}")
    model = FeedForwardNN(input_dim=input_dim, hidden_dim=128, hidden_dim_2=64).to(device)
    
    criterion = nn.L1Loss() # MAE
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training
    print("\n--- Training ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg MAE: {epoch_loss/len(train_loader):.6f}")

    # 5. Prediction & Submission
    print("\n--- Generating Submission ---")
    model.eval()
    with torch.no_grad():
        X_eval_tensor = torch.tensor(X_eval).to(device)
        preds = model(X_eval_tensor).cpu().numpy().flatten()
    
    eval_df['prediction'] = preds
    
    # Aggregation to monthly (recover year and month from date)
    eval_df['year'] = eval_df['date'].dt.year
    eval_df['month'] = eval_df['date'].dt.month
    
    submission = eval_df.groupby(['deviceId', 'year', 'month'])['prediction'].mean().reset_index()
    submission = submission[['deviceId', 'year', 'month', 'prediction']]
    
    output_path = "data/submission_daily_nn.csv"
    submission.to_csv(output_path, index=False)
    print(f"SUCCESS: Daily NN Submission saved to {output_path}")
    print(f"Total rows: {len(submission)}")

if __name__ == "__main__":
    train_and_evaluate()
