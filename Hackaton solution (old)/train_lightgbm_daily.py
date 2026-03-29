import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse

# Configuration
INPUT_FILE = "data/daily_collapsed_data.csv"
TARGET_COL = "x2_mean"
# Columns that are NOT features
DROP_COLS = ["deviceId", "date", "year", "month", "split_group", "period_info"]
# Categorical columns
CAT_COLS = ["deviceType", "period_type"]

def train_and_evaluate():
    print("--- LightGBM Pipeline (Daily) ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run daily_collapse.py first.")
        return

    # Load All Data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # 0. Assign Splits based on Date (don't name it period_type)
    print("Assigning data splits based on dates...")
    df['date'] = pd.to_datetime(df['date'])
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

    # 2. Prepare Features
    # Identify feature columns
    feature_cols = [c for c in train_df.columns if c not in (DROP_COLS + [TARGET_COL])]
    
    # Fill remaining NaNs ONLY in features (to avoid zeroing deviceId if any survived)
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    eval_df[feature_cols] = eval_df[feature_cols].fillna(0)
    
    y_train = train_df[TARGET_COL].values
    X_train = train_df[feature_cols]
    X_eval = eval_df[feature_cols]
    
    # 3. Train LightGBM
    print("\n--- Training Model ---")
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=1000)
    print("Training complete.")

    # 4. Prediction & Aggregation
    print("\n--- Generating Submission ---")
    preds = model.predict(X_eval)
    eval_df['prediction'] = preds
    
    # Aggregation to monthly (recover from date)
    eval_df['year'] = eval_df['date'].dt.year
    eval_df['month'] = eval_df['date'].dt.month
    
    submission = eval_df.groupby(['deviceId', 'year', 'month'])['prediction'].mean().reset_index()
    submission = submission[['deviceId', 'year', 'month', 'prediction']]
    
    output_path = "data/submission_daily_lgb.csv"
    submission.to_csv(output_path, index=False)
    print(f"SUCCESS: Daily LightGBM Submission saved to {output_path}")
    print(f"Total rows: {len(submission)}")

if __name__ == "__main__":
    train_and_evaluate()
