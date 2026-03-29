import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
from sklearn.metrics import mean_absolute_error

# Configuration
INPUT_DIR = "data/engineered_devices"
TARGET_COL = "x2"
DROP_COLS = ["deviceId", "period", "deviceType", "latitude", "longitude", "Timedate", "date"]

def load_data_for_training(files, max_files=50):
    """
    Loads a subset of files into memory for training.
    """
    all_X = []
    all_y = []
    
    print(f"Loading {min(len(files), max_files)} devices for training...")
    
    for i, f in enumerate(files[:max_files]):
        try:
            df = pd.read_csv(f, engine='python')
            train_df = df[df['period'] == 'train'].copy()
            if train_df.empty:
                continue
            
            y = train_df[TARGET_COL].values
            # Drop non-feature columns
            X = train_df.drop(columns=DROP_COLS + [TARGET_COL] + ["year", "month", "day", "hour"], errors='ignore')
            X = X.fillna(0)
            
            all_X.append(X)
            all_y.append(y)
            
            if (i + 1) % 10 == 0:
                print(f"Loaded {i+1} files...", end='\r')
        except Exception as e:
            print(f"\nError loading {f}: {e}")
            
    if not all_X:
        return None, None
        
    X_train = pd.concat(all_X)
    y_train = np.concatenate(all_y)
    print(f"\nTraining data ready. Shape: {X_train.shape}")
    return X_train, y_train

def train_and_evaluate(train_subset=50):
    print("--- LightGBM Pipeline ---")
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not files:
        print(f"Error: No engineered files found in {INPUT_DIR}")
        return

    # 1. Load training data
    X_train, y_train = load_data_for_training(files, max_files=train_subset)
    if X_train is None:
        print("Error: No training data found.")
        return
        
    # 2. Train LightGBM
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
    model = lgb.train(params, dtrain, num_boost_round=500)
    print("Training complete.")

    # 3. Efficient Prediction & Aggregation
    print("\n--- Predicting & Aggregating ---")
    all_results = []
    
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f, engine='python')
            eval_mask = df['period'].isin(['val', 'test'])
            eval_df = df[eval_mask].copy()
            
            if eval_df.empty:
                continue
                
            # Prepare features
            X_eval = eval_df.drop(columns=DROP_COLS + [TARGET_COL] + ["year", "month", "day", "hour"], errors='ignore')
            X_eval = X_eval.fillna(0)
            
            # Predict
            preds = model.predict(X_eval)
            
            # Metadata for aggregation
            meta = eval_df[['deviceId', 'year', 'month']].copy()
            meta['prediction'] = preds
            all_results.append(meta)
            
            if (i + 1) % 20 == 0 or (i + 1) == len(files):
                print(f"Processed {i+1}/{len(files)} devices for prediction", end='\r')
        except Exception as e:
            print(f"\nError processing {f} for prediction: {e}")

    if not all_results:
        print("\nNo validation/test data found for aggregation.")
        return
        
    print("\nGenerating final submission...")
    final_df = pd.concat(all_results)
    
    # Check for necessary columns
    if 'year' in final_df.columns and 'month' in final_df.columns:
        submission = final_df.groupby(['deviceId', 'year', 'month'])['prediction'].mean().reset_index()
        submission = submission[['deviceId', 'year', 'month', 'prediction']]
        output_path = "data/submission_lgb.csv"
        submission.to_csv(output_path, index=False)
        print(f"SUCCESS: LightGBM Submission saved to {output_path}")
        print(f"Total submission rows: {len(submission)}")
    else:
        print("Warning: 'year' or 'month' columns missing. Check if add_dates_back.py was run.")
        output_path = "data/lgb_results_raw.csv"
        final_df.to_csv(output_path, index=False)
        print(f"Raw results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM on Engineered Data")
    parser.add_argument("--train_subset", type=int, default=50, help="Number of devices to use for training dataset")
    args = parser.parse_args()
    
    train_and_evaluate(train_subset=args.train_subset)
