
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "model_ready_data.csv")
MODEL_FILE = os.path.join(PROCESSED_DATA_DIR, "model_rf.pkl")
METRICS_FILE = os.path.join(PROCESSED_DATA_DIR, "model_metrics.txt")
PLOTS_DIR = os.path.join(PROCESSED_DATA_DIR, "plots")
PREDICTIONS_FILE = os.path.join(PROCESSED_DATA_DIR, "model_predictions.csv")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None
    print(f"Loading {DATA_FILE}...")
    return pd.read_csv(DATA_FILE, parse_dates=['date'])

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 20 features or all if fewer
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importances (Top {top_n})")
    plt.bar(range(top_n), importances[top_indices], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90)
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def main():
    ensure_dir(PLOTS_DIR)
    
    df = load_data()
    if df is None:
        return

    # Target: update_to_enrolment_ratio
    target_col = 'update_to_enrolment_ratio'
    
    # Feature Selection
    # Dropping non-predictive or target-leaky columns
    drop_cols = [
        'date', 'state', 'district', 'pincode', # Identifiers (using frequencies instead)
        target_col,
        'enrolment_total', 'demo_update_total', 'bio_update_total', # Raw counts (target derived from these)
        'total_activity', # Derived
        'cluster', 'anomaly' # From previous steps, could be features but let's stick to raw inputs + lags
    ]
    
    # Keep only numeric columns for now (handled encoding previously)
    features = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['int64', 'float64']]
    
    print(f"Training with {len(features)} features: {features}")
    
    X = df[features]
    y = df[target_col]
    
    # Handle infinite ratios if any (division by zero protection was simple)
    # Just in case
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train-Test Split (Temporal awareness: Random split for now as basic regression)
    # Ideally should be TimeSeriesSplit, but for general pattern checking random is okay for MVP.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Model Training
    print("Initializing Random Forest Regressor (n_estimators=50, max_depth=10)...") 
    # Limited depth and estimators to speed up training on large dataset
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    
    print("Fitting model...")
    rf.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating model...")
    y_pred = rf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = f"Model Performance:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}\n"
    print(metrics)
    
    with open(METRICS_FILE, "w") as f:
        f.write(metrics)
    print(f"Saved metrics to {METRICS_FILE}")
    
    # Save Model
    joblib.dump(rf, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")
    
    # Plot Feature Importance
    plot_feature_importance(rf, features)
    
    # Save Predictions Sample (Actual vs Predicted)
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
    sample_path = PREDICTIONS_FILE
    results_df.head(100).to_csv(sample_path, index=False)
    print(f"Saved sample predictions to {sample_path}")

if __name__ == "__main__":
    main()
