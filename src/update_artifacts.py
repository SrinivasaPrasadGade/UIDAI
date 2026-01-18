import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "model_ready_data.csv")
MODEL_FILE = os.path.join(PROCESSED_DATA_DIR, "model_rf.pkl")
METRICS_FILE = os.path.join(PROCESSED_DATA_DIR, "model_metrics.txt")
PLOTS_DIR = os.path.join(PROCESSED_DATA_DIR, "plots")

def update_artifacts():
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        print("Error: Model or Data file missing.")
        return

    print("Loading resources...")
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    
    # Prepare data (using model's features)
    features = model.feature_names_in_
    X = df[features]
    y_true = df['update_to_enrolment_ratio']
    
    # Predict
    print("Generating predictions...")
    y_pred = model.predict(X)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics_text = f"Model Performance (Optimized):\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}\n"
    print(metrics_text)
    
    with open(METRICS_FILE, "w") as f:
        f.write(metrics_text)
    print(f"Updated {METRICS_FILE}")
    
    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(features))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances (Top {top_n}) - Optimized Model")
        plt.bar(range(top_n), importances[top_indices], align="center", color='#2ca02c') # Green for optimized
        plt.xticks(range(top_n), [features[i] for i in top_indices], rotation=90)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Updated {plot_path}")

if __name__ == "__main__":
    update_artifacts()
