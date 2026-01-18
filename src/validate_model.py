import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "processed_data", "model_ready_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "processed_data", "model_rf.pkl")

def validate():
    # 1. Load Resources
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        print("Error: Missing data or model file.")
        return

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    
    print(f"Loaded Data: {df.shape}")
    print(f"Loaded Model Type: {type(model).__name__}")

    # 2. Prepare Features (Align with Model)
    # Get features the model expects
    expected_cols = model.feature_names_in_
    X = df[expected_cols].fillna(0)
    y_true = df['update_to_enrolment_ratio'] # The actual target from history

    # 3. Predict
    print("Running predictions on historical data...")
    y_pred = model.predict(X)

    # 4. Accuracy Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print("-" * 30)
    print("MODEL PERFORMANCE REPORT")
    print("-" * 30)
    print(f"1. R-Squared Score: {r2:.4f} (Closer to 1.0 is better)")
    print(f"2. Avg Error (MAE): {mae:.4f} (Lower is better)")
    
    # 5. Feature Importance (Why is it predicting high?)
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'Feature': expected_cols,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(10)
        
        print("\nTOP 10 DRIVERS OF RISK:")
        print(imp)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp, x='Importance', y='Feature', palette='viridis')
        plt.title('What drives the Model Predictions?')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    validate()
