import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "processed_data", "model_ready_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "processed_data", "model_rf.pkl")

def check_accuracy():
    # 1. Load Data & Model
    print(f"Loading data from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print("ERROR: Data file not found!")
        return
        
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    # 2. Prepare Features (Align with Model)
    # We filter the data to only use columns the model actually knows
    expected_cols = getattr(model, "feature_names_in_", None)
    
    if expected_cols is None:
        print("Error: This model file doesn't have feature names saved inside it.")
        return

    # Create input X and target y
    X = df[expected_cols].fillna(0)
    y_true = df['update_to_enrolment_ratio']

    # 3. Predict & Score
    print("Calculating accuracy metrics...")
    y_pred = model.predict(X)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # 4. Print Results
    print("\n" + "="*40)
    print("      MODEL ACCURACY REPORT      ")
    print("="*40)
    print(f"R-Squared (Accuracy):  {r2:.4f}")
    print(f"Mean Error (MAE):      {mae:.4f}")
    print("="*40)
    
    # Interpretation Guide
    if r2 > 0.7:
        print("✅ VERDICT: EXCELLENT. The model is highly accurate.")
    elif r2 > 0.4:
        print("⚠️ VERDICT: OKAY. The model captures trends but misses details.")
    else:
        print("❌ VERDICT: POOR. The model is mostly guessing. Try XGBoost.")

if __name__ == "__main__":
    check_accuracy()
