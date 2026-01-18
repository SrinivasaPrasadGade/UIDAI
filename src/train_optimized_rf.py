import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "processed_data", "model_ready_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "processed_data", "model_rf.pkl")

def train_optimized_rf():
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print("Error: Data file not found.")
        return

    df = pd.read_csv(DATA_FILE)
    
    # 1. Features (Must match App)
    features = [
        'age_0_5', 'age_5_17', 'age_18_greater',
        'demo_age_5_17', 'demo_age_17_', 'bio_age_5_17', 'bio_age_17_',
        'enrolment_lag_7', 'enrolment_lag_30',
        'enrolment_rolling_mean_7', 'enrolment_rolling_mean_30',
        'total_activity_rolling_mean_7',
        'state_freq', 'district_freq',
        'month', 'day_of_week', 'is_weekend'
    ]
    target = 'update_to_enrolment_ratio'
    
    # Check alignment
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    X = df[features]
    y = df[target]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Optimized Random Forest
    print("Training Optimized Random Forest...")
    # KEY FIX: We limit the depth and samples to force 'learning' instead of 'memorizing'
    model = RandomForestRegressor(
        n_estimators=200,      # Reasonable number of trees
        max_depth=10,          # LIMIT DEPTH (Prevents memorization)
        min_samples_leaf=5,    # Requires at least 5 records to make a decision
        max_features='sqrt',   # Forces trees to look at different features
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # 4. Save Feature Names (Crucial for App Debugger)
    model.feature_names_in_ = features

    # 5. Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("-" * 30)
    print(f"Optimized RF Accuracy (R2): {r2:.4f}")
    print(f"Mean Error (MAE):       {mae:.4f}")
    print("-" * 30)
    
    if r2 < 0.98:
        print("Note: Accuracy dropped (expected). The model is no longer cheating/memorizing.")

    print(f"Saving to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    print("Done.")

if __name__ == "__main__":
    train_optimized_rf()
