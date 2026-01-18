
import pandas as pd
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "processed_data", "master_aadhar_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "processed_data", "model_ready_data.csv")

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None
    print(f"Loading {DATA_FILE}...")
    return pd.read_csv(DATA_FILE, parse_dates=['date'])

def create_features(df):
    print("Creating features...")
    
    # Sort by Key Columns for correct lag calculation
    # We really want to see history for a specific region (Pincode)
    df = df.sort_values(by=['pincode', 'date'])
    
    # 1. Ratio Features (Can be done element-wise)
    print("- Calculating Ratios")
    # Avoid division by zero
    df['enrolment_total_safe'] = df['enrolment_total'].replace(0, 1) 
    df['update_to_enrolment_ratio'] = (df['demo_update_total'] + df['bio_update_total']) / df['enrolment_total_safe']
    df.drop(columns=['enrolment_total_safe'], inplace=True)
    
    # 2. Lag Features & Rolling Averages
    # We must group by pincode to ensure we don't lag values from a different pincode
    print("- Calculating Lags and Rolling Averages (this may take a moment)")
    
    # Define a helper to apply to each group
    # However, groupby().apply() is slow on 2M rows.
    # Faster approach: usage of groupby().shift() and groupby().rolling()
    
    grouped = df.groupby('pincode')
    
    # Lags (7 days, 30 days)
    df['enrolment_lag_7'] = grouped['enrolment_total'].shift(7)
    df['enrolment_lag_30'] = grouped['enrolment_total'].shift(30)
    
    # Rolling Means (7 days, 30 days) -> Center=False (History only)
    # Rolling in pandas requires a sorted index or on the group
    # Since we sorted by pincode, date, we can use transform but rolling is index based.
    # Set index to date for rolling, then reset? 
    # Or just use the sorted order assumption with min_periods
    
    df['enrolment_rolling_mean_7'] = grouped['enrolment_total'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['enrolment_rolling_mean_30'] = grouped['enrolment_total'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    
    df['total_activity_rolling_mean_7'] = grouped['total_activity'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    # 3. Categorical Encoding
    # 'state' is useful. 'district' might be too high cardinality for simple one-hot, 
    # but 'state' is fine (~30 values).
    # For this model, let's keep 'state' as is, or Label Encode it. 
    # Frequency encoding is also good for high cardinality.
    # Let's use Frequency Encoding for 'state' and 'district' to keep dimensionality low.
    print("- Encoding Categoricals")
    
    df['state_freq'] = df.groupby('state')['state'].transform('count') / len(df)
    df['district_freq'] = df.groupby('district')['district'].transform('count') / len(df)
    
    # 4. Extract Date Features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 5. Handling NaNs (Lags create Missing Values)
    # We can drop the first 30 days of data for each pincode, or fill with 0 / mean.
    # For time series forecasting, dropping is often safer to avoid data leakage or bad priors.
    print("- Handling Missing Values")
    initial_shape = df.shape
    df.dropna(subset=['enrolment_lag_30'], inplace=True) # Lose first 30 days
    final_shape = df.shape
    print(f"  Dropped {initial_shape[0] - final_shape[0]} rows due to lag creation.")

    return df

def main():
    df = load_data()
    if df is None:
        return

    df_featured = create_features(df)
    
    print(f"Saving feature-engineered data to {OUTPUT_FILE}...")
    df_featured.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
