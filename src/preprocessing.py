
import pandas as pd
import glob
import os

# --- Configuration ---
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_data")
ENROLMENT_DIR = os.path.join(DATA_DIR, "api_data_aadhar_enrolment")
DEMOGRAPHIC_DIR = os.path.join(DATA_DIR, "api_data_aadhar_demographic")
BIOMETRIC_DIR = os.path.join(DATA_DIR, "api_data_aadhar_biometric")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "master_aadhar_data.csv")
LOG_FILE = os.path.join(PROCESSED_DIR, "processing_log.txt")

def log(message):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

def load_and_aggregate(directory, category_name, value_cols):
    """
    Loads CSVs, converts date, and aggregates by Pincode/Date to remove duplicates.
    """
    log(f"\n--- Processing {category_name} ---")
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        log(f"No files found in {directory}")
        return None
    
    df_list = []
    for f in files:
        try:
            df_temp = pd.read_csv(f)
            df_list.append(df_temp)
        except Exception as e:
            log(f"Error reading {f}: {e}")
            
    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)
    raw_shape = df.shape
    log(f"Raw shape: {raw_shape}")
    
    # Standardize Date
    # Using format='%d-%m-%Y' based on "31-12-2025" seen in profile
    try:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    except Exception as e:
        log(f"Date conversion error: {e}")

    # Drop Invalid Dates
    df = df.dropna(subset=['date'])
    
    # Aggregate to resolve duplicates (Group by Key Columns and Sum Counts)
    group_cols = ['date', 'state', 'district', 'pincode']
    df_agg = df.groupby(group_cols)[value_cols].sum().reset_index()
    
    agg_shape = df_agg.shape
    log(f"Aggregated shape: {agg_shape} (Reduced by {raw_shape[0] - agg_shape[0]} rows)")
    
    return df_agg

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # 1. Enrolment
    # Cols: age_0_5, age_5_17, age_18_greater
    df_enro = load_and_aggregate(
        ENROLMENT_DIR, 
        "Enrolment", 
        ['age_0_5', 'age_5_17', 'age_18_greater']
    )
    
    # 2. Demographic
    # Cols: demo_age_5_17, demo_age_17_
    df_demo = load_and_aggregate(
        DEMOGRAPHIC_DIR, 
        "Demographic", 
        ['demo_age_5_17', 'demo_age_17_']
    )
    
    # 3. Biometric
    # Cols: bio_age_5_17, bio_age_17_
    df_bio = load_and_aggregate(
        BIOMETRIC_DIR, 
        "Biometric", 
        ['bio_age_5_17', 'bio_age_17_']
    )
    
    # 4. Merge All
    log("\n--- Merging Datasets ---")
    
    # Merge Enrolment + Demographic
    # Outer merge because some pincodes might only have updates, others only enrolments
    merge_cols = ['date', 'state', 'district', 'pincode']
    
    master_df = df_enro
    
    if df_demo is not None:
        master_df = pd.merge(master_df, df_demo, on=merge_cols, how='outer')
        
    if df_bio is not None:
        master_df = pd.merge(master_df, df_bio, on=merge_cols, how='outer')
        
    # 5. Final Cleanup
    # Fill NaNs with 0 (because outer merge created NaNs where data was missing for a dataset)
    count_cols = [
        'age_0_5', 'age_5_17', 'age_18_greater',
        'demo_age_5_17', 'demo_age_17_',
        'bio_age_5_17', 'bio_age_17_'
    ]
    master_df[count_cols] = master_df[count_cols].fillna(0)
    
    # Add Total Columns for convenience
    master_df['enrolment_total'] = master_df['age_0_5'] + master_df['age_5_17'] + master_df['age_18_greater']
    master_df['demo_update_total'] = master_df['demo_age_5_17'] + master_df['demo_age_17_']
    master_df['bio_update_total'] = master_df['bio_age_5_17'] + master_df['bio_age_17_']
    master_df['total_activity'] = master_df['enrolment_total'] + master_df['demo_update_total'] + master_df['bio_update_total']
    
    log(f"\nFinal Master Data Shape: {master_df.shape}")
    
    # Save
    master_df.to_csv(OUTPUT_FILE, index=False)
    log(f"Saved Master Dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
