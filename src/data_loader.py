
import pandas as pd
import glob
import os

# --- Configuration ---
DATA_DIR = "."
PROCESSED_DIR = "./processed_data"
ENROLMENT_DIR = os.path.join(DATA_DIR, "api_data_aadhar_enrolment")
DEMOGRAPHIC_DIR = os.path.join(DATA_DIR, "api_data_aadhar_demographic")
BIOMETRIC_DIR = os.path.join(DATA_DIR, "api_data_aadhar_biometric")

def load_and_merge(directory, category_name):
    """Loads all CSVs in a directory and merges them into a single DataFrame."""
    print(f"\n--- Loading {category_name} Data ---")
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        print(f"No files found in {directory}")
        return None
    
    df_list = []
    for f in files:
        print(f"Reading {os.path.basename(f)}...")
        try:
            # Optimize: Read only necessary columns if files are huge, but here we need all.
            # Convert 'date' during read to save memory/time later
            df_temp = pd.read_csv(f) 
            df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        print(f"Successfully merged {len(files)} files. Total shape: {full_df.shape}")
        return full_df
    return None

def initial_profile(df, name):
    """Prints basic info, null counts, and sample checks."""
    if df is None:
        return
    
    print(f"\n=== Profile: {name} ===")
    print(df.info())
    print("\n--- Null Values ---")
    print(df.isnull().sum())
    print("\n--- Sample Data ---")
    print(df.head())
    
    # Check duplicate rows
    dupes = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dupes}")

def main():
    # 1. Load Enrolment
    df_enro = load_and_merge(ENROLMENT_DIR, "Enrolment")
    initial_profile(df_enro, "Enrolment Data")
    
    # 2. Load Demographic
    df_demo = load_and_merge(DEMOGRAPHIC_DIR, "Demographic Updates")
    initial_profile(df_demo, "Demographic Data")
    
    # 3. Load Biometric
    df_bio = load_and_merge(BIOMETRIC_DIR, "Biometric Updates")
    initial_profile(df_bio, "Biometric Data")

    # If all exist, try a sample merge on Pincode/Date to see overlap
    if df_enro is not None and df_demo is not None:
        print("\n--- Overlap Check (Enrolment vs Demographic) ---")
        # Rename columns to avoid collision before merge if needed, but here they seem distinct enough
        # Enro: age_0_5... Demo: demo_age_5_17...
        
        # Merge on keys
        merge_keys = ['date', 'state', 'district', 'pincode']
        # Outer merge to see full scope
        merged_sample = pd.merge(df_enro, df_demo, on=merge_keys, how='outer', indicator=True)
        print("Merge Indicator Counts:")
        print(merged_sample['_merge'].value_counts())
        
        # Save a sample for the User to inspect?
        sample_path = os.path.join(PROCESSED_DIR, "data_sample_merged.csv")
        merged_sample.head(100).to_csv(sample_path, index=False)
        print(f"\nSaved sample merged data to {sample_path}")

if __name__ == "__main__":
    main()
