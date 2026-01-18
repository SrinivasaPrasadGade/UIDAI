
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_FILE = "processed_data/master_aadhar_data.csv"
PLOTS_DIR = "processed_data/plots"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None
    print(f"Loading {DATA_FILE}...")
    return pd.read_csv(DATA_FILE)

def plot_histograms(df, columns):
    print("Generating histograms...")
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        filename = os.path.join(PLOTS_DIR, f"hist_{col}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

def plot_boxplots(df, columns):
    print("Generating boxplots...")
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        filename = os.path.join(PLOTS_DIR, f"box_{col}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

def main():
    ensure_dir(PLOTS_DIR)
    df = load_data()
    if df is None:
        return

    # Basic Statistics
    print("\n--- Basic Statistics ---")
    desc = df.describe()
    print(desc)
    desc.to_csv(os.path.join(PLOTS_DIR, "statistics_summary.csv"))
    print(f"Saved statistics to {os.path.join(PLOTS_DIR, 'statistics_summary.csv')}")

    # Columns to visualize
    # Focusing on total activity columns first
    cols_to_plot = ['enrolment_total', 'demo_update_total', 'bio_update_total', 'total_activity']
    
    # Filter only existing columns
    existing_cols = [c for c in cols_to_plot if c in df.columns]
    
    if existing_cols:
        plot_histograms(df, existing_cols)
        plot_boxplots(df, existing_cols)
    else:
        print("No columns found to plot.")

if __name__ == "__main__":
    main()
