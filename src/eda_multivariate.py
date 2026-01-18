
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
    # Read date column as datetime
    return pd.read_csv(DATA_FILE, parse_dates=['date'])

def plot_correlation_heatmap(df):
    print("Generating correlation heatmap...")
    # Select numerical columns for correlation
    cols = ['enrolment_total', 'demo_update_total', 'bio_update_total', 
            'age_0_5', 'age_5_17', 'age_18_greater']
    
    # Filter only existing columns
    existing_cols = [c for c in cols if c in df.columns]
    
    if len(existing_cols) > 1:
        corr = df[existing_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Activity Types')
        plt.tight_layout()
        filename = os.path.join(PLOTS_DIR, "heatmap_correlation.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

def plot_time_trends(df):
    print("Generating time trend analysis...")
    # Resample by month or week if data is dense, but daily aggregate for now
    # Group by date
    daily_df = df.groupby('date')[['enrolment_total', 'demo_update_total', 'bio_update_total']].sum().reset_index()
    
    plt.figure(figsize=(14, 7))
    plt.plot(daily_df['date'], daily_df['enrolment_total'], label='Enrolment')
    plt.plot(daily_df['date'], daily_df['demo_update_total'], label='Demographic Updates')
    plt.plot(daily_df['date'], daily_df['bio_update_total'], label='Biometric Updates')
    
    plt.title('Daily Activity Trends')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, "trend_time_activity.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_state_wise_analysis(df):
    print("Generating state-wise analysis...")
    # Group by state
    if 'state' not in df.columns:
        return

    state_df = df.groupby('state')['total_activity'].sum().sort_values(ascending=False).head(10).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=state_df, x='total_activity', y='state', palette='viridis', hue='state', legend=False)
    plt.title('Top 10 States by Total Activity')
    plt.xlabel('Total Activity')
    plt.ylabel('State')
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, "bar_state_activity.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_scatter_enrol_vs_demo(df):
    print("Generating scatter plot (Enrolment vs Demographic)...")
    # Sample data for scatter plot to avoid overplotting if too large
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sample_df, x='enrolment_total', y='demo_update_total', alpha=0.5)
    plt.title('Enrolment vs Demographic Updates (Sampled)')
    plt.xlabel('Enrolment Total')
    plt.ylabel('Demographic Update Total')
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, "scatter_enrol_vs_demo.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def main():
    ensure_dir(PLOTS_DIR)
    df = load_data()
    if df is None:
        return

    plot_correlation_heatmap(df)
    plot_time_trends(df)
    plot_state_wise_analysis(df)
    plot_scatter_enrol_vs_demo(df)

if __name__ == "__main__":
    main()
