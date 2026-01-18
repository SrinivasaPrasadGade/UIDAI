
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "master_aadhar_data.csv")
PLOTS_DIR = os.path.join(PROCESSED_DATA_DIR, "plots")
ANOMALIES_FILE = os.path.join(PROCESSED_DATA_DIR, "anomalies.csv")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None
    print(f"Loading {DATA_FILE}...")
    return pd.read_csv(DATA_FILE)

def perform_clustering(df):
    print("Performing K-Means Clustering...")
    # Features for clustering (aggregated by pincode to make sense of regions)
    # We need to aggregate first because the master data is per-date-region
    
    # Aggregating by Pincode
    agg_cols = {'enrolment_total': 'sum', 'demo_update_total': 'sum', 'bio_update_total': 'sum'}
    pincode_df = df.groupby('pincode').agg(agg_cols).reset_index()
    
    # Normalizing
    features = ['enrolment_total', 'demo_update_total', 'bio_update_total']
    X = pincode_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means (K=4 arbitrarily chosen for High/Med/Low/Mixed profiles)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    pincode_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualize Clusters (Pairplot)
    plt.figure(figsize=(10, 8))
    sns.pairplot(pincode_df, vars=features, hue='cluster', palette='viridis')
    filename = os.path.join(PLOTS_DIR, "cluster_pairplot.png")
    plt.savefig(filename)
    plt.close() # pairplot creates its own figure, but good practice
    print(f"Saved {filename}")
    
    # Save clustered data
    pincode_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "pincode_clusters.csv"), index=False)
    print(f"Saved cluster assignments to processed_data/pincode_clusters.csv")

def perform_anomaly_detection(df):
    print("Performing Anomaly Detection (Isolation Forest)...")
    
    # We can look for anomalies in daily records OR aggregated pincode behavior.
    # Let's look for anomalous DAILY records first (e.g., massive spikes).
    
    features = ['enrolment_total', 'demo_update_total', 'bio_update_total']
    X = df[features]
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42) # Top 1% anomalies
    df['anomaly'] = iso.fit_predict(X)
    
    # Filter anomalies (-1 indicates anomaly)
    anomalies = df[df['anomaly'] == -1]
    
    print(f"Found {len(anomalies)} anomalies out of {len(df)} records.")
    
    # Save anomalies
    anomalies.to_csv(ANOMALIES_FILE, index=False)
    print(f"Saved anomalies to {ANOMALIES_FILE}")
    
    # Visualize Anomalies (Scatter plot of Enrolment vs Updates)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='enrolment_total', y='demo_update_total', hue='anomaly', style='anomaly', palette={1: 'blue', -1: 'red'}, alpha=0.6)
    plt.title('Anomaly Detection: Enrolment vs Demographic Updates')
    filename = os.path.join(PLOTS_DIR, "scatter_anomalies.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def main():
    ensure_dir(PLOTS_DIR)
    df = load_data()
    if df is None:
        return
        
    # 1. Clustering (Regional Profiles)
    perform_clustering(df)
    
    # 2. Anomaly Detection (Unusual Records)
    perform_anomaly_detection(df)

if __name__ == "__main__":
    main()
