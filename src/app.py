
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
st.set_page_config(
    page_title="UIDAI Analytics",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
DATA_FILE = "processed_data/model_ready_data.csv"
MODEL_FILE = "processed_data/model_rf.pkl"
ANOMALIES_FILE = "processed_data/anomalies.csv"
CLUSTERS_FILE = "processed_data/pincode_clusters.csv"
PLOTS_DIR = "processed_data/plots"

# --- Custom Apple-Inspired CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* General Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1d1d1f;
        }
        
        /* Background - Subtle Gradient */
        .stApp {
            background: #f5f5f7; /* Apple Light Gray */
        }

        /* CARD STYLE - Glassmorphism */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        }

        /* Sidebar - Glassy Dark Mockup */
        section[data-testid="stSidebar"] {
            background-color: rgba(28, 28, 30, 0.95); /* Apple Dark Gray */
            color: white;
        }
        
        /* Buttons - Pill Shape & Smooth */
        div.stButton > button {
            background-color: #007aff; /* Apple Blue */
            color: white;
            border-radius: 20px;
            border: none;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        div.stButton > button:hover {
            background-color: #005bb5;
            box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
            transform: scale(1.02);
        }

        /* Headings */
        h1, h2, h3 {
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        h1 {
            background: -webkit-linear-gradient(45deg, #1d1d1f, #48484a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Dataframes - Clean Look */
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(0,0,0,0.05);
        }
        
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Functions ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    return df

@st.cache_data
def load_anomalies():
    if not os.path.exists(ANOMALIES_FILE):
        return None
    return pd.read_csv(ANOMALIES_FILE)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

# --- Navigation ---
st.sidebar.title(" UIDAI Analytics")
page = st.sidebar.radio("Navigation", ["Dashboard", "Data Explorer", "Predictor"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.info("Aadhaar Data Analysis System v1.0")

# --- Page: Dashboard ---
if page == "Dashboard":
    st.title("Dashboard")
    
    df = load_data()
    if df is not None:
        # Top Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        
        total_enrolment = df['enrolment_total'].sum()
        total_updates = df['demo_update_total'].sum() + df['bio_update_total'].sum()
        avg_daily_activity = df['total_activity'].mean()
        
        c1.metric("Total Enrolments", f"{total_enrolment:,.0f}")
        c2.metric("Total Updates", f"{total_updates:,.0f}")
        c3.metric("Avg Daily Activity", f"{avg_daily_activity:.1f}")
        
        # Anomalies Metric
        anom_df = load_anomalies()
        anom_count = len(anom_df) if anom_df is not None else 0
        c4.metric("Anomalies Detected", f"{anom_count:,.0f}", delta="-High Risk", delta_color="inverse")
        
        st.markdown("### Activity Trends")
        # Time Series plot
        daily_trend = df.groupby('date')[['enrolment_total', 'demo_update_total', 'bio_update_total']].sum()
        st.line_chart(daily_trend)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Regional Distribution")
            # State Aggregate
            state_agg = df.groupby('state')['enrolment_total'].sum().sort_values(ascending=False).head(10)
            st.bar_chart(state_agg)
            
        with col_right:
            st.markdown("### Update Ratios")
            # Scatter of Enrolment vs Updates
            # Sample to avoid lag
            chart_data = df.sample(2000)[['enrolment_total', 'demo_update_total']]
            st.scatter_chart(chart_data)

    else:
        st.error(f"Data file not found at {DATA_FILE}. Please run data pipeline first.")

# --- Page: Data Explorer ---
elif page == "Data Explorer":
    st.title("Data Explorer")
    
    tab1, tab2 = st.tabs(["Anomalies", "Raw Data"])
    
    with tab1:
        st.subheader("Detected Anomalies")
        st.markdown("These records showed unusual activity patterns (e.g., extremely high updates vs enrolments).")
        anom_df = load_anomalies()
        if anom_df is not None:
            # Filters
            states = list(anom_df['state'].unique())
            selected_state = st.selectbox("Filter by State", ["All"] + states)
            
            if selected_state != "All":
                anom_df = anom_df[anom_df['state'] == selected_state]
            
            st.dataframe(anom_df, use_container_width=True)
            st.caption(f"Showing {len(anom_df)} records.")
        else:
            st.info("No anomalies file found.")
            
    with tab2:
        st.subheader("Master Dataset")
        df = load_data()
        if df is not None:
            st.dataframe(df.head(1000), use_container_width=True)
            st.caption("Showing first 1000 rows.")

# --- Page: Predictor ---
elif page == "Predictor":
    st.title("Dropout Risk Predictor")
    st.markdown("Estimate the `Update-to-Enrolment Ratio` for a hypothetical scenario. A high ratio (>1.0) implies potential initial enrolment failures.")
    
    model = load_model()
    
    if model:
        # Input Form
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                age_0_5 = st.number_input("Enrolments (Age 0-5)", min_value=0, value=10)
                age_5_17 = st.number_input("Enrolments (Age 5-17)", min_value=0, value=20)
                age_18_greater = st.number_input("Enrolments (Age 18+)", min_value=0, value=50)
            
            with c2:
                # We need lag features too, but for a simple demo, we can ask for 'Historic Average'
                hist_avg = st.number_input("Historic 7-Day Avg Enrolment", min_value=0, value=30)
                state_freq = st.slider("State Activity Index (0-1)", 0.0, 1.0, 0.5)

            submit = st.form_submit_button("Analyze Risk")
            
        if submit:
            # Construct input vector matching model features (approximate for demo)
            # Features: ['age_0_5', 'age_5_17', 'age_18_greater', ... 'enrolment_rolling_mean_7', ...]
            # We'll fill missing technical features with reasonable defaults/zeros for the demo
            
            input_data = pd.DataFrame({
                'age_0_5': [age_0_5],
                'age_5_17': [age_5_17],
                'age_18_greater': [age_18_greater],
                'demo_age_5_17': [0], 'demo_age_17_': [0], # Assume 0 for interaction terms
                'bio_age_5_17': [0], 'bio_age_17_': [0],
                'enrolment_lag_7': [hist_avg], # Approximation
                'enrolment_lag_30': [hist_avg],
                'enrolment_rolling_mean_7': [hist_avg],
                'enrolment_rolling_mean_30': [hist_avg],
                'total_activity_rolling_mean_7': [hist_avg + 10],
                'state_freq': [state_freq],
                'district_freq': [0.01],
                'month': [6], # Default mid-year
                'day_of_week': [2],
                'is_weekend': [0]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.markdown("### Risk Analysis Result")
            
            col_metric, col_gauge = st.columns([1, 2])
            
            with col_metric:
                st.metric("Predicted Ratio", f"{prediction:.2f}")
            
            with col_gauge:
                if prediction < 0.5:
                    st.success("Low Risk: Process seems healthy.")
                elif prediction < 1.0:
                    st.warning("Moderate Risk: Monitor for quality issues.")
                else:
                    st.error("High Risk: Potential systemic dropouts detected.")
                    
    else:
        st.warning("Model file not found. Please train the model first.")

