
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import viz_components as viz


# --- Configuration ---
st.set_page_config(
    page_title="UIDAI Analytics",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "model_ready_data.csv")
MODEL_FILE = os.path.join(PROCESSED_DATA_DIR, "model_rf.pkl")
ANOMALIES_FILE = os.path.join(PROCESSED_DATA_DIR, "anomalies.csv")
CLUSTERS_FILE = os.path.join(PROCESSED_DATA_DIR, "pincode_clusters.csv")
PLOTS_DIR = os.path.join(PROCESSED_DATA_DIR, "plots")

# --- Custom Apple-Inspired CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* --- GLOBAL & FONTS --- */
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'SF Pro Display', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #1d1d1f;
        }

        /* --- APP BACKGROUND --- */
        .stApp {
            background-color: #F5F5F7; /* Apple System Light Gray */
        }
        
        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #ffffff; /* Clean White Sidebar */
            border-right: 1px solid rgba(0,0,0,0.05); /* Subtle separator */
        }
        section[data-testid="stSidebar"] h1 {
            color: #1d1d1f !important;
            font-weight: 600;
        }
        
        /* --- METRIC CARDS (WIDGETS) --- */
        div[data-testid="metric-container"] {
            background: #ffffff;
            border-radius: 20px;
            padding: 24px 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03), 0 1px 2px rgba(0,0,0,0.02); /* Soft layered shadow */
            border: 1px solid rgba(0,0,0,0.02);
            transition: all 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.06);
        }
        
        /* Metric Labels */
        div[data-testid="metric-container"] > label {
            font-size: 14px;
            font-weight: 500;
            color: #86868b; /* Apple Secondary Text */
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* Metric Values */
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 600;
            color: #1d1d1f;
        }

        /* --- CHARTS & CONTAINERS --- */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        
        /* Chart Backgrounds - Make them blend */
        div[class*="stChart"] {
            background: #ffffff;
            border-radius: 20px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(0,0,0,0.02);
        }

        /* --- BUTTONS --- */
        div.stButton > button {
            background-color: #0071e3; /* Classic Apple Blue */
            color: white;
            border-radius: 980px; /* Pill shape */
            border: none;
            padding: 12px 28px;
            font-size: 16px;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0, 113, 227, 0.2);
            transition: all 0.25s cubic-bezier(0.25, 0.1, 0.25, 1);
        }
        div.stButton > button:hover {
            background-color: #0077ED;
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0, 113, 227, 0.4);
        }
        div.stButton > button:active {
            transform: scale(0.98);
        }
        
        /* --- HEADERS --- */
        h1, h2, h3 {
            font-family: 'SF Pro Display', 'Inter', sans-serif;
            letter-spacing: -0.02em;
        }
        h1 {
            font-weight: 700;
        }
        h2 {
            font-weight: 600;
            margin-top: 1.5rem;
        }
        h3 {
            font-weight: 500;
            color: #1d1d1f;
        }

        /* --- DATAFRAME --- */
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            overflow: hidden;
        }

        /* --- NOTIFICATIONS --- */
        .stAlert {
            border-radius: 16px;
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
st.sidebar.title("UIDAI Analytics")
page = st.sidebar.radio("Navigation", ["Dashboard", "Data Explorer", "Predictor"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("Aadhaar Data Analysis System v1.0")

# --- Page: Dashboard ---
if page == "Dashboard":
    st.title("Dashboard")
    
    df = load_data()
    if df is not None:
        # --- 1. Top Metrics with Sparklines ---
        st.markdown("### Key Performance Indicators")
        c1, c2, c3, c4 = st.columns(4)
        
        total_enrolment = df['enrolment_total'].sum()
        total_updates = df['demo_update_total'].sum() + df['bio_update_total'].sum()
        avg_daily_activity = df['total_activity'].mean()
        
        # Prepare sparkline data (last 30 days for trend)
        daily_trend = df.groupby('date')['total_activity'].sum()
        sparkline_data = daily_trend.tail(30).values
        
        with c1:
            viz.render_kpi_card("Total Enrolments", total_enrolment, None, sparkline_data, color="#3b82f6")
        
        with c2:
            viz.render_kpi_card("Total Updates", total_updates, None, sparkline_data, color="#8b5cf6")
            
        with c3:
           viz.render_kpi_card("Avg Daily Activity", avg_daily_activity, None, sparkline_data, color="#10b981")
        
        # Anomalies Metric
        anom_df = load_anomalies()
        anom_count = len(anom_df) if anom_df is not None else 0
        with c4:
             viz.render_kpi_card("Anomalies Detected", anom_count, None, None, color="#ef4444")
        
        st.markdown("---")
        
        # --- 2. Activity Trends ---
        st.markdown("### Activity Trends")
        viz.render_activity_trends(df)
        
        st.markdown("---")
        
        # --- 3. Geographic & Ratio Analysis ---
        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            st.markdown("### Regional Distribution")
            viz.render_regional_map(df)
            
        with col_right:
            st.markdown("### Update Ratios")
            viz.render_update_ratios(df)

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
    st.markdown("Estimate the `Update-to-Enrolment Ratio` for a hypothetical scenario.")
    
    model = load_model()
    
    if model:
        # --- 1. Dynamic Feature Detection ---
        # We try to get the feature names the model was trained on
        model_features = getattr(model, "feature_names_in_", None)
        
        with st.form("prediction_form"):
            st.subheader("Scenario Inputs")
            c1, c2 = st.columns(2)
            with c1:
                # Inputs - Adjust names here if you know the exact CSV column headers!
                p1 = st.number_input("Enrolments (Age 0-5)", min_value=0, value=10)
                p2 = st.number_input("Enrolments (Age 5-17)", min_value=0, value=20)
                p3 = st.number_input("Enrolments (Age 18+)", min_value=0, value=50)
            
            with c2:
                hist_avg = st.number_input("Historic 7-Day Avg Enrolment", min_value=0, value=30)
                state_freq = st.slider("State Activity Index (0-1)", 0.0, 1.0, 0.5)

            # Debug Toggle
            st.markdown("---")
            scenario_type = st.radio(
                "Center Scenario Type",
                ["New Enrolment Camp (Expect Low Updates)", "Permanent Center (Expect High Updates)"],
                horizontal=True
            )

            show_debug = st.checkbox("Show Debug/Feature Info", value=False)
            submit = st.form_submit_button("Analyze Risk")
            
        if submit:
            st.info("Processing prediction request...")
            
            # Logic: If it's a "Camp", updates are rare (5% of enrolments). 
            # If it's a "Permanent Center", updates are common (150% of enrolments).
            if "Camp" in scenario_type:
                update_multiplier = 0.05 
            else:
                update_multiplier = 1.5

            # Calculate implied updates based on the scenario
            # (The model needs these counts to understand the context)
            implied_updates_5_17 = p2 * update_multiplier
            implied_updates_18_plus = p3 * update_multiplier
            
            # Total implied activity = Enrolments + Updates
            total_current_activity = (p1 + p2 + p3) + implied_updates_5_17 + implied_updates_18_plus

            # --- 2. Construct Raw Input Dictionary ---
            # NOTE: These keys must match what 'feature_engineering.py' outputted.
            # If your raw data had 'enrolment_0_5', change 'age_0_5' to that below.
            raw_input = {
                'age_0_5': [p1],
                'age_5_17': [p2],
                'age_18_greater': [p3],
                'enrolment_total': [p1 + p2 + p3],
                
                # We now use the multiplier to send "Low Update" signals to the model
                'demo_age_5_17': [implied_updates_5_17 * 0.6], # Split updates into demo/bio 
                'demo_age_17_': [implied_updates_18_plus * 0.6], 
                'bio_age_5_17': [implied_updates_5_17 * 0.4], 
                'bio_age_17_': [implied_updates_18_plus * 0.4],
                
                # Lags - We assume history matches current trend for the rolling mean
                'enrolment_lag_7': [hist_avg],
                'enrolment_lag_30': [hist_avg],
                'enrolment_rolling_mean_7': [hist_avg],
                'enrolment_rolling_mean_30': [hist_avg],
                
                # Crucial: Total activity must reflect the scenario
                # If we tell the model "Total Activity is high" but "Enrolment is low", it forces a High Ratio.
                # Here we balance it.
                'total_activity_rolling_mean_7': [hist_avg * (1 + update_multiplier)],
                
                # Context Features
                'state_freq': [state_freq],
                'district_freq': [0.01], # Default low freq for unknown district
                'month': [6],
                'day_of_week': [2],
                'is_weekend': [0]
            }
            
            input_df = pd.DataFrame(raw_input)

            # --- 3. Align with Model Features ---
            final_input = input_df.copy()
            
            if model_features is not None:
                # Create a dataframe with all zeros for expected columns
                final_input = pd.DataFrame(0, index=[0], columns=model_features)
                
                # Fill known values where names match
                aligned_cols = []
                missing_cols = []
                
                for col in model_features:
                    if col in input_df.columns:
                        final_input[col] = input_df[col]
                        aligned_cols.append(col)
                    else:
                        missing_cols.append(col)
                
                # --- Debugging Output ---
                if show_debug:
                    st.warning("⚠️ Feature Alignment Debugger")
                    with st.expander("Click to see feature details", expanded=True):
                        st.write(f"**Model expects {len(model_features)} features.**")
                        st.write(f"**Successfully Matched ({len(aligned_cols)}):**", aligned_cols)
                        st.error(f"**Missing Inputs (Filled with 0):** {missing_cols}")
                        st.caption("If your key inputs are in 'Missing Inputs', rename the keys in 'raw_input' dictionary to match.")

            # --- 4. Predict ---
            try:
                prediction = model.predict(final_input)[0]
                
                st.markdown("### Risk Analysis Result")
                col_metric, col_error, col_gauge = st.columns([1, 1, 1.5])
                
                with col_metric:
                    st.metric("Predicted Ratio", f"{prediction:.2f}")

                with col_error:
                    error_pct = prediction * 100
                    st.metric("Update vs Enrolment Ratio", f"{error_pct:.1f}%")
                
                with col_gauge:
                    if prediction < 0.5:
                        st.success("Low Risk: Process seems healthy.")
                    elif prediction < 1.0:
                        st.warning("Moderate Risk: Monitor for quality issues.")
                    else:
                        st.error("High Risk: Potential systemic dropouts detected.")
                        
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                if show_debug:
                    st.exception(e)
                    
    else:
        st.warning("Model file not found. Please train the model first.")

