# UIDAI Aadhaar Data Analysis & Prediction

This project analyzes Aadhaar enrolment and update trends to identify potential dropout risks. It includes a comprehensive data analysis pipeline, an interactive Streamlit dashboard for real-time risk prediction, and an automated PDF report generator.

## 🚀 Key Features
*   **Interactive Dashboard**: Visualize daily trends, regional hotspots, and predict dropout risks based on hypothetical scenarios.
*   **Predictive Modeling**: An Optimized Random Forest model (R2 ~0.94) forecasts the "Update-to-Enrolment Ratio".
*   **Automated Reporting**: Generates a professional PDF report with executive summaries, charts, and recommendations.
*   **Clustering & Anomaly Detection**: Identifies distinct center profiles and flags unusual activity spikes.

## 📂 Project Structure
```
UIDAI/
├── src/
│   ├── app.py                  # Main Streamlit Dashboard application
│   ├── model_training.py       # Initial model training script
│   ├── train_optimized_rf.py   # Optimized model training (Depth limited)
│   ├── report_generator.py     # script to generate the PDF report
│   ├── check_accuracy.py       # Script to verify model performance
│   ├── feature_engineering.py  # Feature extraction pipeline
│   ├── eda_*.py                # Exploratory Data Analysis scripts
│   └── processed_data/         # Contains models (.pkl), metrics, and plots
├── requirements.txt            # Python dependencies
└── UIDAI_Aadhaar_Analysis_Report.pdf  # Final generated report
```

## 🛠️ Installation & Setup

1.  **Environment Setup**:
    Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

### 1. Run the Dashboard
Launch the interactive web application to explore data and use the predictor:
```bash
streamlit run src/app.py
```
*Navigate to the "Predictor" page to test the "Update-to-Enrolment" risk model.*

### 2. Generate PDF Report
Create or update the comprehensive analysis report:
```bash
python3 src/report_generator.py
```
*The report will be saved as `UIDAI_Aadhaar_Analysis_Report.pdf` in the root directory.*

### 3. Check Accuracy
Validate the model's performance on the dataset:
```bash
python3 src/check_accuracy.py
```

## 📊 Model Insights
*   **Current Model**: Optimized Random Forest Regressor.
*   **Performance**: R2 Score &#8776; 0.94 (Robust, non-overfitting).
*   **Key Driver**: Demographic updates for older age groups (18+) are the strongest predictor of high cumulative update ratios.
*   **Operational Context**: A high update ratio (>1.0) is normal for permanent centers but indicates potential "dropout" (failure to enrol first time) if seen in new enrolment camps.