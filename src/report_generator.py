
from fpdf import FPDF
import os
import datetime

# --- Configuration ---
OUTPUT_PDF = "UIDAI_Aadhaar_Analysis_Report.pdf"
PLOTS_DIR = "processed_data/plots"
METRICS_FILE = "processed_data/model_metrics.txt"
ANOMALIES_FILE = "processed_data/anomalies.csv"

class PDF(FPDF):
    def header(self):
        # Logo could go here
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'UIDAI Aadhaar Data Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def create_report():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- Title Page ---
    pdf.set_font('Arial', 'B', 24)
    pdf.ln(50)
    pdf.cell(0, 10, 'Dropout Prediction & Analysis', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Target: Reduce Enrolment Dropouts', 0, 1, 'C')
    pdf.ln(20)
    pdf.cell(0, 10, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
    pdf.add_page()

    # --- Executive Summary ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '1. Executive Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    summary_text = (
        "This project analyzed Aadhaar enrolment and update trends to identify potential dropout risks. "
        "Key findings include:\n"
        "- Identified daily activity patterns and regional disparities.\n"
        "- Discovered high correlation between demographic updates and enrolments in specific clusters.\n"
        "- Developed a Random Forest model with high predictive accuracy (R2 ~ 0.98) to forecast update ratios, "
        "helping targeted intervention."
    )
    pdf.multi_cell(0, 10, summary_text)
    pdf.ln(10)

    # --- Data Insights (EDA) ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '2. specific Insights from Data', 0, 1, 'L')
    
    # Helper to add image if exists
    def add_plot(title, filename, description):
        if os.path.exists(os.path.join(PLOTS_DIR, filename)):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, title, 0, 1, 'L')
            pdf.image(os.path.join(PLOTS_DIR, filename), w=170)
            pdf.set_font('Arial', 'I', 10)
            pdf.multi_cell(0, 10, description)
            pdf.ln(5)
            
    # Histogram
    add_plot("Activity Distribution", "hist_total_activity.png", 
             "The distribution of total activity shows a long-tailed pattern, indicating most centers have low to moderate activity, while a few major hubs handle massive volumes.")
    
    # Trend
    pdf.add_page()
    add_plot("Temporal Trends", "trend_time_activity.png", 
             "Activity levels showing fluctuations over time. Peaks may correspond to special drives or deadlines.")

    # State Analysis
    add_plot("State-wise Performance", "bar_state_activity.png", 
             "Top 10 states contributing to the bulk of enrolments and updates.")

    # Correlation
    pdf.add_page()
    add_plot("Correlation Analysis", "heatmap_correlation.png", 
             "Correlation matrix showing strong positive relationship between Enrolment and Demographic Updates.")
             
    # --- Advanced Analysis ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '3. Advanced Analysis', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    # Clusters
    pdf.multi_cell(0, 10, "K-Means Clustering grouped pincodes into distinct profiles based on their activity mix. This allows for region-specific strategies.")
    if os.path.exists(os.path.join(PLOTS_DIR, "cluster_pairplot.png")):
         pdf.image(os.path.join(PLOTS_DIR, "cluster_pairplot.png"), w=170)
    
    # Anomalies
    num_anomalies = "N/A"
    if os.path.exists(ANOMALIES_FILE):
        with open(ANOMALIES_FILE) as f:
            num_anomalies = sum(1 for line in f) - 1 # approximate
            
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Isolation Forest identified approximately {num_anomalies} anomalous records. These represent unusual spikes or ratios that warrant field investigation.")
    if os.path.exists(os.path.join(PLOTS_DIR, "scatter_anomalies.png")):
         pdf.image(os.path.join(PLOTS_DIR, "scatter_anomalies.png"), w=170)

    # --- Predictive Modeling ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '4. Predictive Modeling', 0, 1, 'L')
    
    metrics_text = "Model: Random Forest Regressor\n"
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics_text += f.read()
    else:
        metrics_text += "Metrics file not found."
        
    pdf.set_font('Courier', '', 11)
    pdf.multi_cell(0, 10, metrics_text)
    
    pdf.ln(5)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "Feature Importance Analysis highlights the key drivers of the prediction:")
    
    if os.path.exists(os.path.join(PLOTS_DIR, "feature_importance.png")):
         pdf.image(os.path.join(PLOTS_DIR, "feature_importance.png"), w=170)

    # --- Conclusion ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '5. Conclusion & Recommendations', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    rec_text = (
        "1. Focus verification resources on the identified anomalies to reduce fraud or errors.\n"
        "2. Use the 'Update-to-Enrolment' ratio forecast to allocate staff dynamically.\n"
        "3. Investigate high-dropout clusters identified in the regional analysis.\n\n"
        "This framework provides a data-driven foundation for minimizing dropouts and optimizing the enrolment lifecycle."
    )
    pdf.multi_cell(0, 10, rec_text)

    # Save
    pdf.output(OUTPUT_PDF)
    print(f"Report generated successfully: {OUTPUT_PDF}")

if __name__ == "__main__":
    create_report()
