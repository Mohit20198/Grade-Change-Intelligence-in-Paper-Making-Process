# Grade Change Intelligence in Paper Making Process 🏭📄

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit) ![LightGBM](https://img.shields.io/badge/LightGBM-Predictive_Model-green) ![Hackathon](https://img.shields.io/badge/Status-Hackathon_Project-orange)

An industrial intelligence pipeline and simulation dashboard designed to provide **early warnings** and **actionable optimizations** for off-spec basis weight risks during paper machine grade changes.

## 🌟 Overview
Grade changes in paper manufacturing are highly volatile periods where the risk of producing off-spec paper (broke) increases significantly. This project introduces a predictive intelligence pipeline that uses a 5-minute (300s) lookahead horizon to detect instability *before* it breaches target thresholds, offering operators constrained, safe recommendations to stabilize the process.

## 🚀 Key Features

* **Physics-Grounded Simulation**: Generates high-fidelity synthetic data modeling first-order-lag and deadtime transfer functions for Basis Weight, Moisture, and Ash content.
* **Causal Correlation Discovery**: Utilizes Vector Autoregression (VAR) to extract and validate Granger-causal relationships (e.g., steam pressure → moisture lag).
* **Early Warning Predictor (LightGBM)**: A highly tuned binary classifier providing a 5-minute predictive horizon for off-spec risk.
* **SHAP Explainability Layer**: Breaks down exactly *why* a risk alert is triggering, showing real-time feature contributions (log-odds).
* **Recommendation Engine & What-If**: Uses k-NN historical searches within a constrained action space to recommend safe operator interventions, alongside a simulated "What-If" trajectory.
* **Closed-Loop Feedback**: Logs operator Accept/Reject decisions for future model retraining.

## 📊 Dashboard UI
The project includes a rich **Streamlit Dashboard** featuring:
- **Simulation Control**: Playback historical disturbance events.
- **KPI Metrics**: Real-time Classifier AUC, F1 Score, Avg Lead Time (e.g. 3.5 min), and Stabilization Reduction.
- **Basis Weight Trend**: Live visualization of the process variable approaching the off-spec threshold.
- **Correlation Graph**: Interactive network graph of validated causal links.
- **SHAP Top Contributors & Risk Gauge**: Clear breakdown of current risk and its driving factors.

*(Note: See the `/assets/` or `/output/` folder for screenshots and architecture diagrams).*

## 🛠️ Tech Stack
- **Data & ML**: Python, pandas, numpy, scikit-learn, LightGBM, statsmodels
- **Explainability & Logic**: SHAP, NetworkX (Graphing)
- **Frontend & App**: Streamlit, Plotly
- **Storage**: SQLite (Feedback logs), JSON (API schemas)

## 💻 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r simulator/requirements.txt
   ```
2. **Launch the Dashboard:**
   ```bash
   cd simulator
   python -m streamlit run dashboard.py
   ```
