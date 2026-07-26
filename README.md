# Grade Change Intelligence in Paper Making Process

This project is a predictive intelligence simulator designed for Honeywell, aimed at detecting off-spec events and minimizing grade-change transition times. By utilizing a physics-grounded simulation, machine learning, and explainable AI (SHAP and LLM rationale), the system identifies early warning signals of process instability and surfaces actionable recommendations for operators.

## Project Structure

* `/simulator/`: Core simulation logic, physics-based transfer functions, and disturbance injection models.
* `/feature_engineering/`: Real-time signal processing and feature creation pipelines.
* `/correlation_discovery/`: Disturbance-windowed Granger Causality analysis to surface hidden variable couplings.
* `/modeling/`: LightGBM predictor training, ablation studies, and evaluation logic.
* `/explainability/`: SHAP explainer matrix components for interpreting predictions.
* `/recommendation/`: k-NN optimization engine and LLM rationale generation for actionable guidance.
* `/dashboard/`: Live Streamlit application bridging predictive insights and interactive visualization.
* `/docs/`: Project documentation and architecture diagrams.
* `/output/`: Model artifacts, generated datasets, and analytical figures (large files ignored in git).

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables:
   Copy `.env.example` to `.env` and fill in your actual API keys (OpenAI, Groq).
   ```bash
   cp .env.example .env
   ```

## How to Run

To regenerate the results and run the dashboard, execute the modules in this specific order:
1. **Simulator**: Generate baseline process data (`python simulator/simulator.py`)
2. **Feature Engineering**: Process the raw time series into predictive features (`python feature_engineering/feature_engineering.py`)
3. **Correlation Discovery**: Analyze causality links across disturbances (`python correlation_discovery/correlation_discovery.py`)
4. **Training**: Train the LightGBM models (`python modeling/train_lgbm.py`)
5. **Dashboard**: Launch the interactive Streamlit app (`python -m streamlit run dashboard/dashboard.py`)

## Key Results Summary
* **Ground-Truth Validation**: 5/5 known causal links successfully recovered by the Correlation Discovery Engine; 16 spurious correlations correctly rejected.
* **Early Warning Capability**: 3.5-minute average predictive lead time achieved on 6 out of 9 held-out validation events.
* **Process Stabilization**: The recommendation engine successfully stabilized test disturbances, yielding measurable reductions in grade-change disruption metrics.
