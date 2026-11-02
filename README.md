# Manufactured Home Price Predictor

An interactive machine-learning application for estimating manufactured home prices from property characteristics. Built with Streamlit and a saved prediction model.

**[Try the live application on Hugging Face Spaces](https://huggingface.co/spaces/sulavregmi44/Manufactured_Home_Price_Predictor)**

## Overview

Enter home details such as square footage, sections, bedrooms, region, location type, shipment date, foundation, title, financing status, and new/used condition to receive a predicted price. The dashboard also displays an illustrative range based on reported mean absolute error, regional comparisons, and a summary of the inputs.

## Model and results

The application describes the model as **LightGBM with anomaly removal (2.5σ)** and reports **R² = 0.806** on a filtered test set and **MAE = $13,380**. The interface states that training data covers **2014–2024** and over **60,000 homes**. These figures are reported by the supplied application; training code and evaluation data are not included in this repository.

**Caution:** Predictions for 2025–2027 extrapolate beyond the stated training period and should be treated as speculative. The displayed ±MAE interval is not a calibrated prediction interval. The regional benchmarks in the app are hardcoded reference values.

## Run locally

Requires Python 3.10+ (recommended).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The application expects the trained model at `src/model.pkl` relative to the project root. Run the Streamlit command from the repository root.

## Repository layout

```text
manufactured-home-price-predictor/
├── streamlit_app.py     # Streamlit user interface and feature construction
├── requirements.txt     # Dependencies
├── src/
│   └── model.pkl        # Serialized trained model
├── .gitignore
└── README.md
```

## Technologies

Python, Streamlit, Pandas, NumPy, Matplotlib, Joblib, LightGBM, scikit-learn.

## Notes

- The model file is a Python pickle. Only load it from a trusted source.
- Dependency versions are unpinned because the original environment's dependency lockfile was not supplied. If an environment mismatch occurs, use the original Hugging Face Space's dependency versions.
- This repository includes the inference application and trained model, not the original training dataset or training pipeline.

## Author

[Sulav Regmi](https://github.com/sulavregmi) · [LinkedIn](https://www.linkedin.com/in/sulavregmi44/)
