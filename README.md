# ML Sales Forecasting Engine

> Predicts weekly sales per SKU across 200+ stores with **94% accuracy** using a stacked ensemble model.

## The Problem
A retail client was ordering stock based on gut feeling — regularly overstocking slow SKUs and running out of fast ones. I built this to fix that.

## How It Works
Stacked ensemble: **XGBoost + Random Forest + Ridge Regression**. Feature engineering includes lag features, rolling averages, seasonality flags, and promotion indicators. SHAP values explain which factors drive each store's forecast.

## Results
- 94% forecast accuracy on weekly sales
- Deployed as a scheduled job — fresh predictions every Sunday 2 AM
- Reduced overstock by ~28% in first quarter

## Stack
```
Python 3.11 | Scikit-Learn | XGBoost | SHAP | Optuna | MLflow | Pandas | NumPy
```

## Project Structure
```
ml-sales-forecasting/
├── data/           # Data loading & preprocessing
├── features/       # Feature engineering pipeline
├── models/         # Model training & evaluation
├── explainability/ # SHAP analysis & reports
├── deploy/         # Scheduling & deployment scripts
└── notebooks/      # EDA & research notebooks
```

## Contact
Built by **Shebin S Illikkal** — [Shebinsillikkl@gmail.com](mailto:Shebinsillikkl@gmail.com)
