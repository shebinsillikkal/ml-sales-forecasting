"""
Sales Forecasting — Ensemble Model Training
Author: Shebin S Illikkal
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import shap
import mlflow
import joblib


def build_stacked_model():
    estimators = [
        ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05,
                              max_depth=6, subsample=0.8, random_state=42)),
        ('rf',  RandomForestRegressor(n_estimators=300, max_depth=8,
                                       min_samples_leaf=5, random_state=42))
    ]
    return StackingRegressor(estimators=estimators,
                             final_estimator=Ridge(alpha=1.0),
                             cv=TimeSeriesSplit(n_splits=5))


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mape  = mean_absolute_percentage_error(y_test, preds)
    acc   = (1 - mape) * 100
    print(f"Accuracy: {acc:.1f}%  |  MAPE: {mape:.4f}")
    return acc, mape


def train(data_path: str, model_out: str = "model.pkl"):
    df = pd.read_parquet(data_path)

    feature_cols = [c for c in df.columns if c not in ['date', 'sales', 'store_id', 'sku_id']]
    X = df[feature_cols]
    y = df['sales']

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    with mlflow.start_run():
        model = build_stacked_model()
        model.fit(X_train, y_train)
        acc, mape = evaluate(model, X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("mape", mape)

    joblib.dump(model, model_out)
    print(f"Model saved to {model_out}")
    return model


if __name__ == "__main__":
    train("data/sales_features.parquet")
