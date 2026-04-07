"""
Feature Engineering Pipeline
Author: Shebin S Illikkal
"""
import pandas as pd
import numpy as np


def add_lag_features(df: pd.DataFrame, target: str, lags: list) -> pd.DataFrame:
    for lag in lags:
        df[f'{target}_lag_{lag}'] = df.groupby(['store_id','sku_id'])[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str, windows: list) -> pd.DataFrame:
    for w in windows:
        grp = df.groupby(['store_id','sku_id'])[target]
        df[f'{target}_roll_mean_{w}'] = grp.transform(lambda x: x.shift(1).rolling(w).mean())
        df[f'{target}_roll_std_{w}']  = grp.transform(lambda x: x.shift(1).rolling(w).std())
    return df


def add_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    dt = pd.to_datetime(df[date_col])
    df['week']        = dt.dt.isocalendar().week.astype(int)
    df['month']       = dt.dt.month
    df['quarter']     = dt.dt.quarter
    df['is_month_end']= dt.dt.is_month_end.astype(int)
    df['is_weekend']  = dt.dt.dayofweek.isin([5, 6]).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df, 'sales', [1, 2, 4, 8, 13, 26, 52])
    df = add_rolling_features(df, 'sales', [4, 8, 13, 26])
    df = add_date_features(df)
    df = df.dropna()
    return df
