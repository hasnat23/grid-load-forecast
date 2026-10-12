from __future__ import annotations

import numpy as np
import pandas as pd


def mape(y: pd.Series, p: pd.Series) -> float:
    return float(np.mean(np.abs((y - p) / y)) * 100)


def mae(y: pd.Series, p: pd.Series) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: pd.Series, p: pd.Series) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def coverage(y: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    return float(np.mean((y >= lo) & (y <= hi)) * 100)


def report(y: pd.Series, p: pd.Series) -> dict:
    mask = y.notna() & p.notna()
    y, p = y[mask], p[mask]
    return {"MAPE %": round(mape(y, p), 3), "MAE MW": round(mae(y, p), 1), "RMSE MW": round(rmse(y, p), 1), "hours": int(mask.sum())}
