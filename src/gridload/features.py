"""Feature engineering for day-ahead load forecasting.

Day-ahead means: the forecast for every hour of day D is issued on day D-1
around noon. Load is therefore only known up to ~36 hours before the earliest
target hour, so every load-derived feature uses lags of at least 48 hours.

Weather features use *observed* weather as a stand-in for the weather forecast
that would be available at issue time (a common simplification; it makes the
weather signal slightly optimistic).
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

MIN_LAG = 48

# State populations (millions, 2020) for the regional-holiday share feature.
STATES = {
    "BW": 11.1, "BY": 13.1, "BE": 3.7, "BB": 2.5, "HB": 0.7, "HH": 1.9, "HE": 6.3, "MV": 1.6,
    "NI": 8.0, "NW": 17.9, "RP": 4.1, "SL": 1.0, "SN": 4.1, "ST": 2.2, "SH": 2.9, "TH": 2.1,
}  # fmt: skip


def holiday_share(days: pd.DatetimeIndex) -> pd.Series:
    """Fraction of Germany's population with a public holiday on each day."""
    years = sorted(set(days.year))
    total = sum(STATES.values())
    share = pd.Series(0.0, index=days)
    for state, pop in STATES.items():
        calendar = holidays.Germany(subdiv=state, years=years)
        share += np.array([d.date() in calendar for d in days]) * pop / total
    return share


def build_features(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned with ``load.index`` (UTC, hourly)."""
    idx = load.index
    local = idx.tz_convert("Europe/Berlin")
    f = pd.DataFrame(index=idx)

    # Calendar (local time drives behaviour).
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["month"] = local.month
    f["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    f["is_weekend"] = (local.dayofweek >= 5).astype(int)
    days = pd.DatetimeIndex(local.normalize().tz_localize(None))
    unique_days = days.unique()
    share = holiday_share(unique_days)
    f["holiday_share"] = share.reindex(days).to_numpy()
    prev_day, next_day = share.shift(1, freq="D"), share.shift(-1, freq="D")
    f["holiday_prev"] = prev_day.reindex(days).fillna(0).to_numpy()
    f["holiday_next"] = next_day.reindex(days).fillna(0).to_numpy()
    # Bridge days: a weekday squeezed between a holiday and a weekend.
    f["bridge_day"] = (
        ((f["dow"] == 4) & (f["holiday_prev"] > 0.5)) | ((f["dow"] == 0) & (f["holiday_next"] > 0.5))
    ).astype(int)
    f["christmas_period"] = (((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 1))).astype(int)

    # Load lags (>= MIN_LAG hours).
    y = load["load"]
    for lag in (48, 72, 96, 168, 336):
        f[f"lag_{lag}"] = y.shift(lag)
    lagged = y.shift(MIN_LAG)
    f["roll24_mean"] = lagged.rolling(24).mean()
    f["roll168_mean"] = lagged.rolling(168).mean()
    f["roll168_std"] = lagged.rolling(168).std()
    f["same_hour_week_mean"] = sum(y.shift(24 * k) for k in range(2, 9)) / 7  # same hour, days D-2..D-8

    # Weather.
    w = weather.reindex(idx)
    for col in w.columns:
        f[col] = w[col].to_numpy()
    t = w["temperature_2m"]
    f["hdd"] = (15.0 - t).clip(lower=0)
    f["cdd"] = (t - 22.0).clip(lower=0)
    f["temp_roll24"] = t.rolling(24, min_periods=1).mean().to_numpy()  # thermal inertia of buildings
    f["temp_x_hour"] = t.to_numpy() * np.cos(2 * np.pi * f["hour"] / 24)
    return f
