"""Load data (Open Power System Data) and weather (Open-Meteo archive)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# Largest German metro areas with approximate population (millions), used as weights.
CITIES = {
    "Berlin": (52.52, 13.40, 3.7),
    "Hamburg": (53.55, 9.99, 1.9),
    "Munich": (48.14, 11.58, 1.5),
    "Cologne": (50.94, 6.96, 1.1),
    "Frankfurt": (50.11, 8.68, 0.8),
    "Stuttgart": (48.78, 9.18, 0.6),
    "Dusseldorf": (51.23, 6.78, 0.6),
    "Leipzig": (51.34, 12.37, 0.6),
    "Dortmund": (51.51, 7.47, 0.6),
    "Hanover": (52.38, 9.73, 0.5),
}
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"]


def load_opsd(path: Path | None = None) -> pd.DataFrame:
    """Hourly German load (MW): ``load`` (actual) and ``entsoe_forecast`` (TSO day-ahead forecast)."""
    path = path or DATA / "opsd_60min.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(OPSD_URL, path)
    cols = ["utc_timestamp", "DE_load_actual_entsoe_transparency", "DE_load_forecast_entsoe_transparency"]
    df = pd.read_csv(path, usecols=cols, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    df.columns = ["load", "entsoe_forecast"]
    df = df.loc["2015-01-01":"2020-09-30"]
    df.index = df.index.tz_convert("UTC")
    return df.asfreq("h")


def _get_json(url: str, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read())
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_weather(start: str = "2014-12-01", end: str = "2020-09-30") -> pd.DataFrame:
    """Population-weighted hourly weather over the largest German cities (cached as parquet)."""
    cache = DATA / f"weather_{start}_{end}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames, weights = [], []
    for lat, lon, pop in CITIES.values():
        query = urllib.parse.urlencode(
            {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "hourly": ",".join(WEATHER_VARS), "timezone": "UTC"}
        )
        hourly = _get_json(f"{WEATHER_URL}?{query}")["hourly"]
        frame = pd.DataFrame(hourly).set_index("time")
        frame.index = pd.to_datetime(frame.index).tz_localize("UTC")
        frames.append(frame * pop)
        weights.append(pop)
    weather = sum(frames) / sum(weights)
    weather.to_parquet(cache)
    return weather
