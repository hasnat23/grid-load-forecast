import numpy as np
import pandas as pd
import pytest

from gridload.features import MIN_LAG, build_features, holiday_share
from gridload.metrics import coverage, mape, report


@pytest.fixture(scope="module")
def frames():
    idx = pd.date_range("2019-11-01", "2020-01-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    load = pd.DataFrame({"load": 55_000 + 5_000 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 500, len(idx))}, index=idx)
    weather = pd.DataFrame(
        {"temperature_2m": 5.0, "relative_humidity_2m": 80.0, "cloud_cover": 50.0, "shortwave_radiation": 0.0, "wind_speed_10m": 10.0},
        index=idx,
    )
    return load, weather, build_features(load, weather)


def test_no_lookahead(frames):
    """Changing load in the last MIN_LAG hours must not change any feature at the final hour."""
    load, weather, feats = frames
    tampered = load.copy()
    tampered.iloc[-MIN_LAG:] *= 10
    assert build_features(tampered, weather).iloc[-1].equals(feats.iloc[-1])


def test_lag_columns_respect_min_lag(frames):
    lags = [int(c.split("_")[1]) for c in frames[2].columns if c.startswith("lag_")]
    assert min(lags) >= MIN_LAG


def test_holiday_share():
    days = pd.DatetimeIndex(["2020-12-25", "2020-06-11", "2020-06-12", "2020-11-01"])
    share = holiday_share(days)
    assert share.iloc[0] == pytest.approx(1.0)  # Christmas: nationwide
    assert 0.3 < share.iloc[1] < 0.7  # Corpus Christi: southern/western states only
    assert share.iloc[2] == 0.0
    assert 0.4 < share.iloc[3] < 0.7  # All Saints: BW, BY, NW, RP, SL


def test_bridge_day_and_calendar(frames):
    feats = frames[2]
    # 2019-12-27 is a Friday after Boxing Day (Thursday): a bridge day.
    friday = feats.loc["2019-12-27 10:00"]
    assert friday["bridge_day"] == 1 and friday["christmas_period"] == 1
    assert feats.loc["2020-01-15 10:00", "bridge_day"] == 0
    assert set(feats["hour"].unique()) == set(range(24))


def test_metrics():
    y = pd.Series([100.0, 200.0])
    p = pd.Series([110.0, 180.0])
    assert mape(y, p) == pytest.approx(10.0)
    assert coverage(y, p - 15, p + 15) == 50.0
    assert report(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))["hours"] == 1
