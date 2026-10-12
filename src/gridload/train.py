"""Tune, train and evaluate the day-ahead model.

    python -m gridload.train --trials 60

Splits: train 2015-2018 · validation 2019 (Optuna) · test 2020-01..2020-09 (touched once).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import optuna
import pandas as pd

from .data import load_opsd, load_weather
from .features import build_features
from .metrics import coverage, report

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[2] / "results"
SEED = 7
WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m",
                "hdd", "cdd", "temp_roll24", "temp_x_hour"]  # fmt: skip
HOLIDAY_COLS = ["holiday_share", "holiday_prev", "holiday_next", "bridge_day", "christmas_period"]


def split(df: pd.DataFrame):
    train = df.loc[:"2018-12-31"]
    val = df.loc["2019-01-01":"2019-12-31"]
    test = df.loc["2020-01-01":]
    return train, val, test


def fit(params: dict, X: pd.DataFrame, y: pd.Series, rounds: int, **extra) -> lgb.Booster:
    return lgb.train({**params, **extra, "verbosity": -1, "seed": SEED}, lgb.Dataset(X, y), num_boost_round=rounds)


def tune(Xtr, ytr, Xva, yva, trials: int) -> tuple[dict, int]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression_l1",  # MAE objective: robust to outliers, optimises close to MAPE
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            "verbosity": -1,
            "seed": SEED,
        }
        model = lgb.train(
            params, lgb.Dataset(Xtr, ytr), num_boost_round=4000,
            valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)],
        )  # fmt: skip
        trial.set_user_attr("rounds", model.best_iteration)
        pred = model.predict(Xva, num_iteration=model.best_iteration)
        return float((abs(yva - pred) / yva).mean() * 100)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = {**study.best_params, "objective": "regression_l1", "bagging_freq": 1}
    print(f"best validation MAPE {study.best_value:.3f}% after {trials} trials")
    return best, study.best_trial.user_attrs["rounds"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RESULTS.mkdir(exist_ok=True)

    load = load_opsd()
    feats = build_features(load, load_weather())
    df = feats.join(load).dropna(subset=["lag_336", "roll168_std", "load"])
    features = list(feats.columns)
    train, val, test = split(df)

    params, rounds = tune(train[features], train["load"], val[features], val["load"], args.trials)
    final_rounds = int(rounds * 1.15)  # slightly more rounds for the larger train+val set
    trainval = pd.concat([train, val])
    model = fit(params, trainval[features], trainval["load"], final_rounds)
    # Split-conformal 80% intervals: absolute errors of a train-only model on the 2019
    # calibration year, one width per local hour of day (errors peak in the morning ramp).
    calibration = fit(params, train[features], train["load"], rounds)
    abs_err = (val["load"] - calibration.predict(val[features])).abs()
    hour_of = val["hour"]
    n = abs_err.groupby(hour_of).transform("size")
    width = abs_err.groupby(hour_of).quantile(min(1.0, (n.iloc[0] + 1) * 0.8 / n.iloc[0]))

    y = test["load"]
    pred = pd.Series(model.predict(test[features]), index=test.index)
    half = test["hour"].map(width)
    band_lo, band_hi = pred - half, pred + half
    seasonal_naive = test["lag_168"]
    entsoe = test["entsoe_forecast"]

    results = {
        "test_period": f"{test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d}",
        "models": {
            "LightGBM (this repo)": report(y, pred),
            "ENTSO-E / TSO day-ahead forecast": report(y, entsoe),
            "Seasonal naive (same hour last week)": report(y, seasonal_naive),
        },
        "interval_80pct": {
            "coverage_2020": round(coverage(y, band_lo, band_hi), 1),
            "coverage_by_month": {
                pd.Timestamp(2020, m, 1).strftime("%b"): round(coverage(g, band_lo[g.index], band_hi[g.index]), 1)
                for m, g in y.groupby(y.index.month)
            },
            "mean_width_mw": round(float((band_hi - band_lo).mean()), 1),
        },
        "best_params": params,
        "rounds": final_rounds,
    }

    # Ablations: retrain without feature groups, same params.
    for name, drop in [("without weather", WEATHER_COLS), ("without holiday features", HOLIDAY_COLS)]:
        cols = [c for c in features if c not in drop]
        m = fit(params, trainval[cols], trainval["load"], final_rounds)
        results["models"][f"LightGBM {name}"] = report(y, pd.Series(m.predict(test[cols]), index=test.index))

    monthly = pd.DataFrame(
        {
            "LightGBM": ((y - pred).abs() / y).groupby(y.index.month).mean() * 100,
            "ENTSO-E": ((y - entsoe).abs() / y).groupby(y.index.month).mean() * 100,
        }
    ).round(2)
    monthly.index = [pd.Timestamp(2020, m, 1).strftime("%b") for m in monthly.index]
    results["monthly_mape"] = monthly.to_dict(orient="index")
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["models"], indent=2), "\n80% interval:", results["interval_80pct"])

    plot_weeks(y, pred, band_lo, band_hi, entsoe)
    plot_monthly(monthly)
    plot_importance(model, features)
    model.save_model(str(RESULTS / "model.txt"))


def plot_weeks(y, pred, lo, hi, entsoe) -> None:
    weeks = [("2020-01-13", "Normal winter week (Jan 2020)"), ("2020-03-23", "First COVID-19 lockdown week (Mar 2020)")]
    fig, axes = plt.subplots(len(weeks), 1, figsize=(11, 7), sharey=True)
    for ax, (start, title) in zip(axes, weeks):
        window = slice(start, str(pd.Timestamp(start) + pd.Timedelta(days=7)))
        ax.fill_between(y.loc[window].index, lo.loc[window] / 1e3, hi.loc[window] / 1e3, alpha=0.2, label="80% interval")
        ax.plot(y.loc[window] / 1e3, color="black", lw=1.6, label="Actual")
        ax.plot(pred.loc[window] / 1e3, lw=1.3, label="LightGBM")
        ax.plot(entsoe.loc[window] / 1e3, lw=1.1, ls="--", label="ENTSO-E forecast")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Load (GW)")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast_weeks.png", dpi=140)
    plt.close(fig)


def plot_monthly(monthly: pd.DataFrame) -> None:
    ax = monthly.plot.bar(figsize=(9, 4), rot=0, width=0.75)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Day-ahead error by month, 2020 test period", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "monthly_mape.png", dpi=140)
    plt.close()


def plot_importance(model: lgb.Booster, features: list[str]) -> None:
    gain = pd.Series(model.feature_importance("gain"), index=features).sort_values().tail(15)
    ax = (gain / gain.sum() * 100).plot.barh(figsize=(8, 5))
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("Top 15 features", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
