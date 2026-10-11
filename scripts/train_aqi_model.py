"""Train and evaluate a 3-hour-ahead Delhi AQI baseline model.

Uses time-based features and exact timestamp joins for lags/targets so missing
hours are not silently treated as adjacent observations. A chronological
holdout is used; training labels must also fall before the cutoff.
"""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "delhi_aqi_long.csv"
OUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
HORIZON_HOURS = 3
LAGS = [1, 3, 6, 24]


def aqi_category(values):
    """Indian AQI bands used only for an additional category-agreement metric."""
    values = np.asarray(values, dtype=float)
    return np.select(
        [values <= 50, values <= 100, values <= 200, values <= 300, values <= 400],
        [0, 1, 2, 3, 4],
        default=5,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df["station_name"] = df["station_name"].astype("string").str.strip()
    df = df.dropna(subset=["timestamp", "station_name", "aqi"])
    df = df[(df["aqi"] >= 0) & (df["aqi"] <= 500)]
    # Ensure unique and stable station IDs even if a previous processed file
    # still contains the generic source ID.
    df["station_id"] = (
        "delhi_" + df["station_name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    df = df.drop_duplicates(["station_id", "timestamp"], keep="last")
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    base = df[["timestamp", "station_id", "station_name", "aqi"]].copy()
    base = base.rename(columns={"aqi": "aqi_now"})

    # Exact-hour lag joins avoid using a prior observation that may be several
    # hours old when a timestamp is missing.
    for lag in LAGS:
        lag_frame = df[["station_id", "timestamp", "aqi"]].copy()
        lag_frame["timestamp"] = lag_frame["timestamp"] + pd.Timedelta(hours=lag)
        lag_frame = lag_frame.rename(columns={"aqi": f"aqi_lag_{lag}h"})
        base = base.merge(lag_frame, on=["station_id", "timestamp"], how="left", validate="one_to_one")

    target = df[["station_id", "timestamp", "aqi"]].copy()
    target["timestamp"] = target["timestamp"] - pd.Timedelta(hours=HORIZON_HOURS)
    target = target.rename(columns={"aqi": "target_aqi"})
    base = base.merge(target, on=["station_id", "timestamp"], how="left", validate="one_to_one")

    base["target_timestamp"] = base["timestamp"] + pd.Timedelta(hours=HORIZON_HOURS)
    base["hour"] = base["timestamp"].dt.hour
    base["day_of_week"] = base["timestamp"].dt.dayofweek
    base["month"] = base["timestamp"].dt.month
    base["day_of_year"] = base["timestamp"].dt.dayofyear
    base["hour_sin"] = np.sin(2 * np.pi * base["hour"] / 24)
    base["hour_cos"] = np.cos(2 * np.pi * base["hour"] / 24)
    base["doy_sin"] = np.sin(2 * np.pi * base["day_of_year"] / 365.25)
    base["doy_cos"] = np.cos(2 * np.pi * base["day_of_year"] / 365.25)
    # Categorical dtype lets HistGradientBoosting handle stations without
    # pretending their IDs have a numeric order.
    base["station_id"] = base["station_id"].astype("category")
    return base


def main() -> int:
    if not DATA_FILE.exists():
        print(f"Missing {DATA_FILE}. Run the data preparation scripts first.")
        return 1

    print("Loading Delhi AQI observations...")
    raw = pd.read_csv(DATA_FILE)
    needed = {"timestamp", "station_name", "aqi"}
    if not needed.issubset(raw.columns):
        print(f"Dataset must contain columns: {sorted(needed)}; found {list(raw.columns)}")
        return 2
    df = build_features(raw)
    feature_cols = [
        "aqi_now", "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
        "hour", "day_of_week", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "station_id"
    ]
    model_data = df.dropna(subset=feature_cols + ["target_aqi"]).copy()
    if len(model_data) < 1000:
        print(f"Not enough training samples after exact lag/target matching: {len(model_data):,}")
        return 3

    unique_times = np.array(sorted(model_data["timestamp"].dropna().unique()))
    train_cutoff = pd.Timestamp(unique_times[int(len(unique_times) * 0.70)])
    test_cutoff = pd.Timestamp(unique_times[int(len(unique_times) * 0.80)])
    # Three chronological regions: training, validation for blend selection,
    # and untouched final test. Training labels must be before train_cutoff.
    train = model_data[model_data["target_timestamp"] < train_cutoff]
    validation = model_data[
        (model_data["timestamp"] >= train_cutoff)
        & (model_data["target_timestamp"] < test_cutoff)
    ]
    test = model_data[model_data["timestamp"] >= test_cutoff]
    if len(train) < 500 or len(validation) < 100 or len(test) < 100:
        print(
            f"Insufficient split size: train={len(train):,}, "
            f"validation={len(validation):,}, test={len(test):,}"
        )
        return 4

    X_train, y_train = train[feature_cols], train["target_aqi"]
    X_val, y_val = validation[feature_cols], validation["target_aqi"]
    X_test, y_test = test[feature_cols], test["target_aqi"]
    print(f"Training samples:   {len(train):,}")
    print(f"Validation samples: {len(validation):,}")
    print(f"Test samples:       {len(test):,}")
    print(f"Validation starts:  {train_cutoff}")
    print(f"Final test starts:  {test_cutoff}")
    print("Training HistGradientBoostingRegressor (3-hour-ahead forecast)...")

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=25,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # Learn how much to trust the ML model versus a simple persistence forecast
    # using the validation period only. This weight is locked before final test.
    val_model_pred = np.clip(model.predict(X_val), 0, 500)
    val_persistence = np.clip(X_val["aqi_now"].to_numpy(dtype=float), 0, 500)
    blend_candidates = np.linspace(0.0, 1.0, 21)
    blend_results = [
        (float(mean_absolute_error(y_val, (1 - w) * val_persistence + w * val_model_pred)), float(w))
        for w in blend_candidates
    ]
    validation_mae, model_weight = min(blend_results, key=lambda item: (item[0], item[1]))

    model_pred = np.clip(model.predict(X_test), 0, 500)
    persistence = np.clip(X_test["aqi_now"].to_numpy(dtype=float), 0, 500)
    ensemble_pred = np.clip((1 - model_weight) * persistence + model_weight * model_pred, 0, 500)

    model_mae = float(mean_absolute_error(y_test, model_pred))
    model_rmse = float(np.sqrt(mean_squared_error(y_test, model_pred)))
    persistence_mae = float(mean_absolute_error(y_test, persistence))
    persistence_rmse = float(np.sqrt(mean_squared_error(y_test, persistence)))
    ensemble_mae = float(mean_absolute_error(y_test, ensemble_pred))
    ensemble_rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_pred)))
    ensemble_r2 = float(r2_score(y_test, ensemble_pred))
    band_acc = float(accuracy_score(aqi_category(y_test), aqi_category(ensemble_pred)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "aqi_forecast_model.joblib"
    dump({"model": model, "feature_columns": feature_cols, "horizon_hours": HORIZON_HOURS, "model_weight": model_weight}, model_path)

    predictions = test[["timestamp", "target_timestamp", "station_id", "station_name", "aqi_now", "target_aqi"]].copy()
    # Convert categorical IDs to strings before CSV export.
    predictions["station_id"] = predictions["station_id"].astype(str)
    predictions["ml_model_aqi_3h"] = model_pred
    predictions["persistence_aqi_3h"] = persistence
    predictions["ensemble_aqi_3h"] = ensemble_pred
    predictions["absolute_error"] = np.abs(predictions["target_aqi"] - predictions["ensemble_aqi_3h"])
    pred_path = OUT_DIR / "test_predictions.csv"
    predictions.to_csv(pred_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    metrics = {
        "target": "AQI 3 hours after feature timestamp",
        "horizon_hours": HORIZON_HOURS,
        "data_rows_used": int(len(df)),
        "usable_samples": int(len(model_data)),
        "train_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "test_samples": int(len(test)),
        "validation_start": str(train_cutoff),
        "final_test_start": str(test_cutoff),
        "unique_stations": int(df["station_id"].nunique()),
        "model": "HistGradientBoostingRegressor",
        "validation_selected_ml_model_weight": round(model_weight, 2),
        "validation_best_blend_MAE": round(validation_mae, 3),
        "ml_model_MAE": round(model_mae, 3),
        "ml_model_RMSE": round(model_rmse, 3),
        "persistence_baseline_MAE": round(persistence_mae, 3),
        "persistence_baseline_RMSE": round(persistence_rmse, 3),
        "blended_ensemble_MAE": round(ensemble_mae, 3),
        "blended_ensemble_RMSE": round(ensemble_rmse, 3),
        "blended_ensemble_R2": round(ensemble_r2, 4),
        "aqi_category_agreement": round(band_acc, 4),
        "ensemble_beats_persistence_on_MAE": bool(ensemble_mae < persistence_mae),
        "ensemble_beats_persistence_on_RMSE": bool(ensemble_rmse < persistence_rmse),
        "caveat": "Historical 2023 dataset only; not a live forecast. Performance may not generalise to 2024-2026. Chronological train/validation/test split; ensemble weight selected on validation only.",
    }
    metrics_path = OUT_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nPHASE 2 TRAINING COMPLETE")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Test predictions saved: {pred_path}")
    print("\nFinal-test evaluation (3-hour-ahead):")
    print(f"  ML model MAE:             {model_mae:.2f}")
    print(f"  Persistence baseline MAE: {persistence_mae:.2f}")
    print(f"  Blended ensemble MAE:     {ensemble_mae:.2f}")
    print(f"  ML model RMSE:            {model_rmse:.2f}")
    print(f"  Persistence baseline RMSE:{persistence_rmse:.2f}")
    print(f"  Blended ensemble RMSE:    {ensemble_rmse:.2f}")
    print(f"  Blended ensemble R²:      {ensemble_r2:.3f}")
    print(f"  AQI category agreement:   {band_acc:.1%}")
    print(f"  ML blend weight selected on validation: {model_weight:.2f}")
    print(f"  Beats baseline on MAE:    {'YES' if ensemble_mae < persistence_mae else 'NO'}")
    print("\nUse the reported metrics honestly; the dataset is historical 2023 data, not a live forecast.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
