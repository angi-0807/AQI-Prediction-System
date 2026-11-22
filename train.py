"""
train.py
--------
Model training for the AQI Prediction System.

Class 12 AI Project - Angel Rose

This file trains multiple models:
  1. Linear Regression (Ridge baseline)
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. XGBoost Regressor
  5. LSTM (Deep Learning model for time-series forecasting)

All trained models are saved in the models/ folder.
"""

import os
import warnings

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _save_model(model, name: str) -> str:
    """Save a sklearn/xgboost model as a pickle file."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Saved model: {path}")
    return path


def _quick_eval(name, model, X_val, y_val) -> dict:
    """Evaluate a model on validation data and print basic metrics."""
    if X_val is None or len(X_val) == 0:
        print(f"  {name:30s}  Skipped validation (empty validation set)")
        return {"model": name, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    pred = model.predict(X_val)
    pred = np.clip(pred, 0, 500)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    mae = float(mean_absolute_error(y_val, pred))
    r2 = float(r2_score(y_val, pred))
    print(f"  {name:30s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------
# 1. Linear Regression (Ridge)
# ---------------------------------------------

def train_linear(X_train, y_train, X_val, y_val):
    print("\nTraining Linear Regression (Ridge)...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    _save_model(model, "linear_regression")
    res = _quick_eval("Linear Regression", model, X_val, y_val)
    return model, res


# ---------------------------------------------
# 2. Decision Tree
# ---------------------------------------------

def train_decision_tree(X_train, y_train, X_val, y_val):
    print("\nTraining Decision Tree...")
    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    _save_model(model, "decision_tree")
    res = _quick_eval("Decision Tree", model, X_val, y_val)
    return model, res


# ---------------------------------------------
# 3. Random Forest
# ---------------------------------------------

def train_random_forest(X_train, y_train, X_val, y_val):
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    _save_model(model, "random_forest")
    res = _quick_eval("Random Forest", model, X_val, y_val)
    return model, res


# ---------------------------------------------
# 4. XGBoost
# ---------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val):
    print("\nTraining XGBoost...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
    )

    # Fit with early stopping when validation data is available
    if X_val is not None and len(X_val) > 0:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        except TypeError:
            # For older/newer xgboost API differences
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    _save_model(model, "xgboost")
    res = _quick_eval("XGBoost", model, X_val, y_val)
    return model, res


# ---------------------------------------------
# 5. LSTM model
# ---------------------------------------------

def build_lstm(seq_len: int, n_features: int) -> keras.Model:
    """Build a stacked LSTM model for AQI prediction."""
    inputs = keras.Input(shape=(seq_len, n_features), name="sequence_input")

    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, name="aqi_output")(x)

    model = keras.Model(inputs, outputs, name="LSTM_AQI")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_lstm(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    epochs: int = 60,
    batch_size: int = 128,
    plot_dir: str = "notebooks",
):
    """Train the LSTM model and save the best version."""
    print("\nTraining LSTM model...")

    if len(X_train_seq) == 0:
        raise ValueError("LSTM training sequences are empty. Check preprocessing and sequence length.")

    seq_len, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
    model = build_lstm(seq_len, n_features)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "lstm_best.keras")
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        callbacks.ModelCheckpoint(model_path, save_best_only=True, verbose=0),
    ]

    validation_data = (X_val_seq, y_val_seq) if len(X_val_seq) > 0 else None
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    # Save final model too (best already saved by checkpoint)
    if not os.path.exists(model_path):
        model.save(model_path)
    print(f"  Saved LSTM model: {model_path}")

    # Validation evaluation
    eval_X, eval_y = (X_val_seq, y_val_seq) if len(X_val_seq) > 0 else (X_train_seq, y_train_seq)
    y_pred = model.predict(eval_X, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, 500)
    rmse = float(np.sqrt(mean_squared_error(eval_y, y_pred)))
    mae = float(mean_absolute_error(eval_y, y_pred))
    r2 = float(r2_score(eval_y, y_pred))
    print(f"  {'LSTM':30s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("loss", []), label="Train Loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("LSTM Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history.get("mae", []), label="Train MAE")
    if "val_mae" in history.history:
        axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("LSTM MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curve_path = os.path.join(plot_dir, "lstm_training.png")
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"  Training curve saved: {curve_path}")

    return model, history, {"model": "LSTM", "rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------
# 7-day forecasting using LSTM
# ---------------------------------------------

def forecast_next_n_days(
    model,
    last_sequence: np.ndarray,
    scaler,
    feature_cols: list[str],
    n_days: int = 7,
    aqi_col_idx: int = None,
) -> np.ndarray:
    """
    Forecast AQI for the next n days using iterative LSTM predictions.

    Important note:
    - last_sequence is already scaled (same scaling used during training)
    - LSTM model outputs AQI in the original AQI scale (not scaled)
    - when feeding predictions back into the sequence, we scale them for AQI lag features

    Returns: AQI predictions in original scale, clipped to [0, 500]
    """
    seq = np.array(last_sequence, dtype=float).copy()
    preds = []

    # Choose an AQI-related feature index for scaling feedback
    if aqi_col_idx is None:
        for candidate in ["AQI_lag_1", "AQI_roll_3", "AQI_roll_7", "AQI"]:
            if candidate in feature_cols:
                aqi_col_idx = feature_cols.index(candidate)
                break
        if aqi_col_idx is None:
            aqi_col_idx = 0
    aqi_col_idx = int(max(0, min(aqi_col_idx, seq.shape[1] - 1)))

    # Pre-compute lag feature indices for simple shifting
    lag_indices = {}
    for i, col in enumerate(feature_cols):
        if col.startswith("AQI_lag_"):
            try:
                lag_num = int(col.split("_")[-1])
                lag_indices[lag_num] = i
            except Exception:
                pass

    for _ in range(n_days):
        x = seq[np.newaxis, ...]  # (1, seq_len, n_features)
        pred_raw = float(model.predict(x, verbose=0).flatten()[0])
        pred_raw = float(np.clip(pred_raw, 0, 500))
        preds.append(pred_raw)

        # Scale predicted AQI for inserting into scaled feature columns
        mean_ = scaler.mean_[aqi_col_idx] if hasattr(scaler, "mean_") else 0.0
        scale_ = scaler.scale_[aqi_col_idx] if hasattr(scaler, "scale_") else 1.0
        scale_ = scale_ if scale_ != 0 else 1.0
        pred_scaled_for_feat = (pred_raw - mean_) / scale_

        new_row = seq[-1].copy()

        # Shift AQI lag features if available (simple approximation)
        if lag_indices:
            max_lag = max(lag_indices)
            for n in range(max_lag, 1, -1):
                if n in lag_indices and (n - 1) in lag_indices:
                    new_row[lag_indices[n]] = new_row[lag_indices[n - 1]]
            if 1 in lag_indices:
                idx1 = lag_indices[1]
                m1 = scaler.mean_[idx1] if hasattr(scaler, "mean_") else 0.0
                s1 = scaler.scale_[idx1] if hasattr(scaler, "scale_") else 1.0
                s1 = s1 if s1 != 0 else 1.0
                new_row[idx1] = (pred_raw - m1) / s1
        else:
            new_row[aqi_col_idx] = pred_scaled_for_feat

        # Also update AQI rolling features with scaled predicted value (rough approximation)
        for i, col in enumerate(feature_cols):
            if col.startswith("AQI_roll_"):
                m = scaler.mean_[i] if hasattr(scaler, "mean_") else 0.0
                s = scaler.scale_[i] if hasattr(scaler, "scale_") else 1.0
                s = s if s != 0 else 1.0
                new_row[i] = (pred_raw - m) / s

        seq = np.vstack([seq[1:], new_row])

    return np.clip(np.array(preds, dtype=float), 0, 500)


# ---------------------------------------------
# Train all sklearn models
# ---------------------------------------------

def train_all_sklearn(X_train, y_train, X_val, y_val):
    """Train all traditional ML models and return validation metrics."""
    results = []
    for fn in [train_linear, train_decision_tree, train_random_forest, train_xgboost]:
        _, res = fn(X_train, y_train, X_val, y_val)
        results.append(res)
    return results
