"""
evaluate.py
-----------
Evaluation utilities for the AQI Prediction System.

Class 12 AI Project - Angel Rose

This file includes:
  - Regression metrics (RMSE, MAE, R2)
  - AQI category accuracy
  - Prediction and residual plots
  - Category confusion matrix
  - SHAP explainability for tree models
  - Model comparison chart
  - 7-day forecast plot
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# SHAP is optional at runtime in case it's not installed
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


# ---------------------------------------------
# AQI category helpers
# ---------------------------------------------

CAT_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def aqi_to_category(aqi: float) -> str:
    """Map AQI value to CPCB category."""
    if pd.isna(aqi):
        return "Unknown"
    aqi = float(np.clip(aqi, 0, 1000))
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def aqi_to_categories(values: np.ndarray) -> np.ndarray:
    return np.array([aqi_to_category(v) for v in values], dtype=object)


# ---------------------------------------------
# Core evaluation
# ---------------------------------------------

def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_dir: str = "notebooks",
    save_plots: bool = True,
) -> dict:
    """Evaluate regression predictions and optionally save plots."""
    os.makedirs(plot_dir, exist_ok=True)

    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()
    y_pred = np.clip(y_pred, 0, 500)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    cat_true = aqi_to_categories(y_true)
    cat_pred = aqi_to_categories(y_pred)
    cat_acc = float((cat_true == cat_pred).mean()) if len(cat_true) else 0.0

    print(f"\n{'=' * 55}")
    print(f"  Model Evaluation: {model_name}")
    print(f"{'=' * 55}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  R2 Score     : {r2:.4f}")
    print(f"  Category Acc : {cat_acc * 100:.2f}%")

    if save_plots and len(y_true) > 0:
        _plot_predictions(model_name, y_true, y_pred, plot_dir)
        _plot_residuals(model_name, y_true, y_pred, plot_dir)

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cat_acc": cat_acc,
    }


def _safe_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _plot_predictions(name, y_true, y_pred, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted scatter
    max_val = float(max(np.max(y_true), np.max(y_pred), 1))
    lim = [0, max_val + 20]
    axes[0].scatter(y_true, y_pred, alpha=0.25, s=10, color="#1565C0")
    axes[0].plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Actual AQI")
    axes[0].set_ylabel("Predicted AQI")
    axes[0].set_title(f"Actual vs Predicted - {name}")
    axes[0].legend()

    # Time-series overlay (sample)
    n = min(200, len(y_true))
    idx = np.arange(n)
    axes[1].plot(idx, y_true[:n], label="Actual", color="#E53935", lw=1.5)
    axes[1].plot(idx, y_pred[:n], label="Predicted", color="#1E88E5", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("AQI")
    axes[1].set_title(f"Prediction Overlay (first {n} samples) - {name}")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(plot_dir, f"pred_{_safe_name(name)}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residuals(name, y_true, y_pred, plot_dir):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.2, s=10, color="#6A1B9A")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted AQI")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title(f"Residuals - {name}")

    axes[1].hist(residuals, bins=40, color="#6A1B9A", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual Distribution - {name}")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"residuals_{_safe_name(name)}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------
# Category confusion matrix
# ---------------------------------------------

def category_confusion_matrix(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, plot_dir: str = "notebooks"):
    os.makedirs(plot_dir, exist_ok=True)
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.clip(np.array(y_pred, dtype=float).flatten(), 0, 500)

    cat_true = aqi_to_categories(y_true)
    cat_pred = aqi_to_categories(y_pred)

    present = [c for c in CAT_ORDER if c in set(cat_true) or c in set(cat_pred)]
    if not present:
        print(f"No category data available for {model_name}")
        return

    cm = confusion_matrix(cat_true, cat_pred, labels=present)

    print(f"\n--- AQI Category Report: {model_name} ---")
    print(classification_report(cat_true, cat_pred, labels=present, zero_division=0))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", xticklabels=present, yticklabels=present)
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.title(f"AQI Category Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"cat_cm_{_safe_name(model_name)}.png")
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------
# SHAP explainability
# ---------------------------------------------

def explain_with_shap(
    model,
    X_sample: np.ndarray,
    feature_names: list[str],
    model_name: str = "Model",
    n_samples: int = 300,
    plot_dir: str = "notebooks",
):
    """Generate SHAP plots for supported models."""
    os.makedirs(plot_dir, exist_ok=True)

    if not SHAP_AVAILABLE:
        print(f"  SHAP not installed, skipping explainability for {model_name}.")
        return

    if X_sample is None or len(X_sample) == 0:
        print(f"  SHAP skipped for {model_name}: empty sample.")
        return

    X_sub = np.array(X_sample[:n_samples])
    safe = _safe_name(model_name)

    try:
        # Use TreeExplainer for tree-based models, otherwise a generic explainer
        model_cls = model.__class__.__name__.lower()
        if any(k in model_cls for k in ["forest", "tree", "xgb", "boost"]):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sub)
        else:
            explainer = shap.Explainer(model, X_sub)
            explanation = explainer(X_sub)
            shap_values = explanation.values

        # For classifiers/regressors returning list, pick first
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sub, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"shap_bar_{safe}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sub, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"shap_beeswarm_{safe}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  SHAP plots saved for {model_name}.")
    except Exception as e:
        print(f"  SHAP skipped for {model_name}: {e}")


# ---------------------------------------------
# Forecast plot
# ---------------------------------------------

def plot_forecast(
    city: str,
    historical_aqi: np.ndarray,
    forecast_aqi: np.ndarray,
    historical_dates,
    plot_dir: str = "notebooks",
):
    """Plot recent AQI history and 7-day forecast."""
    os.makedirs(plot_dir, exist_ok=True)

    historical_aqi = np.array(historical_aqi, dtype=float).flatten()
    forecast_aqi = np.array(forecast_aqi, dtype=float).flatten()
    historical_dates = pd.to_datetime(pd.Series(historical_dates)).reset_index(drop=True)

    if len(historical_dates) == 0 or len(historical_aqi) == 0 or len(forecast_aqi) == 0:
        print("  Forecast plot skipped: insufficient data.")
        return

    n_hist = min(60, len(historical_aqi), len(historical_dates))
    hist_dates = historical_dates.iloc[-n_hist:]
    hist_vals = historical_aqi[-n_hist:]

    last_date = pd.to_datetime(hist_dates.iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(forecast_aqi), freq="D")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist_dates, hist_vals, label="Historical AQI", color="#1565C0", lw=2)
    ax.plot(future_dates, forecast_aqi, label="Forecast", color="#E53935", lw=2.5, linestyle="--", marker="o")

    bands = [
        (0, 50, "#2ECC71", "Good"),
        (50, 100, "#82E0AA", "Satisfactory"),
        (100, 200, "#F4D03F", "Moderate"),
        (200, 300, "#E67E22", "Poor"),
        (300, 500, "#E74C3C", "Very Poor/Severe"),
    ]
    for lo, hi, col, label in bands:
        ax.axhspan(lo, hi, alpha=0.07, color=col, label=label)

    ax.set_title(f"7-Day AQI Forecast - {city}")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.set_ylim(0, max(500, float(np.max(hist_vals)) + 20, float(np.max(forecast_aqi)) + 20))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()

    path = os.path.join(plot_dir, f"forecast_{_safe_name(city)}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Forecast plot saved: {path}")


# ---------------------------------------------
# Model comparison
# ---------------------------------------------

def compare_models(results: list[dict], plot_dir: str = "notebooks") -> pd.DataFrame:
    """Create a comparison table and bar charts for all models."""
    os.makedirs(plot_dir, exist_ok=True)
    if not results:
        print("No results to compare.")
        return pd.DataFrame()

    df = pd.DataFrame(results).copy()
    for col in ["rmse", "mae", "r2", "cat_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rmse" in df.columns:
        df = df.sort_values("rmse", ascending=True)

    print("\n--- Model Comparison (sorted by RMSE) ---")
    cols = [c for c in ["model", "rmse", "mae", "r2", "cat_acc"] if c in df.columns]
    print(df[cols].to_string(index=False))

    names = df["model"].astype(str).tolist()
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metric_info = [
        ("rmse", "#E53935", "RMSE (lower is better)"),
        ("mae", "#1E88E5", "MAE (lower is better)"),
        ("r2", "#43A047", "R2 Score (higher is better)"),
    ]

    for ax, (metric, color, title) in zip(axes, metric_info):
        vals = df[metric].fillna(0).values if metric in df.columns else np.zeros(len(names))
        ax.bar(x, vals, color=color, edgecolor="white", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(title)
        for i, v in enumerate(vals):
            offset = 0.01 * (max(vals) - min(vals) + 1)
            ax.text(i, v + offset, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Model Comparison - AQI Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(plot_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Comparison chart saved: {path}")
    return df
