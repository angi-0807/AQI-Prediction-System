"""
main.py
-------
End-to-end pipeline runner for the AQI Prediction System.

Class 12 AI Project - Angel Rose

This script runs the full project flow:
  1. Preprocess the dataset
  2. Train ML models (Linear Regression, Decision Tree, Random Forest, XGBoost)
  3. Train LSTM model for time-series forecasting
  4. Evaluate all models on the test set
  5. Generate SHAP plots for tree models
  6. Generate a 7-day forecast for the most polluted city
  7. Create model comparison chart

Run:
    python main.py
"""

import os
import sys
import warnings

import numpy as np
import joblib

# Add project root to path so imports work when run from anywhere
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocess import run_preprocessing, build_sequences
from train import train_all_sklearn, train_lstm, forecast_next_n_days
from evaluate import (
    evaluate_model,
    category_confusion_matrix,
    explain_with_shap,
    compare_models,
    plot_forecast,
)

warnings.filterwarnings("ignore")

# Create output folders
os.makedirs("notebooks", exist_ok=True)
os.makedirs("models", exist_ok=True)


def main():
    print("=" * 60)
    print(" AQI Prediction System - Class 12 AI Project")
    print(" Made by: Angel Rose")
    print("=" * 60)

    # -----------------------------------------
    # 1. Preprocessing
    # -----------------------------------------
    print("\n[Step 1] Preprocessing data...")
    artefacts = run_preprocessing(
        data_path=os.path.join(PROJECT_ROOT, "data", "city_day.csv"),
        run_eda=True,
        lags=7,
    )

    X_train = artefacts["X_train"]
    X_val = artefacts["X_val"]
    X_test = artefacts["X_test"]
    y_train = artefacts["y_train"]
    y_val = artefacts["y_val"]
    y_test = artefacts["y_test"]
    scaler = artefacts["scaler"]
    feature_cols = artefacts["feature_cols"]
    df = artefacts["df"]
    train_df = artefacts["train_df"]
    val_df = artefacts["val_df"]
    test_df = artefacts["test_df"]
    city_encoder = artefacts["city_encoder"]

    # Save preprocessing artefacts for the Streamlit app
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(city_encoder, os.path.join("models", "city_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join("models", "feature_cols.pkl"))
    print("Saved scaler, city encoder, and feature columns in models/.")

    # -----------------------------------------
    # 2. Train traditional ML models
    # -----------------------------------------
    print("\n[Step 2] Training ML models...")
    _ = train_all_sklearn(X_train, y_train, X_val, y_val)

    # Load saved models for evaluation
    lr = joblib.load(os.path.join("models", "linear_regression.pkl"))
    dt = joblib.load(os.path.join("models", "decision_tree.pkl"))
    rf = joblib.load(os.path.join("models", "random_forest.pkl"))
    xgb = joblib.load(os.path.join("models", "xgboost.pkl"))

    # -----------------------------------------
    # 3. Train LSTM model
    # -----------------------------------------
    print("\n[Step 3] Training LSTM model...")
    SEQ_LEN = 14
    X_train_seq, y_train_seq = build_sequences(train_df, feature_cols, scaler, seq_len=SEQ_LEN)
    X_val_seq, y_val_seq = build_sequences(val_df, feature_cols, scaler, seq_len=SEQ_LEN)
    X_test_seq, y_test_seq = build_sequences(test_df, feature_cols, scaler, seq_len=SEQ_LEN)

    print(f"LSTM train sequences: {X_train_seq.shape}")
    print(f"LSTM val sequences  : {X_val_seq.shape}")
    print(f"LSTM test sequences : {X_test_seq.shape}")

    lstm_model = None
    lstm_eval = None
    if len(X_train_seq) > 0 and len(X_val_seq) > 0:
        lstm_model, _, lstm_eval = train_lstm(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=25,
            batch_size=128,
            plot_dir="notebooks",
        )
    else:
        print("Not enough sequence data for LSTM training. Skipping LSTM.")

    # -----------------------------------------
    # 4. Evaluate models on test set
    # -----------------------------------------
    print("\n[Step 4] Evaluating models on test set...")
    all_results = []

    sklearn_models = [
        ("Linear Regression", lr),
        ("Decision Tree", dt),
        ("Random Forest", rf),
        ("XGBoost", xgb),
    ]

    for name, model in sklearn_models:
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 500)
        res = evaluate_model(name, y_test, y_pred, plot_dir="notebooks", save_plots=True)
        category_confusion_matrix(name, y_test, y_pred, plot_dir="notebooks")
        all_results.append(res)

    # Evaluate LSTM
    if lstm_model is not None and len(X_test_seq) > 0:
        lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
        lstm_pred = np.clip(lstm_pred, 0, 500)
        lstm_eval_test = evaluate_model("LSTM", y_test_seq, lstm_pred, plot_dir="notebooks", save_plots=True)
        category_confusion_matrix("LSTM", y_test_seq, lstm_pred, plot_dir="notebooks")
        all_results.append(lstm_eval_test)
    elif lstm_eval is not None:
        # Fallback: include validation metrics if test sequences unavailable
        all_results.append(lstm_eval)

    # -----------------------------------------
    # 5. SHAP explainability
    # -----------------------------------------
    print("\n[Step 5] Generating SHAP explanations...")
    try:
        explain_with_shap(rf, X_test, feature_cols, "Random Forest", plot_dir="notebooks")
    except Exception as e:
        print(f"SHAP for Random Forest failed: {e}")
    try:
        explain_with_shap(xgb, X_test, feature_cols, "XGBoost", plot_dir="notebooks")
    except Exception as e:
        print(f"SHAP for XGBoost failed: {e}")

    # -----------------------------------------
    # 6. 7-day forecast for a highly polluted city
    # -----------------------------------------
    print("\n[Step 6] Creating 7-day AQI forecast...")
    if lstm_model is not None:
        try:
            top_city = df.groupby("City")["AQI"].median().idxmax()
            city_df = df[df["City"] == top_city].sort_values("Date").reset_index(drop=True)

            if len(city_df) >= SEQ_LEN:
                city_features = city_df[feature_cols].fillna(0)
                city_scaled = scaler.transform(city_features)
                last_seq = city_scaled[-SEQ_LEN:]

                aqi_idx = feature_cols.index("AQI_lag_1") if "AQI_lag_1" in feature_cols else 0
                forecast = forecast_next_n_days(
                    lstm_model,
                    last_seq,
                    scaler,
                    feature_cols,
                    n_days=7,
                    aqi_col_idx=aqi_idx,
                )

                print(f"\n7-Day AQI Forecast for {top_city}:")
                for i, val in enumerate(forecast, 1):
                    print(f"  Day {i}: AQI ~ {val:.1f}")

                plot_forecast(
                    city=top_city,
                    historical_aqi=city_df["AQI"].values,
                    forecast_aqi=forecast,
                    historical_dates=city_df["Date"],
                    plot_dir="notebooks",
                )
            else:
                print(f"Not enough data for {top_city} to create forecast.")
        except Exception as e:
            print(f"Forecast step failed: {e}")
    else:
        print("LSTM model not available, skipping forecast.")

    # -----------------------------------------
    # 7. Compare models
    # -----------------------------------------
    print("\n[Step 7] Comparing model performance...")
    if all_results:
        compare_models(all_results, plot_dir="notebooks")

    print("\nPipeline completed successfully!")
    print("Check the 'notebooks/' folder for plots and 'models/' for saved models.")


if __name__ == "__main__":
    main()
