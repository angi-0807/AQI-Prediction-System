"""
preprocess.py
-------------
Data loading, cleaning, feature engineering, and preprocessing
for the AQI Prediction System.

Class 12 AI Project - Angel Rose

Main steps:
  1. Load the city_day.csv dataset
  2. Handle missing values (interpolation + KNN imputation)
  3. Add AQI category labels
  4. Create time-based and lag/rolling features
  5. Split data in time order (train/val/test)
  6. Scale features using StandardScaler
  7. Build sequences for LSTM model
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------
# Constants used across the project
# ---------------------------------------------

POLLUTANTS = [
    "PM2.5", "PM10", "NO2", "NOx", "NH3",
    "SO2", "CO", "O3", "Benzene", "Toluene",
]

# AQI breakpoints based on Indian CPCB categories
AQI_BREAKPOINTS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]


# ---------------------------------------------
# 1. Loading the dataset
# ---------------------------------------------

def load_data(path: str = "data/city_day.csv") -> pd.DataFrame:
    """Load the AQI dataset and do basic sorting by city and date."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please place city_day.csv in the data/ folder."
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values(["City", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Cities found: {df['City'].nunique()}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


# ---------------------------------------------
# 2. AQI category labeling
# ---------------------------------------------

def aqi_category(aqi: float) -> str:
    """Convert a numeric AQI value into its category name."""
    if pd.isna(aqi):
        return "Unknown"
    aqi = float(np.clip(aqi, 0, 1000))
    for low, high, label in AQI_BREAKPOINTS:
        if low <= aqi <= high:
            return label
    return "Severe"


def add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add an AQI_Category column based on AQI values."""
    df = df.copy()
    if "AQI" in df.columns:
        df["AQI_Category"] = df["AQI"].apply(aqi_category)
    else:
        df["AQI_Category"] = "Unknown"
    return df


# ---------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------------------------

def explore_data(df: pd.DataFrame, plot_dir: str = "notebooks") -> None:
    """Create basic EDA plots and save them in the notebooks folder."""
    os.makedirs(plot_dir, exist_ok=True)

    print("\n--- Missing Values (%) ---")
    missing_pct = df.isnull().mean().sort_values(ascending=False) * 100
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        print("No missing values found.")
    else:
        print(missing_pct.round(2).to_string())

    # Work on a copy for plotting
    plot_df = df.copy()
    if "AQI" not in plot_df.columns:
        print("AQI column missing, skipping EDA plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Median AQI by top polluted cities
    city_medians = (
        plot_df.groupby("City")["AQI"]
        .median()
        .dropna()
        .nlargest(15)
        .sort_values(ascending=True)
    )
    if not city_medians.empty:
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(city_medians)))
        axes[0, 0].barh(city_medians.index, city_medians.values, color=colors)
        axes[0, 0].axvline(200, color="red", linestyle="--", label="Poor threshold (200)")
        axes[0, 0].set_title("Median AQI by City (Top 15)")
        axes[0, 0].set_xlabel("Median AQI")
        axes[0, 0].legend()
    else:
        axes[0, 0].set_title("Median AQI by City (No data)")

    # Plot 2: AQI category distribution
    df_cat = add_aqi_category(plot_df)
    cat_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cat_colors = ["#2ECC71", "#82E0AA", "#F4D03F", "#E67E22", "#E74C3C", "#8E44AD"]
    counts = df_cat["AQI_Category"].value_counts().reindex(cat_order, fill_value=0)
    non_zero = counts[counts > 0]
    if not non_zero.empty:
        axes[0, 1].pie(
            non_zero.values,
            labels=non_zero.index,
            autopct="%1.1f%%",
            colors=[cat_colors[cat_order.index(c)] for c in non_zero.index],
            startangle=140,
        )
    axes[0, 1].set_title("AQI Category Distribution")

    # Plot 3: Monthly average AQI (seasonal pattern)
    monthly = plot_df.copy()
    monthly["Month"] = monthly["Date"].dt.month
    month_avg = monthly.groupby("Month")["AQI"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_vals = [month_avg.get(m, np.nan) for m in range(1, 13)]
    axes[1, 0].plot(month_names, month_vals, marker="o", color="#2196F3", linewidth=2)
    axes[1, 0].fill_between(range(12), 0, np.nan_to_num(month_vals), alpha=0.15, color="#2196F3")
    axes[1, 0].set_title("Average AQI by Month")
    axes[1, 0].set_ylabel("Average AQI")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # Plot 4: Correlation heatmap
    available_cols = [c for c in POLLUTANTS + ["AQI"] if c in plot_df.columns]
    if len(available_cols) > 1:
        corr = plot_df[available_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            linewidths=0.4, ax=axes[1, 1], annot_kws={"size": 7}
        )
        axes[1, 1].set_title("Pollutant Correlation Matrix")
    else:
        axes[1, 1].set_title("Correlation Matrix (Not enough columns)")

    plt.suptitle("India AQI - Exploratory Data Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    eda_path = os.path.join(plot_dir, "eda_overview.png")
    plt.savefig(eda_path, dpi=150)
    plt.close(fig)
    print(f"EDA overview saved to: {eda_path}")

    # Plot 5: City trends for top 4 polluted cities
    top4 = plot_df.groupby("City")["AQI"].median().dropna().nlargest(4).index.tolist()
    if top4:
        fig2, axes2 = plt.subplots(2, 2, figsize=(18, 9))
        axes2 = axes2.flatten()
        for idx, city in enumerate(top4):
            city_df = plot_df[plot_df["City"] == city][["Date", "AQI"]].dropna().copy()
            city_df = city_df.set_index("Date").sort_index()
            monthly_city = city_df["AQI"].resample("ME").mean()
            ax = axes2[idx]
            ax.plot(monthly_city.index, monthly_city.values, color="#E74C3C", linewidth=1.8)
            ax.fill_between(monthly_city.index, monthly_city.values, alpha=0.12, color="#E74C3C")
            ax.axhline(200, color="orange", linestyle="--", linewidth=1, label="Poor (200)")
            ax.axhline(300, color="red", linestyle="--", linewidth=1, label="Very Poor (300)")
            ax.set_title(f"{city} - Monthly AQI Trend")
            ax.set_ylabel("AQI")
            ax.legend(fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for j in range(len(top4), 4):
            axes2[j].axis("off")
        plt.suptitle("AQI Time-Series: Top 4 Most Polluted Cities", fontsize=13)
        plt.tight_layout()
        trend_path = os.path.join(plot_dir, "city_trends.png")
        plt.savefig(trend_path, dpi=150)
        plt.close(fig2)
        print(f"City trends saved to: {trend_path}")


# ---------------------------------------------
# 4. Missing value handling
# ---------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values step by step:
    1. Interpolate values within each city (time-based continuity)
    2. Forward/backward fill remaining gaps within city
    3. Use KNN imputer for any values still missing
    4. Final fallback fill with column median/0
    """
    df = df.copy()
    numeric_cols = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]

    # Interpolate within each city
    for col in numeric_cols:
        df[col] = df.groupby("City")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df[col] = df.groupby("City")[col].transform(lambda s: s.ffill().bfill())

    # KNN imputation for remaining missing values
    if df[numeric_cols].isnull().any().any():
        # KNNImputer can drop columns that are entirely NaN in some cases,
        # so only impute columns that have at least one observed value.
        knn_cols = [c for c in numeric_cols if df[c].notna().any()]
        all_nan_cols = [c for c in numeric_cols if c not in knn_cols]

        for col in all_nan_cols:
            df[col] = 0.0

        if knn_cols and df[knn_cols].isnull().any().any():
            imputer = KNNImputer(n_neighbors=5)
            imputed = imputer.fit_transform(df[knn_cols])
            # Safety check: if shape mismatch occurs, fall back to median fill
            if imputed.shape[1] == len(knn_cols):
                df[knn_cols] = imputed
            else:
                for col in knn_cols:
                    med = df[col].median()
                    df[col] = df[col].fillna(0 if pd.isna(med) else med)

    # Final safety fallback
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median_val) else median_val)

    missing_after = int(df[numeric_cols].isnull().sum().sum())
    print(f"Missing numeric values after imputation: {missing_after}")
    return df


# ---------------------------------------------
# 5. Feature engineering
# ---------------------------------------------

def engineer_features(df: pd.DataFrame, lags: int = 7) -> pd.DataFrame:
    """
    Create useful features for model training:
    - Date features (year, month, day of week, day of year, season)
    - AQI lag features
    - AQI rolling averages
    - Pollutant rolling averages
    """
    df = df.copy().sort_values(["City", "Date"]).reset_index(drop=True)

    # Date-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Season"] = df["Month"].map({
        12: 0, 1: 0, 2: 0,   # Winter
        3: 1, 4: 1, 5: 1,    # Spring/Summer
        6: 2, 7: 2, 8: 2,    # Monsoon
        9: 3, 10: 3, 11: 3,  # Autumn/Post-monsoon
    }).fillna(0).astype(int)

    # AQI lag features
    if "AQI" in df.columns:
        for lag in range(1, lags + 1):
            df[f"AQI_lag_{lag}"] = df.groupby("City")["AQI"].shift(lag)

        # Rolling averages for AQI
        for window in [3, 7, 14]:
            df[f"AQI_roll_{window}"] = (
                df.groupby("City")["AQI"]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    # Rolling average for pollutant values (3-day)
    for pol in [c for c in POLLUTANTS if c in df.columns]:
        df[f"{pol}_roll3"] = (
            df.groupby("City")[pol]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

    # Remove rows where lag features could not be created
    lag_cols = [f"AQI_lag_{i}" for i in range(1, lags + 1) if f"AQI_lag_{i}" in df.columns]
    if lag_cols:
        df.dropna(subset=lag_cols, inplace=True)

    df.reset_index(drop=True, inplace=True)
    print(f"Features after engineering: {df.shape[1]} columns, {df.shape[0]:,} rows")
    return df


# ---------------------------------------------
# 6. Encoding and splitting
# ---------------------------------------------

EXCLUDE_COLS = ["Date", "AQI", "AQI_Bucket", "AQI_Category"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get model feature columns (excluding target/date/text label columns)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != "City"]


def encode_city(df: pd.DataFrame):
    """Label encode city names into numeric form for ML models."""
    le = LabelEncoder()
    df = df.copy()
    df["City_enc"] = le.fit_transform(df["City"].astype(str))
    return df, le


def time_aware_split(df: pd.DataFrame, val_size: float = 0.10, test_size: float = 0.15):
    """
    Split data in chronological order to avoid data leakage.
    Order: train -> validation -> test
    """
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    # Make sure splits are not empty
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n - 1) if n > 2 else n

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # If test is empty (very small dataset), borrow from val
    if test_df.empty and not val_df.empty:
        test_df = val_df.iloc[-1:].copy()
        val_df = val_df.iloc[:-1].copy()

    print("\n--- Time-aware Split ---")
    if not train_df.empty:
        print(f"Train: {len(train_df):,} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    if not val_df.empty:
        print(f"Val  : {len(val_df):,} rows ({val_df['Date'].min().date()} to {val_df['Date'].max().date()})")
    if not test_df.empty:
        print(f"Test : {len(test_df):,} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    return train_df, val_df, test_df


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    """Scale features using StandardScaler fitted only on training data."""
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
    X_val = scaler.transform(val_df[feature_cols].fillna(0)) if not val_df.empty else np.empty((0, len(feature_cols)))
    X_test = scaler.transform(test_df[feature_cols].fillna(0)) if not test_df.empty else np.empty((0, len(feature_cols)))

    y_train = train_df["AQI"].values.astype(float)
    y_val = val_df["AQI"].values.astype(float) if not val_df.empty else np.array([])
    y_test = test_df["AQI"].values.astype(float) if not test_df.empty else np.array([])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ---------------------------------------------
# 7. LSTM sequence builder
# ---------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    seq_len: int = 14,
    city: str = None,
):
    """
    Build 3D sequences for LSTM: (samples, timesteps, features).
    Sequences are created separately for each city to keep time order correct.
    """
    if df.empty:
        return np.empty((0, seq_len, len(feature_cols))), np.array([])

    cities = [city] if city else df["City"].dropna().unique().tolist()
    X_list, y_list = [], []

    for c in cities:
        city_df = df[df["City"] == c].sort_values("Date").reset_index(drop=True)
        if len(city_df) < seq_len + 1:
            continue

        # Make sure all feature columns exist
        for col in feature_cols:
            if col not in city_df.columns:
                city_df[col] = 0.0

        X_scaled = scaler.transform(city_df[feature_cols].fillna(0))
        y_vals = city_df["AQI"].values.astype(float)

        for i in range(seq_len, len(city_df)):
            X_list.append(X_scaled[i - seq_len:i])
            y_list.append(y_vals[i])

    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), np.array([])

    return np.array(X_list, dtype=float), np.array(y_list, dtype=float)


# ---------------------------------------------
# 8. Full preprocessing pipeline
# ---------------------------------------------

def run_preprocessing(data_path: str = "data/city_day.csv", run_eda: bool = True, lags: int = 7):
    """
    Run the complete preprocessing pipeline.
    Returns a dictionary with processed data, splits, scaler, and encoders.
    """
    df_raw = load_data(data_path)

    if run_eda:
        explore_data(df_raw)

    df = handle_missing_values(df_raw)
    df = add_aqi_category(df)
    df = engineer_features(df, lags=lags)
    df, city_encoder = encode_city(df)

    feature_cols = get_feature_columns(df)
    if "City_enc" in df.columns and "City_enc" not in feature_cols:
        feature_cols = ["City_enc"] + feature_cols

    # Final cleanup - no NaNs/inf in features or target
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.dropna(subset=["AQI"]).copy()
    df["AQI"] = df["AQI"].clip(0, 500)

    train_df, val_df, test_df = time_aware_split(df)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    print("\nPreprocessing completed successfully.")
    print(f"Feature count: {len(feature_cols)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape  : {X_val.shape}")
    print(f"X_test shape : {X_test.shape}")

    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "city_encoder": city_encoder,
    }
