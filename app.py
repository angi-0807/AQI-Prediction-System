"""
app.py
------
Streamlit dashboard for the AQI Prediction System.

Class 12 AI Project - Angel Rose

Run:
    streamlit run app.py

Features:
  - City-wise AQI dashboard with latest values and trends
  - Manual AQI prediction using trained ML models
  - All-India city comparison charts
  - 7-day AQI forecast using LSTM model
"""

import os
import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# TensorFlow is only needed when LSTM model exists
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    keras = None
    TF_AVAILABLE = False


# ---------------------------------------------
# Page config and styling
# ---------------------------------------------
st.set_page_config(
    page_title="AQI Prediction Dashboard - Angel Rose",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .aqi-good       {background:#E8F5E9;border-left:5px solid #2ECC71;padding:12px;border-radius:6px;color:#1B5E20;font-size:16px;}
      .aqi-satisfac   {background:#F1F8E9;border-left:5px solid #8BC34A;padding:12px;border-radius:6px;color:#33691E;font-size:16px;}
      .aqi-moderate   {background:#FFFDE7;border-left:5px solid #F4D03F;padding:12px;border-radius:6px;color:#7D6608;font-size:16px;}
      .aqi-poor       {background:#FFF3E0;border-left:5px solid #E67E22;padding:12px;border-radius:6px;color:#7E5109;font-size:16px;}
      .aqi-verypoor   {background:#FFEBEE;border-left:5px solid #E74C3C;padding:12px;border-radius:6px;color:#7B241C;font-size:16px;}
      .aqi-severe     {background:#F3E5F5;border-left:5px solid #8E44AD;padding:12px;border-radius:6px;color:#4A235A;font-size:16px;}
      .alert-hazard   {background:#B71C1C;color:white;padding:14px;border-radius:6px;font-size:18px;text-align:center;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------
# Constants / helper functions
# ---------------------------------------------
AQI_BREAKPOINTS = [
    (50, "Good", "aqi-good", "#2ECC71"),
    (100, "Satisfactory", "aqi-satisfac", "#8BC34A"),
    (200, "Moderate", "aqi-moderate", "#F4D03F"),
    (300, "Poor", "aqi-poor", "#E67E22"),
    (400, "Very Poor", "aqi-verypoor", "#E74C3C"),
    (500, "Severe", "aqi-severe", "#8E44AD"),
]

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def aqi_category_info(aqi: float):
    """Return AQI category label, CSS class, and color."""
    try:
        aqi_val = float(aqi)
    except Exception:
        aqi_val = 0.0
    aqi_val = float(np.clip(aqi_val, 0, 500))
    for limit, label, css, color in AQI_BREAKPOINTS:
        if aqi_val <= limit:
            return label, css, color
    return "Severe", "aqi-severe", "#8E44AD"


# ---------------------------------------------
# Loading data and models
# ---------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_artefacts():
    model_dir = "models"
    artefacts = {}

    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    for name, fname in model_files.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            try:
                artefacts[name] = joblib.load(path)
            except Exception as e:
                artefacts[f"{name}_error"] = str(e)

    # Load LSTM model if available
    lstm_path = os.path.join(model_dir, "lstm_best.keras")
    if os.path.exists(lstm_path) and TF_AVAILABLE:
        try:
            artefacts["LSTM"] = keras.models.load_model(lstm_path)
        except Exception as e:
            artefacts["LSTM_error"] = str(e)

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feat_path = os.path.join(model_dir, "feature_cols.pkl")
    enc_path = os.path.join(model_dir, "city_encoder.pkl")

    artefacts["scaler"] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    artefacts["feature_cols"] = joblib.load(feat_path) if os.path.exists(feat_path) else None
    artefacts["city_encoder"] = joblib.load(enc_path) if os.path.exists(enc_path) else None

    return artefacts


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    path = os.path.join("data", "city_day.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


# ---------------------------------------------
# Plot helpers
# ---------------------------------------------

def aqi_gauge(aqi_value: float, title: str = "Predicted AQI"):
    aqi_value = float(np.clip(aqi_value, 0, 500))
    cat, _, color = aqi_category_info(aqi_value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(aqi_value, 1),
        title={"text": f"{title}<br><span style='font-size:14px;color:{color}'>{cat}</span>"},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#E8F5E9"},
                {"range": [50, 100], "color": "#F1F8E9"},
                {"range": [100, 200], "color": "#FFFDE7"},
                {"range": [200, 300], "color": "#FFF3E0"},
                {"range": [300, 400], "color": "#FFEBEE"},
                {"range": [400, 500], "color": "#F3E5F5"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
    return fig


# ---------------------------------------------
# Sidebar
# ---------------------------------------------

def sidebar(df):
    st.sidebar.title("AQI Dashboard")
    st.sidebar.caption("Class 12 AI Project by Angel Rose")

    cities = sorted(df["City"].dropna().unique().tolist()) if df is not None and "City" in df.columns else ["Delhi"]
    default_city_idx = cities.index("Delhi") if "Delhi" in cities else 0
    city = st.sidebar.selectbox("Select City", cities, index=default_city_idx)

    model_names = ["XGBoost", "Random Forest", "Linear Regression", "Decision Tree", "LSTM"]
    model_name = st.sidebar.selectbox("Select Model (for reference)", model_names, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### AQI Categories")
    for limit, label, _, color in AQI_BREAKPOINTS:
        st.sidebar.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
            f"border-radius:2px;margin-right:6px;'></span>{label} (<= {limit})",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.info("Tip: Run `python main.py` first to train models and generate plots.")
    return city, model_name


# ---------------------------------------------
# Tab 1: City dashboard
# ---------------------------------------------

def tab_city_dashboard(df, city, artefacts):
    if df is None:
        st.warning("Dataset not found. Please place `city_day.csv` inside the `data/` folder.")
        return

    city_df = df[df["City"] == city].sort_values("Date").reset_index(drop=True).copy()
    if city_df.empty:
        st.error(f"No data found for {city}.")
        return

    if "AQI" in city_df.columns:
        city_df["AQI"] = pd.to_numeric(city_df["AQI"], errors="coerce").interpolate(limit_direction="both")

    latest_valid = city_df.dropna(subset=["AQI"])
    if latest_valid.empty:
        st.error(f"AQI data is not available for {city}.")
        return

    latest = latest_valid.iloc[-1]
    latest_aqi = float(latest["AQI"])
    cat, css_class, _ = aqi_category_info(latest_aqi)

    if latest_aqi > 300:
        st.markdown(
            f'<div class="alert-hazard">AIR QUALITY ALERT - {city}: {cat} ({latest_aqi:.0f})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="{css_class}"><b>{city}</b> - Latest AQI: <b>{latest_aqi:.0f}</b> ({cat}) '
            f'on {pd.to_datetime(latest["Date"]).date()}</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # Latest pollutant values
    cols = st.columns(len(POLLUTANTS))
    for col, pol in zip(cols, POLLUTANTS):
        val = pd.to_numeric(latest.get(pol, np.nan), errors="coerce")
        col.metric(pol, f"{val:.1f}" if pd.notna(val) else "N/A")

    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(aqi_gauge(latest_aqi, f"{city} - Latest AQI"), use_container_width=True)

    with c2:
        trend_df = city_df[["Date", "AQI"]].dropna().copy()
        trend_df["Date"] = pd.to_datetime(trend_df["Date"])
        weekly = trend_df.set_index("Date")["AQI"].resample("W").mean().reset_index()
        fig_trend = px.line(weekly, x="Date", y="AQI", title=f"Weekly Average AQI Trend - {city}")
        fig_trend.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="Poor (200)")
        fig_trend.add_hline(y=300, line_dash="dash", line_color="red", annotation_text="Very Poor (300)")
        fig_trend.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    # Pollutant bar chart
    pol_vals = {}
    for p in POLLUTANTS:
        v = pd.to_numeric(latest.get(p, np.nan), errors="coerce")
        pol_vals[p] = float(v) if pd.notna(v) else 0.0
    fig_bar = px.bar(
        x=list(pol_vals.keys()),
        y=list(pol_vals.values()),
        color=list(pol_vals.values()),
        color_continuous_scale="RdYlGn_r",
        labels={"x": "Pollutant", "y": "Concentration"},
        title=f"Latest Pollutant Levels - {city}",
    )
    fig_bar.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Year-month heatmap
    st.markdown("#### Year vs Month AQI Heatmap")
    hm = city_df[["Date", "AQI"]].dropna().copy()
    hm["Year"] = pd.to_datetime(hm["Date"]).dt.year
    hm["Month"] = pd.to_datetime(hm["Date"]).dt.month
    pivot = hm.pivot_table(index="Year", columns="Month", values="AQI", aggfunc="mean")
    if not pivot.empty:
        pivot = pivot.reindex(columns=range(1, 13))
        pivot.columns = [MONTH_NAMES[m] for m in pivot.columns]
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            title=f"Average AQI Heatmap - {city}",
        )
        fig_heat.update_layout(height=330, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data to display heatmap.")


# ---------------------------------------------
# Tab 2: Manual prediction
# ---------------------------------------------

def tab_predict(artefacts):
    st.markdown("### Predict AQI from Pollutant Values")
    st.caption("Enter pollutant values and choose a trained model to predict AQI.")

    available_models = [m for m in ["XGBoost", "Random Forest", "Linear Regression", "Decision Tree"] if m in artefacts]
    if not available_models:
        st.warning("No trained ML models found. Please run `python main.py` first.")
        return

    model_name = st.selectbox("Select Model", available_models, index=0)
    model = artefacts.get(model_name)
    scaler = artefacts.get("scaler")
    feature_cols = artefacts.get("feature_cols")

    if model is None or scaler is None or feature_cols is None:
        st.warning("Model/scaler/feature files are missing. Please run `python main.py` to generate them.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.number_input("PM2.5 (ug/m3)", 0.0, 1000.0, 85.0, 1.0)
        pm10 = st.number_input("PM10 (ug/m3)", 0.0, 1000.0, 110.0, 1.0)
        no2 = st.number_input("NO2 (ug/m3)", 0.0, 500.0, 35.0, 1.0)
    with col2:
        nox = st.number_input("NOx (ug/m3)", 0.0, 500.0, 55.0, 1.0)
        nh3 = st.number_input("NH3 (ug/m3)", 0.0, 500.0, 8.0, 0.5)
        co = st.number_input("CO (mg/m3)", 0.0, 50.0, 1.5, 0.1)
    with col3:
        so2 = st.number_input("SO2 (ug/m3)", 0.0, 500.0, 10.0, 0.5)
        o3 = st.number_input("O3 (ug/m3)", 0.0, 500.0, 40.0, 1.0)
        benzene = st.number_input("Benzene (ug/m3)", 0.0, 100.0, 1.0, 0.1)
        toluene = st.number_input("Toluene (ug/m3)", 0.0, 100.0, 2.0, 0.1)

    month = st.slider("Month", 1, 12, datetime.date.today().month)

    if st.button("Predict AQI", use_container_width=True):
        # Build feature row matching training feature columns
        row = {fc: 0.0 for fc in feature_cols}
        inputs = {
            "PM2.5": pm25, "PM10": pm10, "NO2": no2, "NOx": nox, "NH3": nh3,
            "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene, "Toluene": toluene,
            "Month": month,
        }
        for k, v in inputs.items():
            if k in row:
                row[k] = float(v)

        X = pd.DataFrame([row], columns=feature_cols).fillna(0)
        try:
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled)[0])
            pred = float(np.clip(pred, 0, 500))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        cat, css, _ = aqi_category_info(pred)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(aqi_gauge(pred, f"Predicted AQI ({model_name})"), use_container_width=True)
            st.markdown(f'<div class="{css}">Category: <b>{cat}</b></div>', unsafe_allow_html=True)
        with c2:
            advice = {
                "Good": "Air quality is good. Outdoor activities are safe.",
                "Satisfactory": "Air quality is acceptable for most people.",
                "Moderate": "Sensitive people should reduce long outdoor exposure.",
                "Poor": "Limit outdoor activities, especially for children and elderly.",
                "Very Poor": "Avoid outdoor exertion. Use masks if needed.",
                "Severe": "Health emergency conditions. Stay indoors if possible.",
            }
            st.info(advice.get(cat, "Please monitor AQI updates."))
            st.metric("Predicted AQI", f"{pred:.1f}")
            st.metric("Category", cat)
            st.caption("Note: This is a model prediction for learning/demo purposes.")


# ---------------------------------------------
# Tab 3: All-India heatmap / comparison
# ---------------------------------------------

def tab_heatmap(df):
    if df is None:
        st.warning("Dataset not found. Please add `data/city_day.csv`.")
        return

    st.markdown("### All-India AQI Comparison")
    work_df = df.copy()
    work_df["Date"] = pd.to_datetime(work_df["Date"], errors="coerce")
    work_df["AQI"] = pd.to_numeric(work_df["AQI"], errors="coerce")
    work_df = work_df.dropna(subset=["Date", "City"])

    if work_df.empty:
        st.warning("No usable data found in dataset.")
        return

    min_year = int(work_df["Date"].dt.year.min())
    max_year = int(work_df["Date"].dt.year.max())
    default_year = min(2019, max_year)
    year = st.slider("Select Year", min_year, max_year, default_year)

    year_df = work_df[work_df["Date"].dt.year == year].copy()
    if year_df.empty:
        st.info("No data available for the selected year.")
        return

    city_aqi = year_df.groupby("City", dropna=True)["AQI"].mean().reset_index().dropna()
    city_aqi = city_aqi.sort_values("AQI", ascending=False)

    top_n = st.slider("Number of cities to show", 5, 25, 15)
    top_city_df = city_aqi.head(top_n).sort_values("AQI", ascending=True)

    fig_bar = px.bar(
        top_city_df,
        x="AQI",
        y="City",
        orientation="h",
        color="AQI",
        color_continuous_scale="RdYlGn_r",
        title=f"Average AQI by City ({year})",
        labels={"AQI": "Average AQI"},
    )
    fig_bar.update_layout(height=550, coloraxis_showscale=False, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Monthly Trend (Top 8 Cities)")
    top_cities = city_aqi.head(8)["City"].tolist()
    trend_df = year_df[year_df["City"].isin(top_cities)].copy()
    trend_df["Month"] = trend_df["Date"].dt.month
    month_city = trend_df.groupby(["City", "Month"], as_index=False)["AQI"].mean()
    month_city["Month_Name"] = month_city["Month"].map(MONTH_NAMES)

    fig_line = px.line(
        month_city,
        x="Month_Name",
        y="AQI",
        color="City",
        category_orders={"Month_Name": list(MONTH_NAMES.values())},
        title=f"Monthly Average AQI - Top 8 Cities ({year})",
    )
    fig_line.update_layout(height=420, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_line, use_container_width=True)


# ---------------------------------------------
# Tab 4: 7-day forecast (LSTM)
# ---------------------------------------------

def tab_forecast(df, city, artefacts):
    st.markdown(f"### 7-Day AQI Forecast - {city}")

    if df is None:
        st.warning("Dataset not found. Please add `data/city_day.csv`.")
        return

    lstm = artefacts.get("LSTM")
    scaler = artefacts.get("scaler")
    feature_cols = artefacts.get("feature_cols")

    if lstm is None:
        st.warning("LSTM model not found. Run `python main.py` to train and save the model.")
        return
    if scaler is None or feature_cols is None:
        st.warning("Missing scaler or feature columns. Run `python main.py` first.")
        return

    try:
        from preprocess import handle_missing_values, add_aqi_category, engineer_features, encode_city
        from train import forecast_next_n_days
    except Exception as e:
        st.error(f"Could not import required modules for forecasting: {e}")
        return

    city_df = df[df["City"] == city].sort_values("Date").reset_index(drop=True).copy()
    if len(city_df) < 30:
        st.warning("Not enough data for this city to generate a reliable forecast.")
        return

    with st.spinner("Preparing data and generating forecast..."):
        try:
            city_df_clean = handle_missing_values(city_df)
            city_df_clean = add_aqi_category(city_df_clean)
            city_df_feat = engineer_features(city_df_clean, lags=7)
            city_df_feat, _ = encode_city(city_df_feat)

            # Ensure all required feature columns exist
            for col in feature_cols:
                if col not in city_df_feat.columns:
                    city_df_feat[col] = 0.0
            city_df_feat[feature_cols] = city_df_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            city_scaled = scaler.transform(city_df_feat[feature_cols])
            seq_len = int(lstm.input_shape[1]) if lstm.input_shape and lstm.input_shape[1] else 14
            if len(city_scaled) < seq_len:
                st.warning(f"Not enough rows after preprocessing. Need at least {seq_len} rows.")
                return

            last_seq = city_scaled[-seq_len:]
            aqi_idx = feature_cols.index("AQI_lag_1") if "AQI_lag_1" in feature_cols else 0
            forecast = forecast_next_n_days(
                lstm, last_seq, scaler, feature_cols, n_days=7, aqi_col_idx=aqi_idx
            )
        except Exception as e:
            st.error(f"Forecast generation failed: {e}")
            return

    last_date = pd.to_datetime(city_df_clean["Date"]).max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq="D")
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_AQI": np.round(forecast, 1),
        "Category": [aqi_category_info(v)[0] for v in forecast],
    })

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        max_forecast = float(np.max(forecast))
        cat, css, _ = aqi_category_info(max_forecast)
        st.markdown(f'<div class="{css}">Peak Forecast: <b>{max_forecast:.1f}</b> ({cat})</div>', unsafe_allow_html=True)

    with c2:
        hist = city_df_clean[["Date", "AQI"]].dropna().tail(60).copy()
        hist["Date"] = pd.to_datetime(hist["Date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["AQI"], mode="lines", name="Historical AQI",
            line=dict(color="#1565C0", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Predicted_AQI"], mode="lines+markers", name="7-Day Forecast",
            line=dict(color="#E53935", width=2, dash="dot"), marker=dict(size=8)
        ))
        fig.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="Poor (200)")
        fig.add_hline(y=300, line_dash="dash", line_color="red", annotation_text="Very Poor (300)")
        fig.update_layout(
            title=f"Historical AQI + 7-Day Forecast - {city}",
            xaxis_title="Date",
            yaxis_title="AQI",
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Forecast is generated using the trained LSTM model and recent city data.")


# ---------------------------------------------
# Main app
# ---------------------------------------------

def main():
    st.title("Air Quality Index (AQI) Prediction System")
    st.caption("Class 12 AI Project | Made by Angel Rose")

    df = load_dataset()
    artefacts = load_artefacts()

    # Helpful status messages
    missing_models = [m for m in ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"] if m not in artefacts]
    if missing_models:
        st.sidebar.warning("Some ML models are missing. Run `python main.py` to train them.")
    if "LSTM" not in artefacts:
        st.sidebar.warning("LSTM model not loaded/found. Forecast tab may not work until training is complete.")

    city, _ = sidebar(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "City Dashboard",
        "Predict AQI",
        "India Comparison",
        "7-Day Forecast",
    ])

    with tab1:
        tab_city_dashboard(df, city, artefacts)
    with tab2:
        tab_predict(artefacts)
    with tab3:
        tab_heatmap(df)
    with tab4:
        tab_forecast(df, city, artefacts)


if __name__ == "__main__":
    main()
