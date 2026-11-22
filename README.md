# Air Quality Index (AQI) Prediction System

**Made by Angel Rose**  
*Class 12 AI Project*

---

## Project Overview

This project is an **Air Quality Index (AQI) Prediction System** built using Python and Machine Learning.  
It uses historical air pollution data from Indian cities to:

- analyze AQI trends,
- predict AQI from pollutant values,
- compare different ML models,
- and forecast AQI for the next 7 days using an LSTM model.

The project also includes a **Streamlit dashboard** for interactive visualization and prediction.

---

## Features

- **Data preprocessing** (missing value handling, feature engineering, scaling)
- **Exploratory Data Analysis (EDA)** with saved plots
- **Multiple ML models**:
  - Linear Regression (Ridge)
  - Decision Tree
  - Random Forest
  - XGBoost
- **Deep Learning model**:
  - LSTM (for time-series AQI forecasting)
- **Model evaluation**:
  - RMSE, MAE, R²
  - AQI category accuracy
  - confusion matrix by AQI category
  - SHAP explainability plots (for tree models)
- **Streamlit dashboard**:
  - City dashboard
  - Manual AQI prediction
  - India-wide comparison charts
  - 7-day AQI forecast

---

## Dataset

The project uses the **India Air Quality dataset** (`city_day.csv`) containing daily pollutant and AQI values for multiple cities.

Expected file location:

```text
data/city_day.csv
```

### Common columns in the dataset
- `City`
- `Date`
- `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`
- `AQI`
- `AQI_Bucket`

> Note: The project mainly uses AQI and pollutant columns. Some columns may be missing in certain rows, and the preprocessing script handles that.

---

## Project Structure

```text
Air-Quality-Index-AQI-Prediction-System/
│
├── app.py                  # Streamlit dashboard
├── main.py                 # Runs full training + evaluation pipeline
├── preprocess.py           # Data cleaning, EDA, feature engineering
├── train.py                # ML and LSTM model training
├── evaluate.py             # Metrics, plots, SHAP, comparison charts
├── requirements.txt        # Python dependencies
├── start.bat               # Windows startup script
├── LICENSE                 # MIT License
├── README.md               # Project documentation
├── PROJECT_DOCUMENTATION.pdf  # Detailed project report/documentation
│
├── data/
│   └── city_day.csv        # Dataset
│
├── models/                 # Saved models/scalers/encoders (generated after training)
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lstm_best.keras
│   ├── scaler.pkl
│   ├── city_encoder.pkl
│   └── feature_cols.pkl
│
└── notebooks/              # EDA plots and evaluation outputs (generated)
    ├── eda_overview.png
    ├── city_trends.png
    ├── lstm_training.png
    ├── model_comparison.png
    └── ... other plots
```

---

## Installation (Windows / Linux / macOS)

### 1. Clone or download the project
```bash
git clone <your-repo-link>
cd Air-Quality-Index-AQI-Prediction-System
```

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bat
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run the Project

### Option A: Run full pipeline (training + evaluation + plots)
```bash
python main.py
```

This will:
- preprocess the dataset,
- train all models,
- save models in `models/`,
- save graphs in `notebooks/`.

### Option B: Run the Streamlit dashboard
After training (or if models already exist):

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

### Option C: Windows one-click start
Double-click `start.bat` (or run it from CMD).  
It will:
- create/activate venv,
- install requirements,
- train models if missing,
- launch the Streamlit app.

---

## Models Used and What They Do

### 1. Linear Regression (Ridge)
- A simple baseline regression model.
- Good for understanding basic relationships between pollutants and AQI.

### 2. Decision Tree Regressor
- Splits data into decision rules.
- Easy to interpret but can overfit.

### 3. Random Forest Regressor
- Ensemble of many decision trees.
- Usually gives better accuracy and stability than a single tree.

### 4. XGBoost Regressor
- Gradient boosting model.
- Often performs very well on tabular data like this dataset.

### 5. LSTM (Long Short-Term Memory)
- A deep learning model for time-series data.
- Used here to forecast AQI for the next 7 days based on recent sequence patterns.

---

## Evaluation Metrics

The project evaluates models using:

- **RMSE (Root Mean Squared Error)**: lower is better
- **MAE (Mean Absolute Error)**: lower is better
- **R² Score**: higher is better (closer to 1)
- **Category Accuracy**: checks if predicted AQI falls in the same AQI category (Good, Moderate, Poor, etc.)

---

## AQI Categories (CPCB)

| AQI Range | Category |
|-----------|----------|
| 0 - 50    | Good |
| 51 - 100  | Satisfactory |
| 101 - 200 | Moderate |
| 201 - 300 | Poor |
| 301 - 400 | Very Poor |
| 401 - 500 | Severe |

---

## Output Files Generated

After running `python main.py`, you should see:

- `models/*.pkl` and `models/lstm_best.keras`
- `models/scaler.pkl`, `models/feature_cols.pkl`, `models/city_encoder.pkl`
- `notebooks/eda_overview.png`
- `notebooks/city_trends.png`
- `notebooks/lstm_training.png`
- `notebooks/model_comparison.png`
- prediction/residual/confusion matrix/SHAP plots in `notebooks/`

---

## Notes / Limitations

- Forecast accuracy depends on data quality and missing values in the dataset.
- LSTM forecasting in this project uses an iterative approach and is meant for educational/demo purposes.
- The dashboard prediction tab uses a simplified feature-row input; for best results, train and use the full pipeline.

---

## Future Improvements

- Add live AQI API integration
- Add city-wise model tuning
- Improve LSTM multi-feature forecasting
- Deploy dashboard online (Streamlit Cloud / Render)
- Add user login and report export

---

## Acknowledgements

- CPCB / Kaggle India Air Quality dataset sources
- Open-source Python libraries: Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Streamlit, Plotly, SHAP

---

## Author

**Angel Rose**  
Class 12 AI Project  
(2026)
