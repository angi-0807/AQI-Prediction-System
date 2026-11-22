@echo off
REM ===============================================
REM AQI Prediction System - Startup Script
REM Class 12 AI Project by Angel Rose
REM ===============================================

title AQI Prediction System - Angel Rose

echo.
echo ===============================================
echo   AQI Prediction System
echo   Class 12 AI Project by Angel Rose
echo ===============================================
echo.

REM Move to the folder where this .bat file is located
cd /d "%~dp0"

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

REM Create virtual environment if it does not exist
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing/checking dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM Train models if key model files are missing
if not exist "models\random_forest.pkl" (
    echo [INFO] Model files not found. Running training pipeline...
    python main.py
    if errorlevel 1 (
        echo [ERROR] Training pipeline failed. Please check errors above.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Existing trained models found. Skipping training.
)

REM Start Streamlit app
echo.
echo [INFO] Starting Streamlit dashboard...
echo Open the URL shown below in your browser (usually http://localhost:8501)
echo.
streamlit run app.py

if errorlevel 1 (
    echo [ERROR] Streamlit failed to start.
    pause
    exit /b 1
)

pause
