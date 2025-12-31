@echo off
REM webapp/run_webapp.bat
REM Script to run the RAPTOR vs GraphRAG Web Application

echo ============================================================
echo RAPTOR vs GraphRAG Comparison Web Application
echo ============================================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Navigate to webapp directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "..\venv\Scripts\activate.bat"
) else (
    echo No virtual environment found, using system Python
)

REM Install required packages if needed
echo.
echo Checking dependencies...
pip install flask flask-cors sentence-transformers numpy scikit-learn -q 2>nul

REM Set Flask environment variables
set FLASK_APP=app.py
set FLASK_ENV=development
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password123

echo.
echo Starting server...
echo.
echo ============================================================
echo Web Application running at: http://localhost:5000
echo ============================================================
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the application
python app.py

pause
