@echo off
REM Week 5: The Final Showdown - RAPTOR vs GraphRAG
REM Run this script to generate all comparison materials

echo ========================================
echo WEEK 5: THE FINAL SHOWDOWN
echo RAPTOR vs GraphRAG Comparison
echo ========================================
echo.

cd /d "D:\Raptor\graphrag_project"

REM Activate virtual environment if it exists
if exist "..\raptor_env\Scripts\activate.bat" (
    call ..\raptor_env\Scripts\activate.bat
    echo Virtual environment activated.
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo No virtual environment found, using system Python.
)

echo.
echo Starting Week 5 comparison analysis...
echo.

python week5_comparison\main.py

echo.
echo ========================================
echo Week 5 Complete!
echo ========================================
echo.
echo Check the outputs folder for:
echo   - comparison/    : Comparison reports
echo   - paper/         : Research paper content
echo   - visualizations/: Charts and graphs
echo   - presentation/  : Slide deck
echo.

pause
