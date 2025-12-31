@echo off
REM =====================================================
REM Run Week 3: Knowledge Graph Construction in Neo4j
REM =====================================================

cd /d "D:\Raptor\graphrag_project"

echo ========================================
echo Starting Week 3: Graph Construction
echo ========================================
echo.

REM Check if Neo4j is accessible
echo Checking Neo4j connection...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "..\raptor_env\Scripts\activate.bat" (
    call ..\raptor_env\Scripts\activate.bat
)

REM Run the main script
python week3_graph_construction\main.py

echo.
echo ========================================
echo Week 3 Complete!
echo ========================================

pause
