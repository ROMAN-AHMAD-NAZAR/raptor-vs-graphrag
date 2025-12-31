@echo off
REM =====================================================
REM Run Week 4: Graph-Based Retrieval (GraphRAG Core)
REM =====================================================

cd /d "D:\Raptor\graphrag_project"

echo ========================================
echo Starting Week 4: Graph-Based Retrieval
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "..\raptor_env\Scripts\activate.bat" (
    call ..\raptor_env\Scripts\activate.bat
)

echo Checking dependencies...
echo.

REM Check if Neo4j is running (optional - just informational)
echo Make sure Neo4j Desktop is running with your database started!
echo.

REM Run the main script
python week4_graph_retrieval\main.py

echo.
echo ========================================
echo Week 4 Complete!
echo ========================================

pause
