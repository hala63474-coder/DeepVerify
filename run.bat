@echo off
REM ============================================================
REM  DeepVerify - One-click launcher (Windows)
REM  Double-click this file to start the full project.
REM ============================================================
setlocal ENABLEDELAYEDEXPANSION
title DeepVerify | Launcher
color 0B

echo.
echo  ============================================================
echo    DeepVerify  ^|  Six-Signal Deepfake Detection Engine
echo    Irbid National University - DSAI
echo  ============================================================
echo.

cd /d "%~dp0"
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PY=%VENV%\Scripts\python.exe"

REM --- 1. Create virtualenv if missing ------------------------
if not exist "%PY%" (
    echo [setup] Creating Python virtual environment...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [error] Python is not on PATH. Install Python 3.10+ from python.org and try again.
        echo.
        pause
        exit /b 1
    )
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [error] Failed to create virtual environment.
        pause
        exit /b 1
    )
    "%PY%" -m pip install --upgrade pip
    echo [setup] Installing dependencies (this may take a few minutes)...
    "%PY%" -m pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 (
        echo [error] Dependency install failed.
        pause
        exit /b 1
    )
)

REM --- 2. Free port 5000 if already in use --------------------
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    echo [setup] Killing previous server PID %%a on port 5000...
    taskkill /F /PID %%a >nul 2>nul
)

REM --- 3. Open browser shortly after server starts ------------
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start http://127.0.0.1:5000/"

echo.
echo  [run] Starting DeepVerify on http://127.0.0.1:5000/
echo  [run] Press Ctrl+C in this window to stop the server.
echo.

cd /d "%ROOT%app"
"%PY%" app.py

echo.
echo [info] Server stopped.
pause
endlocal
