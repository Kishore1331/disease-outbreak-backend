@echo off
title Epidemiology — Municipal Disease Surveillance Platform
echo =====================================================================
echo  Epidemiology — Municipal Disease Surveillance Platform
echo  Ward-Level Outbreak Early Warning & Predictive Modeling
echo  Coimbatore Municipal Corporation Public Health Division
echo =====================================================================
echo.

set PATH=C:\Program Files\nodejs;%PATH%

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Epidemiology Backend (FastAPI)" cmd /k "cd /d %~dp0 && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

timeout /t 2 /nobreak >nul

echo [2/2] Starting React Frontend on http://localhost:5173 ...
start "Epidemiology Frontend (Vite)" cmd /k "cd /d %~dp0frontend && npm.cmd run dev -- --host 127.0.0.1 --port 5173"

echo.
echo =====================================================================
echo  Platform is LIVE!
echo    - Frontend Dashboard: http://localhost:5173
echo    - Backend API Docs:   http://127.0.0.1:8000/docs
echo =====================================================================
echo.
timeout /t 2 >nul
start http://localhost:5173
