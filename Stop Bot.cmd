@echo off
setlocal
title Stop SOL Market Brief
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dashboard.ps1" -Action Stop
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)