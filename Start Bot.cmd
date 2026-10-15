@echo off
setlocal
title SOL Market Brief
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dashboard.ps1" -Action Start
if errorlevel 1 (
    echo.
    echo The dashboard could not start. See the message above.
    pause
    exit /b 1
)