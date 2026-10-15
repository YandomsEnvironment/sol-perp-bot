@echo off
rem Auto-start the SOL-PERP bot's background systems (recorder + dashboard).
rem Placed in the user's Startup folder so both survive reboots/logons.
cd /d C:\Users\lumin\Desktop\sol-perp-bot

rem avoid double-starting if already running
tasklist /FI "IMAGENAME eq python.exe" | find /I "python.exe" >nul
wmic process where "CommandLine like '%%recorder.py%%'" get ProcessId 2>nul | findstr /r "[0-9]" >nul
if not errorlevel 1 (
    echo recorder already running
) else (
    start "" /min ".venv\Scripts\python.exe" "data\recorder.py" ^> logs\recorder.out 2^> logs\recorder.err
)

powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://localhost:8501/api/health' -UseBasicParsing -TimeoutSec 3).Content | Out-Null; Write-Host 'dashboard already running' } catch { Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'live/dashboard.py','--host','127.0.0.1','--port','8501' -WorkingDirectory 'C:\Users\lumin\Desktop\sol-perp-bot' -WindowStyle Hidden }"
