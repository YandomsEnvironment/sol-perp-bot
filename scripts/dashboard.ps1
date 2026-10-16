param(
    [ValidateSet('Start', 'Stop')][string]$Action = 'Start',
    [ValidateRange(1024, 65535)][int]$Port = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venv\Scripts\python.exe'
$dashboard = Join-Path $root 'live\dashboard.py'
$logs = Join-Path $root 'logs'
$url = "http://127.0.0.1:$Port"
$telegram = Join-Path $root 'live\telegram_bot.py'
$started = $null

function Get-Listener {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Assert-OurProcess($processId) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if (-not $process -or
        $process.CommandLine -notlike "*$dashboard*" -or
        $process.CommandLine -notlike "*$python*") {
        throw "Port $Port belongs to another application. Nothing was stopped. Close it or use a different port."
    }
}

try {
    $listener = Get-Listener
    if ($listener) {
        Assert-OurProcess $listener.OwningProcess
        if ($Action -eq 'Stop') {
            Stop-Process -Id $listener.OwningProcess -Force
            Get-CimInstance Win32_Process -Filter "CommandLine like '%telegram_bot.py%'" -ErrorAction SilentlyContinue |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Write-Host 'Dashboard and Telegram bot stopped. You can close the browser tab.'
            exit 0
        }
        Write-Host 'Dashboard is already running.'
    } elseif ($Action -eq 'Stop') {
        Get-CimInstance Win32_Process -Filter "CommandLine like '%telegram_bot.py%'" -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Write-Host 'Dashboard is already stopped.'
        exit 0
    } else {
        if (-not (Test-Path $python)) {
            throw 'Python environment is missing. Follow the Install section in README.md first.'
        }
        New-Item -ItemType Directory -Path $logs -Force | Out-Null
        Write-Host 'Starting SOL Market Brief...'
        $started = Start-Process -FilePath $python -ArgumentList @(
            '-u', "`"$dashboard`"", '--host', '127.0.0.1', '--port', "$Port"
        ) -WorkingDirectory $root -WindowStyle Hidden -PassThru `
          -RedirectStandardOutput (Join-Path $logs "dashboard-$Port.out") `
          -RedirectStandardError (Join-Path $logs "dashboard-$Port.err")
        $tgProc = Get-CimInstance Win32_Process -Filter "CommandLine like '%telegram_bot.py%'" -ErrorAction SilentlyContinue
        if (-not $tgProc -and (Test-Path $telegram)) {
            Start-Process -FilePath $python -ArgumentList @('-u', "`"$telegram`"") -WorkingDirectory $root -WindowStyle Hidden `
              -RedirectStandardOutput (Join-Path $logs "telegram.out") `
              -RedirectStandardError (Join-Path $logs "telegram.err")
        }
    }

    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try {
            $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 2
            if ($health.ok -eq $true) { $ready = $true; break }
        } catch { }
        if ($started) {
            $started.Refresh()
            if ($started.HasExited) { break }
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw "Dashboard did not become ready. Check $logs\dashboard-$Port.err for details."
    }
    $listener = Get-Listener
    if (-not $listener) { throw 'Dashboard listener disappeared during startup.' }
    Assert-OurProcess $listener.OwningProcess
    Write-Host "Ready: $url"
    Write-Host 'Read-only research dashboard. No orders will be placed.'
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
} catch {
    if ($started) {
        $started.Refresh()
        if (-not $started.HasExited) { Stop-Process -Id $started.Id -Force -ErrorAction SilentlyContinue }
    }
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}