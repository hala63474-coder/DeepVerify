$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Py   = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "[setup] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv $Venv
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r (Join-Path $Root "requirements.txt")
}

$existing = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[setup] Stopping previous server on port 5000..." -ForegroundColor Yellow
    Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

Write-Host "[run] Launching DeepVerify on http://127.0.0.1:5000/" -ForegroundColor Green
Set-Location (Join-Path $Root "app")
Start-Process "http://127.0.0.1:5000/"
& $Py app.py
