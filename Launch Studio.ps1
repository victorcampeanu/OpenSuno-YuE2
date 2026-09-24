$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$studioPort = if ($env:YUE2_PORT) { $env:YUE2_PORT } else { '7862' }
$studioUrl = "http://127.0.0.1:$studioPort"
try {
    $runningStudio = Invoke-RestMethod -Uri "$studioUrl/api/config" -TimeoutSec 2
    if ($runningStudio.models -and $runningStudio.defaults) {
        Write-Host "OpenSuno is already running at $studioUrl"
        return
    }
} catch { }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & (Join-Path $PSScriptRoot 'Install OpenSuno.ps1')
}
if (Test-Path -LiteralPath '.cuda-venv\Scripts\python.exe') {
    & '.\.cuda-venv\Scripts\python.exe' 'check_cuda_runtime.py'
} else {
    & '.\.venv\Scripts\python.exe' 'check_runtime.py'
}
if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed. Run Install OpenSuno.ps1.' }
Write-Host "Open $studioUrl in your browser. Keep this terminal open; Ctrl+C stops Studio."
& '.\.venv\Scripts\python.exe' 'server.py'
if ($LASTEXITCODE -ne 0) { throw "Studio exited with code $LASTEXITCODE" }
