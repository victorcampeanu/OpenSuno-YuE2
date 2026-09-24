# Starts this Windows PC as an OpenSuno render node: a Studio on another machine sends songs here to render
# on the NVIDIA GPU with the official CUDA runtime.
# Set $env:OPENSUNO_NODE_TOKEN to accept connections from other machines (without it, only this PC can connect).
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & (Join-Path $PSScriptRoot 'Install OpenSuno.ps1')
}
if (Test-Path -LiteralPath '.cuda-venv\Scripts\python.exe') {
    & '.\.cuda-venv\Scripts\python.exe' 'check_cuda_runtime.py'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime check failed. Run Install OpenSuno.ps1.' }
} else {
    Write-Host 'No .cuda-venv yet: the CUDA model will show as not installed until Install OpenSuno.ps1 has run.'
}
if (-not $env:OPENSUNO_NODE_TOKEN) {
    Write-Host 'OPENSUNO_NODE_TOKEN is not set: only this PC can connect. To serve other machines, run:'
    Write-Host '  $env:OPENSUNO_NODE_TOKEN = "choose-a-secret"; .\"Launch Render Node.ps1"'
}
$port = if ($env:OPENSUNO_NODE_PORT) { $env:OPENSUNO_NODE_PORT } else { '7863' }
Write-Host "Render node on port $port. Allow it through Windows Firewall when asked. Keep this window open; Ctrl+C stops it."
& '.\.venv\Scripts\python.exe' 'render_node.py'
if ($LASTEXITCODE -ne 0) { throw "Render node exited with code $LASTEXITCODE" }
