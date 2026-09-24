$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.11 for Windows with the Python launcher, then rerun this script.'
}
foreach ($binary in @('ffmpeg', 'ffprobe', 'nvidia-smi')) {
    if (-not (Get-Command $binary -ErrorAction SilentlyContinue)) {
        throw "$binary is missing from PATH. Install FFmpeg and a current NVIDIA driver, then rerun."
    }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv')
}
if (-not (Test-Path -LiteralPath '.transcribe-venv\Scripts\python.exe')) {
    Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.transcribe-venv')
}
Invoke-Checked '.\.venv\Scripts\python.exe' @('-m', 'pip', 'install', '-r', 'requirements-installed.txt')
# Use the same analysis versions as the Mac, with CUDA-enabled Torch wheels.
Invoke-Checked '.\.transcribe-venv\Scripts\python.exe' @('-m', 'pip', 'install', 'torch==2.8.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
Invoke-Checked '.\.transcribe-venv\Scripts\python.exe' @('-m', 'pip', 'install', '-r', 'requirements-transcriber.txt')
if (-not (Test-Path -LiteralPath '.cuda-venv\Scripts\python.exe')) {
    Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.cuda-venv')
}
Invoke-Checked '.\.cuda-venv\Scripts\python.exe' @('-m', 'pip', 'install', 'torch==2.10.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
Invoke-Checked '.\.cuda-venv\Scripts\python.exe' @('-m', 'pip', 'install', '-r', 'requirements-cuda.txt')
Invoke-Checked '.\.cuda-venv\Scripts\python.exe' @('check_cuda_runtime.py')
Invoke-Checked '.\.venv\Scripts\python.exe' @('check_runtime.py')
Invoke-Checked '.\.transcribe-venv\Scripts\python.exe' @('-c', 'import torch, torchaudio, transformers, scipy, pretty_midi; assert torch.cuda.is_available(), "CUDA is unavailable for cover analysis"')
Write-Host 'OpenSuno is ready. Run Launch Studio.ps1, then download a model in Models.'
