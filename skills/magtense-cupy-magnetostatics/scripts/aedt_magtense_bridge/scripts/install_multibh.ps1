# One-command installer for Windows when Intel ifx/MKL, make and Python 3.12+ are available.
# Run from repository root: powershell -ExecutionPolicy Bypass -File .\scripts\install_multibh.ps1
param(
    [string]$MagTenseRepo = ".\MagTense",
    [ValidateSet("ifx", "gfortran")]
    [string]$Compiler = "ifx"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Install Git before running this script" }
if (-not (Get-Command make -ErrorAction SilentlyContinue)) { throw "Install make (Conda/MSYS) before running this script" }
if (-not (Test-Path (Join-Path $MagTenseRepo "Makefile"))) {
    git clone https://github.com/cmt-dtu-energy/MagTense.git $MagTenseRepo
    if ($LASTEXITCODE -ne 0) { throw "MagTense git clone failed" }
}
python patch_magtense.py --repo $MagTenseRepo --build --fc $Compiler
if ($LASTEXITCODE -ne 0) { throw "MagTense native rebuild failed" }
python -m pip uninstall -y magtense
if ($LASTEXITCODE -ne 0) { throw "pip uninstall failed" }
python -m pip install -e (Join-Path $MagTenseRepo "python")
if ($LASTEXITCODE -ne 0) { throw "Installing patched MagTense failed" }
python native_smoke.py
if ($LASTEXITCODE -ne 0) { throw "Multi-BH native smoke test failed" }
Write-Host "Multi-BH MagTense installed and smoke-tested." -ForegroundColor Green
