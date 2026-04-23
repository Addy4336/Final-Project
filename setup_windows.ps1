param(
    [string]$VenvName = ".venv_local"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot $VenvName
$pythonInVenv = Join-Path $venvPath "Scripts\python.exe"

Write-Host "[setup] Project root: $projectRoot"

if (-not (Test-Path $pythonInVenv)) {
    Write-Host "[setup] Creating virtual environment at $venvPath"
    python -m venv $VenvName
}

Write-Host "[setup] Upgrading pip"
& $pythonInVenv -m pip install --upgrade pip

Write-Host "[setup] Installing dependencies from requirements.txt"
& $pythonInVenv -m pip install -r (Join-Path $projectRoot "requirements.txt")

$tesseractCandidates = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
)

$tesseractFound = $false
foreach ($candidate in $tesseractCandidates) {
    if (Test-Path $candidate) {
        Write-Host "[setup] Found Tesseract at: $candidate"
        $tesseractFound = $true
        break
    }
}

if (-not $tesseractFound) {
    Write-Warning "Tesseract binary not found in default locations. OCR will be limited until Tesseract is installed or added to PATH."
}

Write-Host "[setup] Complete. To run the server:"
Write-Host "  .\\$VenvName\\Scripts\\Activate.ps1"
Write-Host "  uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000"
