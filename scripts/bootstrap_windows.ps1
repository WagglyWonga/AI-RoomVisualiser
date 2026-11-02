param(
    [string]$PythonSelector = "-3.12",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment at $VenvPath using py $PythonSelector"
$ErrorActionPreference = "Continue"
py $PythonSelector -m venv $VenvPath
$venvExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"

$venvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Expected virtual environment python not found at $venvPython"
}

Write-Host "Virtual environment created. Detected version:"
& $venvPython --version

if ($venvExitCode -ne 0) {
    Write-Warning "The virtual environment command reported a non-zero exit code. Verifying pip availability next."
}

$ErrorActionPreference = "Continue"
& $venvPython -m pip --version > $null 2> $null
$pipExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"

if ($pipExitCode -ne 0) {
    Write-Warning "The Python 3.12 virtual environment was created, but pip is unavailable inside it."
    Write-Warning "This can happen in restricted environments when ensurepip cannot use the temp directory."
    Write-Host ""
    Write-Host "Fallback for this environment:"
    Write-Host "1. py -3.12 scripts\\preflight.py"
    Write-Host "2. py -3.12 scripts\\smoke_test_models.py"
    Write-Host "3. Create the virtual environment from a normal local shell if you need pip-managed installs"
    exit 1
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "1. .\$VenvPath\Scripts\Activate.ps1"
Write-Host "2. python -m pip install --upgrade pip"
Write-Host "3. python -m pip install -e "".[dev]"""
Write-Host "4. python scripts\preflight.py"
