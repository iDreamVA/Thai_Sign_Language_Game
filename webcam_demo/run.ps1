$ErrorActionPreference = 'Stop'
$appPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    throw 'ไม่พบ .venv ให้ติดตั้งตาม README.md ก่อน'
}
& $appPython (Join-Path $PSScriptRoot 'app.py') @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
