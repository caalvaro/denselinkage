#!/usr/bin/env pwsh
# Local CI parity — runs the same checks GitHub Actions runs.
#
# CI (see .github/workflows/ci.yml) runs:
#   * lint-and-type on Python 3.12: ruff check src/ tests/ && mypy src/
#   * test matrix on 3.10/3.11/3.12: pytest -m "not adapter and not slow"
#
# This script reproduces the lint + type-check exactly (Python 3.12 venv
# is required because pandas-stubs / numpy stubs have shipped strict-mode
# changes between versions). The test suite runs under .venv (Python 3.10
# locally) — close enough for most regressions; the full matrix lives in CI.

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Py312 = Join-Path $RepoRoot '.venv-py312\Scripts'
$Py310 = Join-Path $RepoRoot '.venv\Scripts'

if (-not (Test-Path "$Py312\mypy.exe")) {
    Write-Error "Missing $Py312 — run: uv venv --python 3.12 .venv-py312; uv pip install --python .venv-py312\Scripts\python.exe -e '.[dev]'"
}
if (-not (Test-Path "$Py310\pytest.exe")) {
    Write-Error "Missing $Py310 — run: uv venv --python 3.10 .venv; uv pip install --python .venv\Scripts\python.exe -e '.[dev,faiss]'"
}

Write-Host '== ruff (py3.12) ==' -ForegroundColor Cyan
& "$Py312\ruff.exe" check src/ tests/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== mypy (py3.12) ==' -ForegroundColor Cyan
& "$Py312\mypy.exe" src/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== pytest (py3.10, CI markers) ==' -ForegroundColor Cyan
& "$Py310\pytest.exe" -m "not adapter and not slow" -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'All checks passed.' -ForegroundColor Green
