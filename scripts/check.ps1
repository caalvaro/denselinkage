#!/usr/bin/env pwsh
# Local CI parity — runs the same checks GitHub Actions runs, against the
# repo's single .venv (created from .python-version).
#
# CI (.github/workflows/ci.yml) runs, per Python 3.10–3.13:
#   * lint-and-type: ruff check + ruff format --check + mypy + compileall,
#     over src/ tests/ examples/
#   * test matrix: pytest -m "not adapter and not slow"
#
# This reproduces all of that once, under the repo's single .venv; the full
# interpreter matrix lives in CI.

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Bin = Join-Path $RepoRoot '.venv\Scripts'

if (-not (Test-Path "$Bin\ruff.exe") -or
    -not (Test-Path "$Bin\mypy.exe") -or
    -not (Test-Path "$Bin\pytest.exe")) {
    Write-Error "Missing dev tools in .venv — run: uv venv .venv; uv pip install --python .venv\Scripts\python.exe -e '.[dev]'"
}

Write-Host '== ruff check (src, tests, examples) ==' -ForegroundColor Cyan
& "$Bin\ruff.exe" check src/ tests/ examples/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== ruff format --check ==' -ForegroundColor Cyan
& "$Bin\ruff.exe" format --check src/ tests/ examples/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== mypy (src + examples) ==' -ForegroundColor Cyan
& "$Bin\mypy.exe" src/ examples/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== compileall examples ==' -ForegroundColor Cyan
& "$Bin\python.exe" -m compileall -q examples
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# --cov is what makes this parity: pytest alone never exercises the
# `fail_under = 100` gate that CI enforces.
Write-Host '== pytest (CI markers, with the 100% coverage gate) ==' -ForegroundColor Cyan
& "$Bin\pytest.exe" -m "not adapter and not slow" -q --cov=denselinkage --cov-report=term
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'All checks passed.' -ForegroundColor Green
