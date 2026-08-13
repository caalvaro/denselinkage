#!/usr/bin/env pwsh
# Local CI parity: runs the same checks GitHub Actions runs, against the
# repo's single .venv (created from .python-version).
#
# CI (.github/workflows/ci.yml) runs, per Python 3.10-3.13:
#   * lint-and-type: ruff check + ruff format --check + mypy + compileall +
#     running the dependency-free examples, over src/ tests/ examples/
#     .claude/hooks/
#   * test matrix: pytest -m "not adapter and not slow"
#
# This reproduces all of that once, under the repo's single .venv; the full
# interpreter matrix lives in CI.
#
# ASCII only, deliberately. Windows PowerShell 5.1 decodes a BOM-less .ps1 as
# the system ANSI codepage, so a UTF-8 em dash arrives as three CP1252
# characters, one of which is U+201D. 5.1 accepts curly quotes as string
# delimiters, so a single em dash inside a double-quoted string ends it early
# and the whole file stops parsing.

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Bin = Join-Path $RepoRoot '.venv\Scripts'

if (-not (Test-Path "$Bin\ruff.exe") -or
    -not (Test-Path "$Bin\mypy.exe") -or
    -not (Test-Path "$Bin\pytest.exe")) {
    # `uv sync --locked`, not `uv pip install -e '.[dev]'`: the latter resolves
    # fresh and gives a different toolchain from the one CI runs (issue #9).
    Write-Error "Missing dev tools in .venv. Run: uv sync --locked --extra dev"
}

Write-Host '== ruff check (src, tests, examples, hooks) ==' -ForegroundColor Cyan
& "$Bin\ruff.exe" check src/ tests/ examples/ .claude/hooks/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== ruff format --check ==' -ForegroundColor Cyan
& "$Bin\ruff.exe" format --check src/ tests/ examples/ .claude/hooks/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== mypy (src + examples + hooks) ==' -ForegroundColor Cyan
& "$Bin\mypy.exe" src/ examples/ .claude/hooks/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== compileall examples ==' -ForegroundColor Cyan
& "$Bin\python.exe" -m compileall -q examples
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Compiling an example is not running it. CI executes these four; without them a
# module-scope crash reaches the PR.
Write-Host '== run the dependency-free examples ==' -ForegroundColor Cyan
foreach ($example in '00_quickstart', '03_custom_embedder', '04_dedupe', '05_failure_accounting') {
    & "$Bin\python.exe" "examples/$example.py" | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# --cov is what makes this parity: pytest alone never exercises the
# `fail_under = 100` gate that CI enforces.
Write-Host '== pytest (CI markers, with the 100% coverage gate) ==' -ForegroundColor Cyan
& "$Bin\pytest.exe" -m "not adapter and not slow" -q --cov=denselinkage --cov-report=term
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'All checks passed.' -ForegroundColor Green
