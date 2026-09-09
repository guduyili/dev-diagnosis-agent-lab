[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^(D\d{2}|W0[1-8]-[ABC])$')]
    [string]$Day,
    [switch]$Cumulative
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Missing project Python. Run uv sync --frozen in the project root first.'
}
$taskReportDir = Join-Path $taskRoot 'reports\raw'
New-Item -ItemType Directory -Path $taskReportDir -Force | Out-Null
$taskStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$taskReport = Join-Path $taskReportDir "daily-upstream-$Day-$taskStamp.xml"
$taskSelector = if ($Cumulative) { '--through-day' } else { '--day' }
$taskOldUtf8 = $env:PYTHONUTF8
$taskExitCode = 1
Push-Location -LiteralPath $taskRoot
try {
    $env:PYTHONUTF8 = '1'
    & $taskPython -m pytest tests/daily $taskSelector $Day -q "--junitxml=$taskReport"
    $taskExitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONUTF8 = $taskOldUtf8
    Pop-Location
}
exit $taskExitCode
