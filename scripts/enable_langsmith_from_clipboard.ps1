param(
    [string]$Project = "deepresearch-dev",
    [string]$Environment = "development",
    [string]$Tags = "local,deepresearch",
    [switch]$HideInputs,
    [switch]$HideOutputs
)

$ErrorActionPreference = "Stop"
$apiKey = (Get-Clipboard -Raw).Trim()
if (-not $apiKey) {
    throw "Clipboard is empty. Copy your LangSmith API key first."
}
if ($apiKey -notmatch '^(lsv2_|ls__)') {
    Write-Warning "The clipboard text does not look like a standard LangSmith API key. Continuing anyway."
}

$script = Join-Path $PSScriptRoot "enable_langsmith.ps1"
& $script `
    -ApiKey $apiKey `
    -Project $Project `
    -Environment $Environment `
    -Tags $Tags `
    -HideInputs:$HideInputs `
    -HideOutputs:$HideOutputs
