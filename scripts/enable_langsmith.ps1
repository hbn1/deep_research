param(
    [Parameter(Mandatory = $true)]
    [string]$ApiKey,

    [string]$Project = "deepresearch-dev",

    [string]$Environment = "development",

    [string]$Tags = "local,deepresearch",

    [switch]$HideInputs,

    [switch]$HideOutputs
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$envPath = Join-Path $root ".env"

if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath (Join-Path $root ".env.example") -Destination $envPath
}

$updates = [ordered]@{
    "LANGSMITH_ENABLED" = "true"
    "LANGSMITH_TRACING" = "true"
    "LANGSMITH_API_KEY" = $ApiKey
    "LANGSMITH_PROJECT" = $Project
    "LANGSMITH_ENDPOINT" = "https://api.smith.langchain.com"
    "LANGSMITH_TAGS" = $Tags
    "LANGSMITH_ENVIRONMENT" = $Environment
    "LANGSMITH_SAMPLE_RATE" = "1.0"
    "LANGSMITH_HIDE_INPUTS" = $(if ($HideInputs) { "true" } else { "false" })
    "LANGSMITH_HIDE_OUTPUTS" = $(if ($HideOutputs) { "true" } else { "false" })
}

$lines = New-Object System.Collections.Generic.List[string]
if (Test-Path -LiteralPath $envPath) {
    foreach ($line in Get-Content -LiteralPath $envPath -Encoding UTF8) {
        $lines.Add($line)
    }
}

if (-not ($lines | Where-Object { $_ -match '^\s*# --- LangSmith' })) {
    if ($lines.Count -gt 0 -and $lines[$lines.Count - 1].Trim() -ne "") {
        $lines.Add("")
    }
    $lines.Add("# --- LangSmith 可观测性 ---")
}

foreach ($key in $updates.Keys) {
    $value = $updates[$key]
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($key))=") {
            $lines[$i] = "$key=$value"
            $found = $true
            break
        }
    }
    if (-not $found) {
        $lines.Add("$key=$value")
    }
}

Set-Content -LiteralPath $envPath -Encoding UTF8 -Value $lines
Write-Host "LangSmith enabled for project '$Project'. Restart the backend for changes to take effect."
