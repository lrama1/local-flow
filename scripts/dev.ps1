# Windows convenience wrapper: runs backend + frontend dev servers (see scripts/dev.mjs).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
node scripts/dev.mjs
