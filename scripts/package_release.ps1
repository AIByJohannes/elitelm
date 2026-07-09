# EliteLM Release Packaging Script for Windows ARM64
$ErrorActionPreference = "Stop"

# 1. Paths setup
$projectRoot = (Get-Item $PSScriptRoot).Parent.FullName
$distDir = Join-Path $projectRoot "dist"
$packageDir = Join-Path $distDir "elitelm-windows-arm64"
$zipPath = Join-Path $distDir "elitelm-windows-arm64.zip"

Write-Host "Cleaning up old package directories..." -ForegroundColor Cyan
if (Test-Path $distDir) {
    Remove-Item -Recurse -Force $distDir
}
New-Item -ItemType Directory -Path $packageDir | Out-Null
New-Item -ItemType Directory -Path (Join-Path $packageDir "bin") | Out-Null

# 2. Compile release binaries
Write-Host "Building EliteLM workspace in Release mode..." -ForegroundColor Cyan
Set-Location $projectRoot
cargo build --release --workspace

# 3. Verify compilation outputs
$cliExe = Join-Path $projectRoot "target\release\elitelm-cli.exe"
$serverExe = Join-Path $projectRoot "target\release\elitelm-server.exe"

if (-not (Test-Path $cliExe)) {
    throw "elitelm-cli.exe not found in release output target."
}
if (-not (Test-Path $serverExe)) {
    throw "elitelm-server.exe not found in release output target."
}

# 4. Copy assets into packaging layout
Write-Host "Staging files for release package..." -ForegroundColor Cyan
Copy-Item $cliExe (Join-Path $packageDir "bin\elitelm-cli.exe")
Copy-Item $serverExe (Join-Path $packageDir "bin\elitelm-server.exe")

$exampleYaml = Join-Path $projectRoot "elitelm.example.yaml"
if (Test-Path $exampleYaml) {
    Copy-Item $exampleYaml (Join-Path $packageDir "elitelm.example.yaml")
}

$readme = Join-Path $projectRoot "README.md"
if (Test-Path $readme) {
    Copy-Item $readme (Join-Path $packageDir "README.md")
}

# 5. Create zip archive
Write-Host "Compressing package into $zipPath..." -ForegroundColor Cyan
Compress-Archive -Path "$packageDir\*" -DestinationPath $zipPath -Force

# Cleanup temp folder
Remove-Item -Recurse -Force $packageDir

Write-Host "Package successfully created: $zipPath" -ForegroundColor Green
