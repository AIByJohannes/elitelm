<#
.SYNOPSIS
    Initializes the Visual Studio Developer Environment for ARM64.

.DESCRIPTION
    This script locates the latest Visual Studio installation and activates the
    Developer Environment configured for the ARM64 architecture.
    
    IMPORTANT: You must "dot-source" this script for the environment variables
    to persist in your current shell session.

    Usage:
        . .\scripts\init_env.ps1

.NOTES
    Requires Visual Studio 2019 or later with C++ workload.
#>

$ErrorActionPreference = "Stop"

# Locate vswhere.exe
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found at '$vswhere'. Is Visual Studio installed?"
    return
}

# Find latest Visual Studio installation with C++ tools
$installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath

if (-not $installPath) {
    Write-Warning "Visual Studio with C++ tools was not found."
    Write-Warning "Please install 'Visual Studio 2022' (Community is free)."
    Write-Warning "Download: https://visualstudio.microsoft.com/downloads/"
    Write-Warning "IMPORTANT: During installation, select the 'Desktop development with C++' workload."
    Write-Warning "Ensure 'C++ Clang Compiler' and 'C++ CMake tools' are selected on the right."
    throw "Visual Studio C++ environment not found."
}

# Construct path to Launch-VsDevShell.ps1
$devShellScript = Join-Path $installPath "Common7\Tools\Launch-VsDevShell.ps1"

if (-not (Test-Path $devShellScript)) {
    Write-Error "Launch-VsDevShell.ps1 not found at '$devShellScript'."
    return
}

Write-Host "Found Visual Studio at: $installPath"
Write-Host "Initializing Developer Environment for ARM64..."

# Execute the VS Dev Shell script
# Note: When dot-sourced, this affects the current session.
& $devShellScript -Arch arm64 -SkipAutomaticLocation
