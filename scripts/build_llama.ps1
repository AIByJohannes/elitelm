$cmakePath = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$repoDir = Resolve-Path ".\llama.cpp"
$buildDir = "$repoDir\build"
$preset = "arm64-windows-llvm-release"

Write-Host "Configuring cmake..."
Set-Location $repoDir

if (Test-Path "$repoDir\CMakePresets.json") {
    Write-Host "Using preset: $preset"
    & $cmakePath --preset $preset
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed (preset: $preset)"
        exit 1
    }
} else {
    if (-not (Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
    }
    & $cmakePath -S . -B $buildDir -G "Visual Studio 17 2022" -A ARM64 -T ClangCL -DGGML_OPENMP=OFF
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed (ClangCL toolset)"
        exit 1
    }
}

Write-Host "Building..."
if (Test-Path $buildDir) {
    & $cmakePath --build $buildDir --config Release -j 8
} else {
    & $cmakePath --build --preset $preset
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit 1
}

Write-Host "Build complete."
