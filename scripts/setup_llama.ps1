$url = "https://github.com/ggml-org/llama.cpp/releases/download/b7801/llama-b7801-bin-win-cpu-arm64.zip"
$destDir = Resolve-Path "."
$binDir = "$destDir\bin\llama.cpp"
$zipPath = "$destDir\llama_cpp.zip"

if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

Write-Host "Downloading llama.cpp from $url..."
# GitHub requires User-Agent. Using a browser-like one.
$userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
Invoke-WebRequest -Uri $url -OutFile $zipPath -UserAgent $userAgent

if (-not (Test-Path $zipPath)) {
    Write-Error "Download failed: File not found."
    exit 1
}

Write-Host "Extracting to $binDir..."
Expand-Archive -Path $zipPath -DestinationPath $binDir -Force

Remove-Item $zipPath

Write-Host "llama.cpp setup complete."
Write-Host "Contents of $binDir :"
Get-ChildItem $binDir | Select-Object Name