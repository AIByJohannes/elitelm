try {
    $content = Invoke-WebRequest -Uri "https://github.com/ggerganov/llama.cpp/releases/tag/b7801" -UseBasicParsing
    $links = $content.Links | Where-Object { $_.href -like "*arm64*.zip" } | Select-Object -ExpandProperty href
    if ($links) {
        Write-Host "Found links:"
        $links | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "No links found matching *arm64*.zip"
    }
} catch {
    Write-Host "Error fetching page: $_"
}
