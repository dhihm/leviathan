# Compares the Rust tokenizer with llama.cpp's /tokenize on a corpus of strings.
#
#   . <repo>\cache\env.ps1
#   .\tools\bench\tokenizer-check.ps1 -Model <gguf> [-Special]
#
# Every string of tokenizer-corpus.json is tokenized by `lev tokenize` and by a llama-server
# (add_special false); the id lists must be identical. Prints the differing strings.
param(
    [Parameter(Mandatory)] [string] $Model,
    [string] $Corpus = (Join-Path $PSScriptRoot 'tokenizer-corpus.json'),
    [string] $Lev = "$env:CARGO_TARGET_DIR\release\lev.exe",
    [string] $LlamaBin = "$env:USERPROFILE\.cache\lemonade\bin\llamacpp\vulkan",
    [switch] $Special,
    [int] $Port = 8097
)
$ErrorActionPreference = 'Stop'
$strings = [System.IO.File]::ReadAllText($Corpus, [Text.Encoding]::UTF8) | ConvertFrom-Json
$ErrorActionPreference = 'Continue'   # lev prints a timing line on stderr
$ours = if ($Special) { & $Lev tokenize $Model $Corpus special 2>$null } else { & $Lev tokenize $Model $Corpus 2>$null }
$ErrorActionPreference = 'Stop'
$ours = @($ours)

$srv = Start-Process -FilePath "$LlamaBin\llama-server.exe" -WindowStyle Hidden -PassThru -ArgumentList @('-m', $Model, '-ngl', '0', '-c', '512', '--port', "$Port", '--no-webui', '--no-warmup')
try {
    for ($i = 0; $i -lt 90; $i++) { Start-Sleep 1; try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq 'ok') { break } } catch { } }
    $bad = 0
    for ($i = 0; $i -lt $strings.Count; $i++) {
        $body = @{ content = [string]$strings[$i]; add_special = $false; parse_special = [bool]$Special } | ConvertTo-Json -Compress
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json; charset=utf-8' -UseBasicParsing
        $want = (([Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray()) | ConvertFrom-Json).tokens) -join ','
        if ($want -ne $ours[$i]) {
            $bad++
            $shown = ([string]$strings[$i]) -replace "`r", '\r' -replace "`n", '\n' -replace "`t", '\t'
            if ($shown.Length -gt 70) { $shown = $shown.Substring(0, 70) + '...' }
            Write-Host ("DIFF #{0}: {1}" -f $i, $shown)
            Write-Host ("   llama.cpp: {0}" -f $want.Substring(0, [Math]::Min(120, $want.Length)))
            Write-Host ("   ours     : {0}" -f ([string]$ours[$i]).Substring(0, [Math]::Min(120, ([string]$ours[$i]).Length)))
        }
    }
    Write-Host ("{0} of {1} strings identical{2}" -f ($strings.Count - $bad), $strings.Count, $(if ($Special) { ' (special tokens parsed)' } else { '' }))
    if ($bad -gt 0) { exit 1 }
} finally {
    Stop-Process -Id $srv.Id -ErrorAction SilentlyContinue
}
