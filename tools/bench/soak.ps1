# Time-series benchmark: this engine and llama.cpp take turns on the same GPU
# while a sampler records AMD GPU telemetry (ADL PMLog: temperatures, GFX clock,
# power, activity, throttler status). The point is to compare the two under the
# same thermal and power history instead of one cold or hot snapshot.
#
#   . <repo>\cache\env.ps1
#   .\tools\bench\soak.ps1 -Model <gguf> -Ids <ids file> -OutDir <dir>
#
# Output (all timestamps are unix milliseconds):
#   telemetry.csv  unix_ms,gfx_c,soc_c,cpu_c,gfx_mhz,mem_mhz,asic_w,gfx_act_pct,throttler  (~2 samples/s)
#   ours.csv       SOAK,unix_ms,prefill_tps,plain_tps,spec_tps,acceptance
#   llama.csv      LLAMA,unix_ms,prefill_tps,decode_tps
#   phases.csv     PHASE,unix_ms,engine,start|end
param(
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $Ids,
    [Parameter(Mandatory)] [string] $OutDir,
    [string] $Lev = "$env:CARGO_TARGET_DIR\release\lev.exe",
    [string] $LlamaBin = "$env:USERPROFILE\.cache\lemonade\bin\llamacpp\vulkan",
    [int] $Rounds = 4,
    [int] $SliceSeconds = 100,
    [int] $Port = 8099
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$now = { [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() }
$stop = Join-Path $OutDir 'stop.flag'
Remove-Item $stop -ErrorAction SilentlyContinue
foreach ($f in 'telemetry.csv', 'ours.csv', 'llama.csv', 'phases.csv') { Remove-Item (Join-Path $OutDir $f) -ErrorAction SilentlyContinue }
$phases = Join-Path $OutDir 'phases.csv'
$mark = { param($engine, $what) Add-Content $phases "PHASE,$(& $now),$engine,$what" }

# Sampler in its own process so it keeps ticking while a benchmark runs.
$adl = Join-Path $PSScriptRoot 'adl.ps1'
$sampler = Start-Job -ArgumentList $adl, (Join-Path $OutDir 'telemetry.csv'), $stop -ScriptBlock {
    param($adlScript, $csv, $stopFlag)
    . $adlScript
    while (-not (Test-Path $stopFlag)) {
        try {
            $s = Read-AdlSensors -Adapter 0
            if ($s) {
                $ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
                Add-Content $csv ('{0},{1},{2},{3},{4},{5},{6},{7},{8}' -f $ms, $s.temp_gfx_c, $s.temp_soc_c, $s.temp_cpu_c, $s.gfx_clk_mhz, $s.mem_clk_mhz, $s.asic_power_w, $s.gfx_activity_pct, $s.throttler_status)
            }
        } catch { }
        Start-Sleep -Milliseconds 400
    }
}

$promptIds = [int[]]((Get-Content $Ids -Raw).Trim().Split(','))
try {
    for ($r = 1; $r -le $Rounds; $r++) {
        Write-Host "round ${r}/${Rounds}: ours"
        & $mark 'ours' 'start'
        & $Lev gpu-soak $Model $SliceSeconds $Ids | Add-Content (Join-Path $OutDir 'ours.csv')
        & $mark 'ours' 'end'

        Write-Host "round ${r}/${Rounds}: llama.cpp"
        $srv = Start-Process -FilePath "$LlamaBin\llama-server.exe" -WindowStyle Hidden -PassThru -ArgumentList @('-m', $Model, '-ngl', '99', '-c', '2048', '--port', "$Port", '--no-webui', '--no-warmup')
        $ok = $false
        for ($i = 0; $i -lt 90 -and -not $ok; $i++) {
            Start-Sleep 1
            try { $ok = (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq 'ok' } catch { }
        }
        if (-not $ok) { Stop-Process -Id $srv.Id -ErrorAction SilentlyContinue; throw 'llama-server did not become healthy' }
        & $mark 'llama' 'start'
        $end = (Get-Date).AddSeconds($SliceSeconds)
        $body = @{ prompt = $promptIds; n_predict = 64; temperature = 0; cache_prompt = $false } | ConvertTo-Json -Compress
        while ((Get-Date) -lt $end) {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType 'application/json' -UseBasicParsing -TimeoutSec 300
            $j = [Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray()) | ConvertFrom-Json
            Add-Content (Join-Path $OutDir 'llama.csv') ('LLAMA,{0},{1:F2},{2:F2}' -f (& $now), $j.timings.prompt_per_second, $j.timings.predicted_per_second)
        }
        & $mark 'llama' 'end'
        Stop-Process -Id $srv.Id -ErrorAction SilentlyContinue
        Start-Sleep 3
    }
} finally {
    Set-Content $stop 'stop'
    Wait-Job $sampler -Timeout 10 | Out-Null
    Remove-Job $sampler -Force -ErrorAction SilentlyContinue
}
Write-Host "done: $OutDir"
