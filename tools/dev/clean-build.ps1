# Build and cache hygiene for the Rust workspace.
#
#   .\tools\dev\clean-build.ps1                 # report sizes, delete stale/failed output
#   .\tools\dev\clean-build.ps1 -DryRun         # report only
#   .\tools\dev\clean-build.ps1 -Days 3         # treat output unused for 3 days as stale
#
# Deleted output is always reproducible: it is rebuilt or re-emitted on demand.
# Model files (Hugging Face cache) and sources are never touched.
param(
    [int]$Days = 7,
    [switch]$DryRun,
    [string]$Cache = 'C:\claude-work\cache'
)

$ErrorActionPreference = 'Stop'

function Get-DirMB([string]$Path) {
    if (-not (Test-Path $Path)) { return 0 }
    $sum = (Get-ChildItem $Path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    [math]::Round(($sum / 1MB), 1)
}

function Report([string]$Title) {
    Write-Host $Title
    foreach ($d in 'cargo-target', 'forge', 'cargo-home', 'rustup-home') {
        $p = Join-Path $Cache $d
        if (Test-Path $p) { Write-Host ('  {0,-14} {1,10:N1} MB' -f $d, (Get-DirMB $p)) }
    }
}

function Remove-Path([string]$Path) {
    if ($DryRun) { Write-Host "  [dry-run] would delete $Path"; return }
    Remove-Item $Path -Recurse -Force -Confirm:$false
}

Report 'before:'
$cutoff = (Get-Date).AddDays(-$Days)

# 1. Failed builds: any marked failed stage under the cache is removed at once.
Get-ChildItem $Cache -Recurse -Directory -Force -Filter '.FAILED-*' -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Path $_.FullName }

# 2. Incremental compilation data is not needed (release builds disable it).
Get-ChildItem (Join-Path $Cache 'cargo-target') -Recurse -Directory -Force -Filter 'incremental' -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Path $_.FullName }

# 3. Stale build output: files in cargo-target and forge not written within -Days.
foreach ($d in 'cargo-target', 'forge') {
    $root = Join-Path $Cache $d
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem $root -Recurse -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object { Remove-Path $_.FullName }
}

Report 'after:'
