# Renders the CSVs written by soak.ps1 as one SVG with five stacked panels on a
# shared time axis (never a dual axis): prefill throughput, decode throughput,
# GPU temperature, GFX clock and ASIC power. A strip under the last panel shows
# which engine was running. Also prints a per-slice summary table.
#
#   .\tools\bench\plot-soak.ps1 -Dir <soak output dir> -Out <file.svg>
param(
    [Parameter(Mandatory)] [string] $Dir,
    [Parameter(Mandatory)] [string] $Out,
    [string] $Title = 'Ornith-1.5-9B on Radeon 890M: throughput against GPU clock, power and temperature'
)
$inv = [Globalization.CultureInfo]::InvariantCulture
function Read-Rows($name) {
    $f = Join-Path $Dir $name
    if (-not (Test-Path $f)) { return @() }
    Get-Content $f | Where-Object { $_.Trim() } | ForEach-Object { , ($_.Split(',')) }
}
function D($x) { [double]::Parse("$x", $inv) }
$tel = Read-Rows 'telemetry.csv'
$ours = Read-Rows 'ours.csv'
$llama = Read-Rows 'llama.csv'
$phases = Read-Rows 'phases.csv'
$t0 = D $tel[0][0]
function Min($ms) { ((D $ms) - $t0) / 60000.0 }
function Pts($rows, $xi, $yi) { $rows | ForEach-Object { [pscustomobject]@{ x = (Min $_[$xi]); y = (D $_[$yi]) } } }

$S = @{
    prefill_ours = Pts $ours 1 2;  prefill_llama = Pts $llama 1 2
    plain_ours = Pts $ours 1 3;    spec_ours = Pts $ours 1 4;   decode_llama = Pts $llama 1 3
    gfx_c = Pts $tel 0 1;          soc_c = Pts $tel 0 2;        clk = Pts $tel 0 4;   pwr = Pts $tel 0 6
}
$xmax = [math]::Ceiling(($S.clk | Measure-Object x -Maximum).Maximum)
function Nice($v, $step) { [math]::Ceiling($v / $step) * $step }
function Ymax($arrays, $step) { Nice (($arrays | ForEach-Object { $_ } | Measure-Object y -Maximum).Maximum) $step }

$W = 1100; $ml = 78; $mr = 118; $ph = 150; $gap = 46; $top = 126; $np = 5
$H = $top + $np * $ph + ($np - 1) * $gap + 100
$pw = $W - $ml - $mr
function X($m) { $ml + $pw * $m / $xmax }

$sb = New-Object Text.StringBuilder
function Add($t) { $null = $sb.AppendLine($t) }
Add ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {0} {1}" width="{0}" height="{1}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">' -f $W, $H)
Add ('<title>{0}</title>' -f $Title)
Add @'
<style>
  :root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --g:#52514e; }
  @media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#33322f; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --g:#c3c2b7; } }
  .bg { fill: var(--surface); } .t { fill: var(--ink); } .t2 { fill: var(--ink2); } .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--ink2); stroke-width: 1; }
  .s1 { stroke: var(--s1); fill: var(--s1); } .s2 { stroke: var(--s2); fill: var(--s2); } .s3 { stroke: var(--s3); fill: var(--s3); } .g { stroke: var(--g); fill: var(--g); }
  .ln { fill: none; stroke-width: 2; } .dash { stroke-dasharray: 6 4; } .mk { stroke: var(--surface); stroke-width: 2; }
  .sep { stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 2 4; opacity: .55; }
  .band1 { fill: var(--s1); opacity: .9; } .band2 { fill: var(--s2); opacity: .9; }
</style>
'@
Add ('<rect class="bg" width="{0}" height="{1}"/>' -f $W, $H)
Add ('<text class="t" x="{0}" y="28" font-size="17" font-weight="600">{1}</text>' -f $ml, $Title)
Add ('<text class="t2" x="{0}" y="48" font-size="12">Same GPU, the two engines take turns (strip under the last panel). Prompt 512 tokens, 64 generated, greedy. Telemetry: AMD ADL PMLog, about 2 samples/s.</text>' -f $ml)

# Legend: names sit next to the swatches, so identity is never color alone.
$leg = @(@('s1', 'This engine: prefill, plain decode'), @('s3', 'This engine: MTP speculative decode'), @('s2', 'llama.cpp b11393 (plain)'), @('g', 'GPU telemetry'))
$lx = $ml
foreach ($l in $leg) {
    Add ('<circle class="{0}" cx="{1}" cy="76" r="5"/><text class="t" x="{2}" y="80" font-size="12">{3}</text>' -f $l[0], ($lx + 5), ($lx + 16), $l[1])
    $lx += 30 + 6.4 * $l[1].Length
}

function Panel($idx, $title, $unit, $ymin, $ymax, $ystep, $lines) {
    $y0 = $top + $idx * ($ph + $gap)
    $yy = { param($v) $y0 + $ph - $ph * ($v - $ymin) / ($ymax - $ymin) }
    Add ('<text class="t" x="{0}" y="{1}" font-size="13" font-weight="600">{2} <tspan class="t2" font-weight="400">({3})</tspan></text>' -f $ml, ($y0 - 12), $title, $unit)
    for ($v = $ymin; $v -le $ymax + 1e-9; $v += $ystep) {
        $y = & $yy $v
        Add ('<line class="grid" x1="{0}" x2="{1}" y1="{2:F1}" y2="{2:F1}"/><text class="t2" x="{3}" y="{4:F1}" font-size="11" text-anchor="end">{5}</text>' -f $ml, ($ml + $pw), $y, ($ml - 8), ($y + 4), $v)
    }
    foreach ($p in $phases) {
        if ($p[3] -eq 'start') { $xx = X (Min $p[1]); Add ('<line class="sep" x1="{0:F1}" x2="{0:F1}" y1="{1}" y2="{2}"/>' -f $xx, $y0, ($y0 + $ph)) }
    }
    Add ('<line class="axis" x1="{0}" x2="{0}" y1="{1}" y2="{2}"/><line class="axis" x1="{0}" x2="{3}" y1="{2}" y2="{2}"/>' -f $ml, $y0, ($y0 + $ph), ($ml + $pw))
    $labels = @()
    foreach ($ln in $lines) {
        $pts = @($ln.pts)
        if ($pts.Count -eq 0) { continue }
        $path = New-Object Text.StringBuilder
        $prev = $null
        foreach ($p in $pts) {
            $cmd = if ($null -eq $prev -or ($p.x - $prev.x) -gt $ln.gap) { 'M' } else { 'L' }
            $null = $path.AppendFormat($inv, '{0}{1:F1},{2:F1} ', $cmd, (X $p.x), (& $yy $p.y))
            $prev = $p
        }
        $dash = if ($ln.dash) { ' dash' } else { '' }
        Add ('<path class="ln {0}{1}" d="{2}"/>' -f $ln.cls, $dash, $path.ToString().Trim())
        if ($ln.markers) {
            foreach ($p in $pts) {
                $tip = '{0}: {1:F1} {2} at {3:F1} min' -f $ln.name, $p.y, $unit, $p.x
                Add ('<circle class="mk {0}" cx="{1:F1}" cy="{2:F1}" r="4"><title>{3}</title></circle>' -f $ln.cls, (X $p.x), (& $yy $p.y), $tip)
            }
        }
        $labels += [pscustomobject]@{ y = (& $yy $pts[-1].y); text = $ln.label }
    }
    # End labels, pushed apart so they never overlap.
    $placed = @()
    foreach ($l in ($labels | Sort-Object y)) {
        $y = $l.y
        if ($placed.Count -gt 0 -and ($y - $placed[-1].y) -lt 14) { $y = $placed[-1].y + 14 }
        $placed += [pscustomobject]@{ y = $y; text = $l.text }
    }
    foreach ($l in $placed) { Add ('<text class="t" x="{0}" y="{1:F1}" font-size="11">{2}</text>' -f ($ml + $pw + 8), ($l.y + 4), $l.text) }
}

Panel 0 'Prefill throughput' 'tok/s' 0 (Ymax @($S.prefill_ours, $S.prefill_llama) 100) 100 @(
    @{ pts = $S.prefill_ours; cls = 's1'; name = 'this engine'; label = 'ours'; markers = $true; gap = 0.5 },
    @{ pts = $S.prefill_llama; cls = 's2'; name = 'llama.cpp'; label = 'llama.cpp'; markers = $true; gap = 0.5 })
Panel 1 'Decode throughput' 'tok/s' 0 (Ymax @($S.spec_ours, $S.plain_ours, $S.decode_llama) 5) 5 @(
    @{ pts = $S.spec_ours; cls = 's3'; name = 'this engine, MTP'; label = 'ours (MTP)'; markers = $true; gap = 0.5 },
    @{ pts = $S.plain_ours; cls = 's1'; name = 'this engine, plain'; label = 'ours (plain)'; markers = $true; gap = 0.5 },
    @{ pts = $S.decode_llama; cls = 's2'; name = 'llama.cpp'; label = 'llama.cpp'; markers = $true; gap = 0.5 })
$tmin = [math]::Floor((($S.gfx_c + $S.soc_c) | Measure-Object y -Minimum).Minimum / 10) * 10
Panel 2 'Temperature' '&#176;C' $tmin (Ymax @($S.gfx_c, $S.soc_c) 10) 10 @(
    @{ pts = $S.gfx_c; cls = 'g'; name = 'GFX'; label = 'GFX'; markers = $false; gap = 0.2 },
    @{ pts = $S.soc_c; cls = 'g'; name = 'SoC'; label = 'SoC'; markers = $false; gap = 0.2; dash = $true })
Panel 3 'GFX clock' 'MHz' 0 (Ymax @($S.clk) 500) 500 @(@{ pts = $S.clk; cls = 'g'; name = 'GFX clock'; label = 'GFX clock'; markers = $false; gap = 0.2 })
Panel 4 'ASIC power' 'W' 0 (Ymax @($S.pwr) 10) 10 @(@{ pts = $S.pwr; cls = 'g'; name = 'ASIC power'; label = 'ASIC power'; markers = $false; gap = 0.2 })

# Strip under the last panel: who was running when.
$ys = $top + ($np - 1) * ($ph + $gap) + $ph + 16
$open = @{}
foreach ($p in $phases) {
    if ($p[3] -eq 'start') { $open[$p[2]] = Min $p[1] }
    elseif ($open.ContainsKey($p[2])) {
        $x1 = X $open[$p[2]]; $x2 = X (Min $p[1])
        $cls = if ($p[2] -eq 'ours') { 'band1' } else { 'band2' }
        $nm = if ($p[2] -eq 'ours') { 'this engine' } else { 'llama.cpp' }
        Add ('<rect class="{0}" x="{1:F1}" y="{2}" width="{3:F1}" height="12" rx="2"><title>{4}</title></rect>' -f $cls, $x1, $ys, [math]::Max(1, $x2 - $x1 - 2), $nm)
        $open.Remove($p[2])
    }
}
for ($m = 0; $m -le $xmax; $m += [math]::Max(1, [math]::Ceiling($xmax / 12))) {
    Add ('<text class="t2" x="{0:F1}" y="{1}" font-size="11" text-anchor="middle">{2}</text>' -f (X $m), ($ys + 30), $m)
}
Add ('<text class="t2" x="{0}" y="{1}" font-size="12" text-anchor="end">minutes since start</text>' -f ($ml + $pw), ($ys + 52))
Add '</svg>'
[IO.File]::WriteAllText($Out, $sb.ToString(), (New-Object Text.UTF8Encoding($false)))
Write-Host "wrote $Out"

# Summary per slice.
function Avg($rows, $a, $b, $ti, $vi) {
    $v = $rows | Where-Object { (D $_[$ti]) -ge $a -and (D $_[$ti]) -le $b } | ForEach-Object { D $_[$vi] }
    if (@($v).Count) { '{0:F1}' -f (($v | Measure-Object -Average).Average) } else { '-' }
}
Write-Host ('{0,-5} {1,-9} {2,8} {3,8} {4,8} {5,9} {6,8} {7,8}' -f 'round', 'engine', 'prefill', 'decode', 'spec', 'gfx MHz', 'power W', 'gfx C')
$round = @{ ours = 0; llama = 0 }
$open2 = @{}
foreach ($p in $phases) {
    if ($p[3] -eq 'start') { $open2[$p[2]] = D $p[1]; $round[$p[2]]++ }
    elseif ($open2.ContainsKey($p[2])) {
        $a = $open2[$p[2]]; $b = D $p[1]
        if ($p[2] -eq 'ours') { $rows = $ours; $pre = 2; $dec = 3; $spc = 4 } else { $rows = $llama; $pre = 2; $dec = 3; $spc = -1 }
        $spec = if ($spc -ge 0) { Avg $rows $a $b 1 $spc } else { '-' }
        Write-Host ('{0,-5} {1,-9} {2,8} {3,8} {4,8} {5,9} {6,8} {7,8}' -f $round[$p[2]], $p[2], (Avg $rows $a $b 1 $pre), (Avg $rows $a $b 1 $dec), $spec, (Avg $tel $a $b 0 4), (Avg $tel $a $b 0 6), (Avg $tel $a $b 0 1))
        $open2.Remove($p[2])
    }
}
