<#
Caption videos by dropping them onto caption-drop.cmd.

Settings live in %LOCALAPPDATA%\publikclip\caption-drop.json and are asked for
once, on the first run. Run with no files to change them, or -Setup to force
the prompts.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Files,
    [switch] $Setup,
    [switch] $NoPause
)

$ErrorActionPreference = 'Stop'
$ConfigPath = Join-Path $env:LOCALAPPDATA 'publikclip\caption-drop.json'
$VideoExt = @('.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.ts', '.flv', '.wmv')
$Presets = @('classic', 'beast', 'hormozi', 'minimal', 'karaoke-pop')

function Wait-Close {
    <# The window is spawned by a double-click or a drop, so it closes the
       instant the script ends -- the pause is the only way to read the result.
       Guarded because a non-interactive host has no stdin to read and would
       otherwise fail here, after the work already succeeded. #>
    if ($NoPause) { return }
    try { Read-Host '  Press Enter to close' | Out-Null } catch { }
}

function Read-Config {
    if (Test-Path $ConfigPath) {
        try { return Get-Content $ConfigPath -Raw | ConvertFrom-Json } catch { }
    }
    return $null
}

function Write-Config($outDir, $preset, $tags) {
    $dir = Split-Path $ConfigPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [pscustomobject]@{ output = $outDir; preset = $preset; tags = [bool]$tags } |
        ConvertTo-Json | Out-File -FilePath $ConfigPath -Encoding utf8
}

function Select-OutputFolder($current) {
    # The native folder picker, so this is a click rather than typing a path.
    Add-Type -AssemblyName System.Windows.Forms
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = 'Where should captioned videos be saved?'
    if ($current -and (Test-Path $current)) { $dlg.SelectedPath = $current }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dlg.SelectedPath }
    return $null
}

function Invoke-Setup($cfg) {
    Write-Host ''
    Write-Host '  publikclip caption -- setup' -ForegroundColor Cyan
    Write-Host ''
    $outDir = Select-OutputFolder $cfg.output
    if (-not $outDir) { Write-Host '  Cancelled; nothing changed.'; return $null }

    Write-Host "  Output folder: $outDir"
    Write-Host ''
    Write-Host '  Caption style:'
    for ($i = 0; $i -lt $Presets.Count; $i++) { Write-Host "    [$($i+1)] $($Presets[$i])" }
    $default = if ($cfg.preset) { $cfg.preset } else { 'classic' }
    $answer = Read-Host "  Choose 1-$($Presets.Count) (enter for $default)"
    $preset = $default
    if ($answer -match '^\d+$' -and [int]$answer -ge 1 -and [int]$answer -le $Presets.Count) {
        $preset = $Presets[[int]$answer - 1]
    }

    $tagsAnswer = Read-Host '  Detect [laughs] tags too? Slower. y/N'
    $tags = $tagsAnswer -match '^(y|yes)$'

    Write-Config $outDir $preset $tags
    Write-Host ''
    Write-Host "  Saved. Style '$preset', tags $(if ($tags) { 'on' } else { 'off' })." -ForegroundColor Green
    Write-Host "  Settings: $ConfigPath"
    return (Read-Config)
}

function Resolve-Publikclip {
    <# Prefer the installed app's bundled environment; fall back to the repo
       checkout this script sits in, so it works before an install too. #>
    $res = Join-Path $env:LOCALAPPDATA 'publikclip\resources'
    $uv = Join-Path $res 'bin\uv.exe'
    if ((Test-Path $uv) -and (Test-Path (Join-Path $res 'pipeline'))) {
        return @{ Exe = $uv; Args = @('--directory', (Join-Path $res 'pipeline'), 'run', 'publikclip') }
    }
    $repoPipeline = Join-Path (Split-Path $PSScriptRoot -Parent) 'pipeline'
    if (Test-Path $repoPipeline) {
        $cmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($cmd) {
            return @{ Exe = $cmd.Source; Args = @('--directory', $repoPipeline, 'run', 'publikclip') }
        }
    }
    return $null
}

# --- main -------------------------------------------------------------------

$cfg = Read-Config
if ($Setup -or -not $cfg -or -not $cfg.output) {
    $cfg = Invoke-Setup $cfg
    if (-not $cfg) { Wait-Close; exit 1 }
}

$videos = @()
foreach ($f in ($Files | Where-Object { $_ })) {
    if (-not (Test-Path $f)) { Write-Host "  skipped (not found): $f" -ForegroundColor Yellow; continue }
    $item = Get-Item $f
    if ($item.PSIsContainer) { Write-Host "  skipped (folder): $($item.Name)" -ForegroundColor Yellow; continue }
    if ($VideoExt -notcontains $item.Extension.ToLower()) {
        Write-Host "  skipped (not a video): $($item.Name)" -ForegroundColor Yellow; continue
    }
    $videos += $item
}

if ($videos.Count -eq 0) {
    Write-Host ''
    Write-Host '  Drag video files onto caption-drop.cmd to caption them.' -ForegroundColor Cyan
    Write-Host "  Output folder : $($cfg.output)"
    Write-Host "  Style         : $($cfg.preset)"
    Write-Host "  [laughs] tags : $(if ($cfg.tags) { 'on' } else { 'off' })"
    Write-Host ''
    Write-Host '  Run this file with no videos to change those settings.'
    Wait-Close
    exit 0
}

$runner = Resolve-Publikclip
if (-not $runner) {
    Write-Host '  Could not find publikclip. Install the app, or install uv and run from the repo.' -ForegroundColor Red
    Wait-Close; exit 1
}
if (-not (Test-Path $cfg.output)) { New-Item -ItemType Directory -Force -Path $cfg.output | Out-Null }

$ok = 0; $failed = @()
for ($i = 0; $i -lt $videos.Count; $i++) {
    $v = $videos[$i]
    $dest = Join-Path $cfg.output ($v.BaseName + '.captioned.mp4')
    Write-Host ''
    Write-Host "  [$($i+1)/$($videos.Count)] $($v.Name)" -ForegroundColor Cyan
    Write-Host "      -> $dest"

    $callArgs = $runner.Args + @('caption', $v.FullName, '--preset', $cfg.preset, '-o', $dest)
    if ($cfg.tags) { $callArgs += '--tags' }
    & $runner.Exe @callArgs
    if ($LASTEXITCODE -eq 0) {
        $ok++
        Write-Host "      done" -ForegroundColor Green
    } else {
        $failed += $v.Name
        Write-Host "      FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host ''
Write-Host "  $ok of $($videos.Count) captioned into $($cfg.output)" -ForegroundColor $(if ($failed) { 'Yellow' } else { 'Green' })
foreach ($f in $failed) { Write-Host "    failed: $f" -ForegroundColor Red }
Wait-Close
