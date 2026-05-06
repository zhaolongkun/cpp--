$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
  param([string]$Message)
  Write-Host "[Step] $Message" -ForegroundColor Cyan
}

function Fail {
  param([string]$Message)
  throw "[Error] $Message"
}

function Require-Path {
  param(
    [string]$Path,
    [string]$Label
  )
  if (-not (Test-Path -LiteralPath $Path)) {
    Fail "$Label not found: $Path"
  }
}

function Convert-ToIntSafe {
  param([object]$Value)
  if ($null -eq $Value) { return 0 }
  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text)) { return 0 }
  $result = 0
  if ([int]::TryParse($text, [ref]$result)) {
    return $result
  }
  return 0
}

function Convert-ToDoubleSafe {
  param([object]$Value)
  if ($null -eq $Value) { return 0.0 }
  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text)) { return 0.0 }
  $result = 0.0
  if ([double]::TryParse($text, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$result)) {
    return $result
  }
  if ([double]::TryParse($text, [ref]$result)) {
    return $result
  }
  return 0.0
}

function Sync-FileIfNeeded {
  param(
    [string]$SourcePath,
    [string]$TargetPath
  )

  Require-Path -Path $SourcePath -Label "Source artifact"

  $copyRequired = $true
  if (Test-Path -LiteralPath $TargetPath) {
    $srcItem = Get-Item -LiteralPath $SourcePath
    $dstItem = Get-Item -LiteralPath $TargetPath
    if ($srcItem.Length -eq $dstItem.Length -and $srcItem.LastWriteTimeUtc -eq $dstItem.LastWriteTimeUtc) {
      $copyRequired = $false
    }
  }

  if ($copyRequired) {
    Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force
  }
}

function Find-ArtifactSourceDir {
  param(
    [string]$ProjectRoot,
    [string]$ArtifactRun
  )

  $candidateDirs = @()

  $projectArtifactDir = Join-Path (Join-Path $ProjectRoot "artifacts") $ArtifactRun
  $candidateDirs += $projectArtifactDir

  $candidateDirs += Get-ChildItem -LiteralPath $ProjectRoot -Directory | ForEach-Object {
    Join-Path $_.FullName ("artifacts\" + $ArtifactRun)
  }

  foreach ($dir in $candidateDirs) {
    $modelPath = Join-Path $dir "model.onnx"
    $scalerPath = Join-Path $dir "scaler.json"
    $specPath = Join-Path $dir "feature_spec.json"
    if ((Test-Path -LiteralPath $modelPath) -and (Test-Path -LiteralPath $scalerPath) -and (Test-Path -LiteralPath $specPath)) {
      return $dir
    }
  }

  Fail "Could not find artifacts\$ArtifactRun under any first-level directory of $ProjectRoot"
}

function Parse-Args {
  param([string[]]$ArgList)

  $result = [ordered]@{
    Mode = "replay"
    Config = ""
    ReplayCsv = ""
    LogCsv = ""
    ArtifactRun = "train_run1"
    MaxRuntimeMs = 0
  }

  $i = 0
  while ($i -lt $ArgList.Count) {
    $arg = $ArgList[$i]
    switch ($arg) {
      "--mode" {
        $i++
        if ($i -ge $ArgList.Count) { Fail "Missing value for --mode" }
        $result.Mode = $ArgList[$i]
      }
      "--config" {
        $i++
        if ($i -ge $ArgList.Count) { Fail "Missing value for --config" }
        $result.Config = $ArgList[$i]
      }
      "--replay_csv" {
        $i++
        if ($i -ge $ArgList.Count) { Fail "Missing value for --replay_csv" }
        $result.ReplayCsv = $ArgList[$i]
      }
      "--log_csv" {
        $i++
        if ($i -ge $ArgList.Count) { Fail "Missing value for --log_csv" }
        $result.LogCsv = $ArgList[$i]
      }
      "--artifact_run" {
        $i++
        if ($i -ge $ArgList.Count) { Fail "Missing value for --artifact_run" }
        $result.ArtifactRun = $ArgList[$i]
      }
      "--max_runtime_ms" {
        $i++
        if ($i -ge $ArgList.Count) { Fail "Missing value for --max_runtime_ms" }
        $result.MaxRuntimeMs = [int]$ArgList[$i]
      }
      default {
        Fail "Unknown argument: $arg"
      }
    }
    $i++
  }

  return $result
}

$opts = Parse-Args -ArgList $args
if ($opts.Mode -notin @("replay", "cam")) {
  Fail "Unsupported mode: $($opts.Mode). Use --mode replay or --mode cam."
}

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$workspaceRoot = Split-Path (Split-Path $projectRoot -Parent) -Parent
$asciiRoot = Join-Path $workspaceRoot "cpp_control_ascii"
$exeDir = Join-Path $asciiRoot "build\nmake-msvc-onnx-release"
$exePath = Join-Path $exeDir "tracker.exe"
$opencvBin = "C:\Users\Administrator\miniconda3\envs\py310\Library\bin"

$configPath = if ($opts.Config) { $opts.Config } else { Join-Path $projectRoot "config\tracker.yaml" }
$replayCsvPath = if ($opts.ReplayCsv) { $opts.ReplayCsv } else { Join-Path $projectRoot "data\detections.csv" }
$logCsvPath = if ($opts.LogCsv) { $opts.LogCsv } else { Join-Path $projectRoot "logs\tracker_log.csv" }

$artifactSrcDir = Find-ArtifactSourceDir -ProjectRoot $projectRoot -ArtifactRun $opts.ArtifactRun
$artifactDstDir = Join-Path $asciiRoot ("artifacts_ascii\" + $opts.ArtifactRun)
$modelSrc = Join-Path $artifactSrcDir "model.onnx"
$scalerSrc = Join-Path $artifactSrcDir "scaler.json"
$specSrc = Join-Path $artifactSrcDir "feature_spec.json"
$modelDst = Join-Path $artifactDstDir "model.onnx"
$scalerDst = Join-Path $artifactDstDir "scaler.json"
$specDst = Join-Path $artifactDstDir "feature_spec.json"

Require-Path -Path $asciiRoot -Label "ASCII project root"
Require-Path -Path $exePath -Label "tracker.exe"
Require-Path -Path (Join-Path $exeDir "onnxruntime.dll") -Label "onnxruntime.dll"
Require-Path -Path (Join-Path $exeDir "onnxruntime_providers_shared.dll") -Label "onnxruntime_providers_shared.dll"
Require-Path -Path $opencvBin -Label "OpenCV bin dir"
Require-Path -Path $configPath -Label "Config file"
if ($opts.Mode -eq "replay") {
  Require-Path -Path $replayCsvPath -Label "Replay CSV"
}

Write-Step "Syncing model artifacts to ASCII path"
New-Item -ItemType Directory -Force -Path $artifactDstDir | Out-Null
Sync-FileIfNeeded -SourcePath $modelSrc -TargetPath $modelDst
Sync-FileIfNeeded -SourcePath $scalerSrc -TargetPath $scalerDst
Sync-FileIfNeeded -SourcePath $specSrc -TargetPath $specDst

$env:R2S_MODEL_ONNX = $modelDst
$env:R2S_SCALER_JSON = $scalerDst
$env:R2S_FEATURE_SPEC_JSON = $specDst
$env:PATH = "$opencvBin;$env:PATH"

Write-Step "R2S_MODEL_ONNX=$env:R2S_MODEL_ONNX"
Write-Step "R2S_SCALER_JSON=$env:R2S_SCALER_JSON"
Write-Step "R2S_FEATURE_SPEC_JSON=$env:R2S_FEATURE_SPEC_JSON"

if (Test-Path -LiteralPath $logCsvPath) {
  Remove-Item -LiteralPath $logCsvPath -Force
}

$trackerArgs = @("--mode", $opts.Mode, "--config", $configPath)
if ($opts.Mode -eq "replay") {
  $trackerArgs += @("--replay_csv", $replayCsvPath)
}
if ($opts.MaxRuntimeMs -gt 0) {
  $trackerArgs += @("--max_runtime_ms", [string]$opts.MaxRuntimeMs)
}

Write-Step "Running tracker"
$oldLocation = Get-Location
try {
  Set-Location -LiteralPath $projectRoot
  & $exePath @trackerArgs
  $exitCode = $LASTEXITCODE
}
finally {
  Set-Location -LiteralPath $oldLocation
}

if ($exitCode -ne 0) {
  Fail "tracker.exe failed with exit code $exitCode"
}

Require-Path -Path $logCsvPath -Label "tracker_log.csv"

Write-Step "Running minimal log check"
$rows = Import-Csv -LiteralPath $logCsvPath
$rowCount = $rows.Count
$usedCount = ($rows | Where-Object { (Convert-ToIntSafe $_.infer_used_model) -eq 1 }).Count
$fallbackCount = ($rows | Where-Object { (Convert-ToIntSafe $_.fallback_delta_zero) -eq 1 }).Count

$statusGroups = $rows |
  Group-Object -Property infer_status |
  Sort-Object Count -Descending

$deltaSum = 0.0
$cmdDiffSum = 0.0
foreach ($row in $rows) {
  $dx = [math]::Abs((Convert-ToDoubleSafe $row.delta_cmd_x))
  $dy = [math]::Abs((Convert-ToDoubleSafe $row.delta_cmd_y))
  $deltaSum += ($dx + $dy)

  $sentX = Convert-ToDoubleSafe $row.cmd_sent_x
  $sentY = Convert-ToDoubleSafe $row.cmd_sent_y
  $baseX = Convert-ToDoubleSafe $row.cmd_base_x
  $baseY = Convert-ToDoubleSafe $row.cmd_base_y
  $cmdDiffSum += ([math]::Abs($sentX - $baseX) + [math]::Abs($sentY - $baseY))
}

$meanDelta = if ($rowCount -gt 0) { $deltaSum / $rowCount } else { 0.0 }
$meanCmdDiff = if ($rowCount -gt 0) { $cmdDiffSum / $rowCount } else { 0.0 }

Write-Host ""
Write-Host ("rows={0} used={1} fallback={2}" -f $rowCount, $usedCount, $fallbackCount) -ForegroundColor Green
Write-Host "infer_status distribution:"
foreach ($group in $statusGroups) {
  Write-Host ("  {0}: {1}" -f $group.Name, $group.Count)
}
Write-Host ("meanDelta={0}" -f $meanDelta)
Write-Host ("meanCmdDiff={0}" -f $meanCmdDiff)
Write-Host ""
Write-Host "[Done] Run completed successfully." -ForegroundColor Green
Write-Host "[Info] log: $logCsvPath"
