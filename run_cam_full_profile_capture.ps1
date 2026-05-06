param(
  [ValidateSet("stable", "maneuver", "recovery")]
  [string]$Scenario = "stable",
  [int]$MaxRuntimeMs = 0,
  [string]$Config = "config\tracker_train_full.yaml",
  [string]$LogsDir = "outputs\training_full\logs"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$pythonExe = "C:\Users\Administrator\miniconda3\envs\py310\python.exe"
$env:PATH = "C:\Users\Administrator\miniconda3\envs\py310\Library\bin;$env:PATH"

$moduleDir = $null
foreach ($dir in Get-ChildItem -LiteralPath $projectRoot -Directory) {
  $candidate = Join-Path $dir.FullName "build_pseudo_expert_batch.py"
  if (Test-Path -LiteralPath $candidate) {
    $moduleDir = $dir.FullName
    break
  }
}
if (-not $moduleDir) {
  throw "Failed to locate module directory under project root."
}

$scriptBuildPseudo = Join-Path $moduleDir "build_pseudo_expert_batch.py"
$scriptAudit = Join-Path $moduleDir "audit_augmented_dataset.py"
$scriptProgress = Join-Path $moduleDir "refresh_full_collection_progress.py"
$scriptCompare = Join-Path $moduleDir "compare_pseudo_expert_variants.py"
$scriptByScenario = Join-Path $moduleDir "summarize_full_collection_progress_by_scenario.py"
$scriptFailure = Join-Path $moduleDir "build_scenario_failure_breakdown.py"
$scriptTargetProfile = Join-Path $moduleDir "build_target_presentation_success_profile.py"
$scriptCaptureRate = Join-Path $moduleDir "build_scenario_capture_success_rate.py"
$scriptRepeatStatus = Join-Path $moduleDir "build_scenario_repeat_coverage_status.py"
$scriptTaskBoard = Join-Path $moduleDir "build_next_capture_tasks.py"
$scriptManeuverAnalysis = Join-Path $moduleDir "build_maneuver_success_template_analysis.py"
$scriptPaperBlocks = Join-Path $moduleDir "build_paper_results_blocks.py"
$scriptReadiness = Join-Path $moduleDir "refresh_training_readiness.py"
$scriptContrast = Join-Path $moduleDir "build_target_presentation_contrast_report.py"
$scriptGuidance = Join-Path $moduleDir "build_scenario_capture_guidance.py"
$guidanceCsv = "outputs\training_full\scenario_capture_guidance.csv"
$guidanceMd = "outputs\training_full\scenario_capture_guidance.md"
$repeatStatusCsv = "outputs\training_full\scenario_repeat_coverage_status.csv"
$repeatStatusMd = "outputs\training_full\scenario_repeat_coverage_status.md"
$taskBoardCsv = "outputs\training_full\next_capture_tasks.csv"
$taskBoardMd = "outputs\training_full\next_capture_tasks.md"
$maneuverAnalysisCsv = "outputs\training_full\maneuver_success_template_analysis.csv"
$maneuverAnalysisMd = "outputs\training_full\maneuver_success_template_analysis.md"

function Refresh-CaptureGuidance {
  if (
    (Test-Path -LiteralPath "outputs\training_full\full_collection_progress.csv") -and
    (Test-Path -LiteralPath "outputs\training_full\full_collection_progress_by_scenario.csv") -and
    (Test-Path -LiteralPath "outputs\training_full\target_presentation_success_profile.csv") -and
    (Test-Path -LiteralPath "outputs\training_full\scenario_capture_success_rate.csv")
  ) {
    & $pythonExe $scriptGuidance `
      --progress_csv "outputs\training_full\full_collection_progress.csv" `
      --scenario_summary_csv "outputs\training_full\full_collection_progress_by_scenario.csv" `
      --profile_csv "outputs\training_full\target_presentation_success_profile.csv" `
      --success_rate_csv "outputs\training_full\scenario_capture_success_rate.csv" `
      --output_csv $guidanceCsv `
      --output_md $guidanceMd
  }
}

function Refresh-RepeatCoverageOutputs {
  if (
    (Test-Path -LiteralPath "outputs\training_full\full_collection_progress.csv") -and
    (Test-Path -LiteralPath "outputs\training_full\full_collection_progress_by_scenario.csv") -and
    (Test-Path -LiteralPath "outputs\training_full\scenario_capture_success_rate.csv") -and
    (Test-Path -LiteralPath $guidanceCsv) -and
    (Test-Path -LiteralPath "outputs\training_full\target_presentation_success_profile.csv")
  ) {
    & $pythonExe $scriptRepeatStatus `
      --scenario_summary_csv "outputs\training_full\full_collection_progress_by_scenario.csv" `
      --success_rate_csv "outputs\training_full\scenario_capture_success_rate.csv" `
      --output_csv $repeatStatusCsv `
      --output_md $repeatStatusMd

    & $pythonExe $scriptTaskBoard `
      --progress_csv "outputs\training_full\full_collection_progress.csv" `
      --guidance_csv $guidanceCsv `
      --repeat_status_csv $repeatStatusCsv `
      --output_csv $taskBoardCsv `
      --output_md $taskBoardMd

    & $pythonExe $scriptManeuverAnalysis `
      --progress_csv "outputs\training_full\full_collection_progress.csv" `
      --profile_csv "outputs\training_full\target_presentation_success_profile.csv" `
      --output_csv $maneuverAnalysisCsv `
      --output_md $maneuverAnalysisMd

    & $pythonExe $scriptPaperBlocks `
      --scenario_summary_csv "outputs\training_full\full_collection_progress_by_scenario.csv" `
      --repeat_status_csv $repeatStatusCsv `
      --success_rate_csv "outputs\training_full\scenario_capture_success_rate.csv" `
      --dataset_audit_json "outputs\training_full\dataset_audit.json" `
      --registry_csv "outputs\training_full\training_experiment_registry.csv" `
      --maneuver_analysis_md $maneuverAnalysisMd `
      --paper_dir "paper"
  }
}

function Show-CaptureGuidance {
  param(
    [string]$ScenarioName
  )

  if (!(Test-Path -LiteralPath $guidanceCsv)) {
    return
  }

  $row = Import-Csv -LiteralPath $guidanceCsv | Where-Object { $_.scenario -eq $ScenarioName } | Select-Object -First 1
  if (-not $row) {
    return
  }

  Write-Host (
    "[Guidance] {0}: success {1}/{2}, valid rows {3}, usable sequences {4}, remaining hold-run gap {5}" -f
    $row.scenario,
    $row.successful_runs,
    $row.attempted_runs,
    $row.current_pseudo_valid_rows,
    $row.current_usable_sequences,
    $row.gap_recovery_hold_runs
  ) -ForegroundColor DarkCyan

  if ([double]$row.observed_bbox_area_mean_success -gt 0) {
    Write-Host (
      "[Guidance] template det_conf_mean~{0:N3}, bbox_area_mean~{1:N1}, bbox_area_p90~{2:N1}, tracked_hold~{3:N0} ms" -f
      [double]$row.observed_det_conf_mean_success,
      [double]$row.observed_bbox_area_mean_success,
      [double]$row.observed_bbox_area_p90_success,
      [double]$row.observed_reacquire_tracked_hold_ms_success
    ) -ForegroundColor DarkCyan
  }

  Write-Host ("[Next target] {0}" -f $row.next_capture_target) -ForegroundColor Yellow
}

function Show-RepeatStatus {
  param(
    [string]$ScenarioName
  )

  if (!(Test-Path -LiteralPath $repeatStatusCsv)) {
    return
  }

  $row = Import-Csv -LiteralPath $repeatStatusCsv | Where-Object { $_.scenario -eq $ScenarioName } | Select-Object -First 1
  if (-not $row) {
    return
  }

  Write-Host (
    "[Repeat status] {0}: stage={1}, successful_runs {2}/{3}, hold_runs {4}/{5}" -f
    $row.scenario,
    $row.coverage_stage,
    $row.successful_runs_current,
    $row.successful_runs_target,
    $row.meets_hold_runs_current,
    $row.meets_hold_runs_target
  ) -ForegroundColor DarkMagenta
}

function Show-NextCaptureTasks {
  if (!(Test-Path -LiteralPath $taskBoardCsv)) {
    return
  }

  $tasks = Import-Csv -LiteralPath $taskBoardCsv | Select-Object -First 3
  foreach ($task in $tasks) {
    Write-Host ("[Task] {0}: {1}" -f $task.task_id, $task.objective) -ForegroundColor DarkYellow
  }
}

if ($MaxRuntimeMs -le 0) {
  if ($Scenario -eq "recovery") {
    $MaxRuntimeMs = 35000
  } else {
    $MaxRuntimeMs = 45000
  }
}

$logsDirAbs = Join-Path $projectRoot $LogsDir
New-Item -ItemType Directory -Path $logsDirAbs -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tempLog = Join-Path $projectRoot "outputs\training_full\tracker_log.csv"
$archiveLog = Join-Path $logsDirAbs ("tracker_log_full_{0}_{1}.csv" -f $Scenario, $timestamp)

Write-Host "[Capture] full-profile $Scenario session" -ForegroundColor Cyan
if ($Scenario -eq "stable") {
  Write-Host "[Protocol] Keep target stably visible near image center for 45 s." -ForegroundColor Yellow
  Write-Host "[Priority note] stable is calibration-only. New evidence collection should prioritize recovery, then maneuver." -ForegroundColor Yellow
} elseif ($Scenario -eq "maneuver") {
  Write-Host "[Protocol] Use the maneuver_collection_protocol.md checklist; keep target in-frame throughout the maneuver phases." -ForegroundColor Yellow
} else {
  Write-Host "[Protocol] Use the recovery_collection_protocol.md checklist; stable tracking -> controlled exit -> center-near re-entry -> hold." -ForegroundColor Yellow
}
Write-Host "[Safe boundary] mock actuator, armed=false." -ForegroundColor Yellow
Refresh-CaptureGuidance
Refresh-RepeatCoverageOutputs
Show-NextCaptureTasks
Show-CaptureGuidance -ScenarioName $Scenario
Show-RepeatStatus -ScenarioName $Scenario

if (Test-Path -LiteralPath $tempLog) {
  Remove-Item -LiteralPath $tempLog -Force
}

& ".\build\msvc-opencv-release\tracker.exe" --mode cam --config $Config --max_runtime_ms $MaxRuntimeMs

if (!(Test-Path -LiteralPath $tempLog)) {
  throw "Expected full-profile log not found: $tempLog"
}

$tempLogInfo = Get-Item -LiteralPath $tempLog
if ($tempLogInfo.Length -le 0) {
  Remove-Item -LiteralPath $tempLog -Force
  throw "Capture produced an empty log. Tracker/camera startup likely failed; archived outputs were not updated."
}

Copy-Item -LiteralPath $tempLog -Destination $archiveLog -Force

& $pythonExe $scriptBuildPseudo `
  --input_dir $logsDirAbs `
  --output_dir "outputs\training_full\batch_augmented" `
  --summary_csv "outputs\training_full\batch_augmented_summary.csv" `
  --pattern "tracker_log_full_*.csv" `
  --variants "future_smoothed_base,future_error_aware" `
  --future_horizon 5 `
  --min_future_len 3 `
  --dedup_by_frame_id

& $pythonExe $scriptAudit `
  --summary_csv "outputs\training_full\batch_augmented_summary.csv" `
  --report_json "outputs\training_full\dataset_audit.json" `
  --report_txt "outputs\training_full\dataset_audit.txt" `
  --seq_len 8

& $pythonExe $scriptProgress `
  --summary_csv "outputs\training_full\batch_augmented_summary.csv" `
  --output_csv "outputs\training_full\full_collection_progress.csv" `
  --seq_len 8 `
  --hold_ms 200

& $pythonExe $scriptCompare `
  --summary_csv "outputs\training_full\batch_augmented_summary.csv" `
  --output_csv "outputs\training_full\pseudo_expert_variant_comparison.csv" `
  --seq_len 8

& $pythonExe $scriptByScenario `
  --progress_csv "outputs\training_full\full_collection_progress.csv" `
  --output_csv "outputs\training_full\full_collection_progress_by_scenario.csv" `
  --target_full_logs 20 `
  --target_pseudo_valid_rows 3000 `
  --target_usable_sequences 500 `
  --per_scenario_target_valid_rows 800 `
  --per_scenario_target_usable_sequences 120 `
  --recovery_hold_target_runs 3

& $pythonExe $scriptFailure `
  --summary_csv "outputs\training_full\batch_augmented_summary.csv" `
  --output_csv "outputs\training_full\scenario_failure_breakdown.csv" `
  --hold_ms 200

& $pythonExe $scriptTargetProfile `
  --progress_csv "outputs\training_full\full_collection_progress.csv" `
  --output_csv "outputs\training_full\target_presentation_success_profile.csv" `
  --center_margin_x 60 `
  --center_margin_y 40

& $pythonExe $scriptContrast `
  --profile_csv "outputs\training_full\target_presentation_success_profile.csv" `
  --output_csv "outputs\training_full\target_presentation_contrast_report.csv"

& $pythonExe $scriptCaptureRate `
  --failure_breakdown_csv "outputs\training_full\scenario_failure_breakdown.csv" `
  --output_csv "outputs\training_full\scenario_capture_success_rate.csv"

& $pythonExe $scriptGuidance `
  --progress_csv "outputs\training_full\full_collection_progress.csv" `
  --scenario_summary_csv "outputs\training_full\full_collection_progress_by_scenario.csv" `
  --profile_csv "outputs\training_full\target_presentation_success_profile.csv" `
  --success_rate_csv "outputs\training_full\scenario_capture_success_rate.csv" `
  --output_csv $guidanceCsv `
  --output_md $guidanceMd

& $pythonExe $scriptRepeatStatus `
  --scenario_summary_csv "outputs\training_full\full_collection_progress_by_scenario.csv" `
  --success_rate_csv "outputs\training_full\scenario_capture_success_rate.csv" `
  --output_csv $repeatStatusCsv `
  --output_md $repeatStatusMd

& $pythonExe $scriptTaskBoard `
  --progress_csv "outputs\training_full\full_collection_progress.csv" `
  --guidance_csv $guidanceCsv `
  --repeat_status_csv $repeatStatusCsv `
  --output_csv $taskBoardCsv `
  --output_md $taskBoardMd

& $pythonExe $scriptManeuverAnalysis `
  --progress_csv "outputs\training_full\full_collection_progress.csv" `
  --profile_csv "outputs\training_full\target_presentation_success_profile.csv" `
  --output_csv $maneuverAnalysisCsv `
  --output_md $maneuverAnalysisMd

& $pythonExe $scriptPaperBlocks `
  --scenario_summary_csv "outputs\training_full\full_collection_progress_by_scenario.csv" `
  --repeat_status_csv $repeatStatusCsv `
  --success_rate_csv "outputs\training_full\scenario_capture_success_rate.csv" `
  --dataset_audit_json "outputs\training_full\dataset_audit.json" `
  --registry_csv "outputs\training_full\training_experiment_registry.csv" `
  --maneuver_analysis_md $maneuverAnalysisMd `
  --paper_dir "paper"

& $pythonExe $scriptReadiness `
  --registry_csv "outputs\training_full\training_experiment_registry.csv" `
  --dataset_audit_json "outputs\training_full\dataset_audit.json" `
  --scenario_summary_csv "outputs\training_full\full_collection_progress_by_scenario.csv"

Write-Host "[OK] archived log: $archiveLog" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\full_collection_progress.csv" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\full_collection_progress_by_scenario.csv" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\scenario_failure_breakdown.csv" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\target_presentation_success_profile.csv" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\target_presentation_contrast_report.csv" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\scenario_capture_success_rate.csv" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\scenario_capture_guidance.csv / scenario_capture_guidance.md" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\scenario_repeat_coverage_status.csv / scenario_repeat_coverage_status.md" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\next_capture_tasks.csv / next_capture_tasks.md" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\maneuver_success_template_analysis.csv / maneuver_success_template_analysis.md" -ForegroundColor Green
Write-Host "[OK] refreshed paper result blocks and submission-boundary notes" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\pseudo_expert_variant_comparison.csv" -ForegroundColor Green
Write-Host "[OK] refreshed outputs\\training_full\\dataset_audit.txt / dataset_audit.json" -ForegroundColor Green
Show-NextCaptureTasks
Show-CaptureGuidance -ScenarioName $Scenario
Show-RepeatStatus -ScenarioName $Scenario
