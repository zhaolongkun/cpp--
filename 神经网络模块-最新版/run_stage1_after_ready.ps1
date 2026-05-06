param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$StepName,
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    Write-Host "==> $StepName"
    & 'C:\Users\Administrator\miniconda3\envs\py310\python.exe' @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $StepName (exit code $LASTEXITCODE)"
    }
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$OutputsDir = Join-Path $Root 'outputs'
$DataDir = Join-Path $Root 'data'
$RunDir = Join-Path $DataDir 'run01'
$RawLogsDir = Join-Path $DataDir 'raw_logs'

$ObsoleteManifest = Join-Path $OutputsDir 'obsolete_stage1_artifacts.json'
$TagSummary = Join-Path $OutputsDir 'stage1_tag_validation_summary.json'
$MergedCsv = Join-Path $DataDir 'stage1_merged_logs.csv'
$DatasetReport = Join-Path $OutputsDir 'stage1_dataset_report.json'
$DatasetNpz = Join-Path $RunDir 'stage1_clean_dataset.npz'
$MetaJson = Join-Path $RunDir 'stage1_clean_meta.json'
$BaselineSummary = Join-Path $OutputsDir 'stage1_baseline_summary.json'
$VisualDir = Join-Path $OutputsDir 'stage1_visuals'

if (-not (Test-Path $ObsoleteManifest)) {
    throw "Missing obsolete manifest: $ObsoleteManifest"
}
if (-not (Test-Path $TagSummary)) {
    throw "Missing tag validation summary: $TagSummary"
}

$TagInfo = Get-Content $TagSummary -Raw | ConvertFrom-Json
if (-not $TagInfo.ready_for_merge) {
    throw ("ready_for_merge is false; usable_counts_by_type=" + (($TagInfo.usable_counts_by_type | ConvertTo-Json -Compress)))
}

if ($DryRun) {
    Write-Host "Dry run passed preflight checks."
    Write-Host "Would execute: merge -> dataset_report -> build_dataset -> dataset_report(refresh) -> baseline -> train/eval tcn_gru -> train/eval dual_state -> visualize"
    exit 0
}

Invoke-PythonStep -StepName 'merge_tracker_logs' -Args @(
    (Join-Path $Root 'scripts\merge_tracker_logs.py'),
    '--input_dir', $RawLogsDir,
    '--output_csv', $MergedCsv
)

Invoke-PythonStep -StepName 'report_stage1_dataset (pre-build)' -Args @(
    (Join-Path $Root 'scripts\report_stage1_dataset.py'),
    '--csv', $MergedCsv,
    '--output_json', $DatasetReport
)

Invoke-PythonStep -StepName 'build_stage1_clean_dataset' -Args @(
    (Join-Path $Root 'scripts\build_stage1_clean_dataset.py'),
    '--input_csv', $MergedCsv,
    '--output_dir', $RunDir,
    '--seq_len', '16',
    '--stride', '1',
    '--algorithm_latency_ms', '8',
    '--control_latency_ms', '16',
    '--actuation_latency_ms', '20'
)

Invoke-PythonStep -StepName 'report_stage1_dataset (with split)' -Args @(
    (Join-Path $Root 'scripts\report_stage1_dataset.py'),
    '--csv', $MergedCsv,
    '--dataset_npz', $DatasetNpz,
    '--output_json', $DatasetReport
)

Invoke-PythonStep -StepName 'eval_stage1_baselines' -Args @(
    (Join-Path $Root 'scripts\eval_stage1_baselines.py'),
    '--csv', $MergedCsv,
    '--meta_json', $MetaJson,
    '--output_json', $BaselineSummary
)

Invoke-PythonStep -StepName 'train_stage1_tcn_gru' -Args @(
    (Join-Path $Root 'train_stage1_clean.py'),
    '--config', (Join-Path $Root 'configs\stage1_tcn_gru.yaml')
)

Invoke-PythonStep -StepName 'eval_stage1_tcn_gru' -Args @(
    (Join-Path $Root 'eval_stage1_clean.py'),
    '--config', (Join-Path $Root 'configs\stage1_tcn_gru.yaml')
)

Invoke-PythonStep -StepName 'train_stage1_dual_state' -Args @(
    (Join-Path $Root 'train_stage1_clean.py'),
    '--config', (Join-Path $Root 'configs\stage1_clean.yaml')
)

Invoke-PythonStep -StepName 'eval_stage1_dual_state' -Args @(
    (Join-Path $Root 'eval_stage1_clean.py'),
    '--config', (Join-Path $Root 'configs\stage1_clean.yaml')
)

Invoke-PythonStep -StepName 'visualize_stage1_segments' -Args @(
    (Join-Path $Root 'scripts\visualize_stage1_segments.py'),
    '--csv', $MergedCsv,
    '--meta_json', $MetaJson,
    '--tcn_config', (Join-Path $Root 'configs\stage1_tcn_gru.yaml'),
    '--dual_config', (Join-Path $Root 'configs\stage1_clean.yaml'),
    '--output_dir', $VisualDir
)

Write-Host "Stage1 rerun completed."
