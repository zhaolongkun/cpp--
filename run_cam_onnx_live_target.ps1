param(
  [int]$MaxRuntimeMs = 35000,
  [string]$Config = "config\tracker_cam_live_target_paper.yaml",
  [string]$LogCsv = "paper\generated\live_cam\tracker_log.csv",
  [ValidateSet("positive", "recovery")]
  [string]$Protocol = "positive"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$env:R2S_MODEL_ONNX = "D:\kun-data\kun-code-data\cpp_control_ascii\artifacts_ascii\train_run1\model.onnx"
$env:R2S_SCALER_JSON = "D:\kun-data\kun-code-data\cpp_control_ascii\artifacts_ascii\train_run1\scaler.json"
$env:R2S_FEATURE_SPEC_JSON = "D:\kun-data\kun-code-data\cpp_control_ascii\artifacts_ascii\train_run1\feature_spec.json"
$env:PATH = "D:\kun-data\kun-code-data\反无\cpp智能控制\build\nmake-msvc-onnx-release;C:\Users\Administrator\miniconda3\envs\py310\Library\bin;$env:PATH"

Write-Host "[Step] Camera + ONNX live target capture ($Protocol)" -ForegroundColor Cyan
Write-Host "[Target condition] aim for bbox size about 90x100 px (area around 9k px^2 or larger)." -ForegroundColor Yellow
Write-Host "[Target condition] keep target center within roughly +/-60 px horizontally and +/-40 px vertically around image center." -ForegroundColor Yellow
Write-Host "[Target condition] detector-positive frames in the preserved good run had det_conf around 0.67-0.69 and long continuous visibility." -ForegroundColor Yellow
if ($Protocol -eq "positive") {
  Write-Host "[Info] Positive protocol: bring the target into the central region and hold it stably for at least 10-15 s." -ForegroundColor Yellow
  Write-Host "[Info] Avoid fast hand jitter. Use a larger, clearer target and keep it mostly centered." -ForegroundColor Yellow
  Write-Host "[Info] Goal: second target-present full-chain run with track_count>0, lost_flag=0, alpha_gate>0, delta!=0." -ForegroundColor Yellow
} else {
  Write-Host "[Info] Recovery protocol Phase A (0-8 s): stable positive tracking near image center." -ForegroundColor Yellow
  Write-Host "[Info] Recovery protocol Phase B (8-10 s): move target fully out of view for a short controlled loss." -ForegroundColor Yellow
  Write-Host "[Info] Recovery protocol Phase C (10 s+): re-enter near image center and hold > 1 s, preferably > 2 s." -ForegroundColor Yellow
  Write-Host "[Info] Goal: loss -> reacquire -> hold with tracked_after_reacquire_ms >= 200 ms, ideally >= 500 ms." -ForegroundColor Yellow
}
Write-Host "[Info] Safe boundary: actuator.mode=mock, armed=false, no motor USB." -ForegroundColor Yellow
Write-Host "[Info] Press q to stop early, or wait for timeout." -ForegroundColor Yellow

if (Test-Path -LiteralPath $LogCsv) {
  Remove-Item -LiteralPath $LogCsv -Force
}

& ".\build\nmake-msvc-onnx-release\tracker.exe" --mode cam --config $Config --max_runtime_ms $MaxRuntimeMs

& "C:\Users\Administrator\miniconda3\envs\py310\python.exe" ".\paper\finalize_live_capture.py" --log_csv $LogCsv
