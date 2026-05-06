@echo off
setlocal

set ROOT_DIR=%~dp0
set CSV_PATH=%ROOT_DIR%..\logs\tracker_log.csv
set DATASET_PATH=%ROOT_DIR%artifacts\dataset.npz
set RUN_DIR=%ROOT_DIR%artifacts\runs\exp1

if not exist "%ROOT_DIR%artifacts" mkdir "%ROOT_DIR%artifacts"
if not exist "%ROOT_DIR%artifacts\runs" mkdir "%ROOT_DIR%artifacts\runs"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

python "%ROOT_DIR%build_dataset.py" --csv "%CSV_PATH%" --out "%DATASET_PATH%" --seq_len 12
if errorlevel 1 goto :err

python "%ROOT_DIR%train_nce.py" --dataset "%DATASET_PATH%" --out_dir "%RUN_DIR%" --epochs 40 --batch_size 256
if errorlevel 1 goto :err

python "%ROOT_DIR%export_onnx.py" --ckpt "%RUN_DIR%\best.pt" --out "%RUN_DIR%\nce_pnr.onnx"
if errorlevel 1 goto :err

echo.
echo [OK] training pipeline finished.
echo checkpoint: %RUN_DIR%\best.pt
echo onnx:       %RUN_DIR%\nce_pnr.onnx
goto :eof

:err
echo.
echo [ERROR] pipeline failed.
exit /b 1
