@echo off
setlocal
set "ROOT=%~dp0"
set "PRESET=msvc-opencv-release"
set "BUILD=%ROOT%build\msvc-opencv-release"
set "EXE=%BUILD%\tracker.exe"
set "CONFIG=%ROOT%config\tracker.yaml"
set "TEMPORAL_MODEL="
set "TEMPORAL_STATS="
set "TEMPORAL_ASCII_DIR=%TEMP%\smart_gimbal_tracker_cpp\temporal_comp"
set "TEMPORAL_MODEL_ASCII=%TEMPORAL_ASCII_DIR%\causal_cnn_gru.onnx"
set "TEMPORAL_STATS_ASCII=%TEMPORAL_ASCII_DIR%\stats.json"
set "STARTUP_TRACE=%ROOT%logs\tracker_startup_trace.txt"
set "PY310=C:\Users\Administrator\miniconda3\envs\py310"
set "PYTHON=%PY310%\python.exe"
if not defined YOLO_CAMERA_INDEX set "YOLO_CAMERA_INDEX=0"
if not defined YOLO_CAMERA_BACKEND set "YOLO_CAMERA_BACKEND=dshow"
if not defined YOLO_SHM_WAIT_SEC set "YOLO_SHM_WAIT_SEC=20"
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = [System.IO.Path]::GetFullPath('%ROOT%'); $path = Get-ChildItem -LiteralPath $root -Recurse -File -Filter 'causal_cnn_gru.onnx' | Where-Object { $_.FullName -match 'checkpoints' -and $_.DirectoryName -notmatch 'backup|smoke' } | Select-Object -First 1 -ExpandProperty FullName; if ($path) { Write-Output $path }"`) do set "TEMPORAL_MODEL=%%I"
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = [System.IO.Path]::GetFullPath('%ROOT%'); $path = Get-ChildItem -LiteralPath $root -Recurse -File -Filter 'stats.json' | Where-Object { $_.FullName -match 'checkpoints' -and $_.DirectoryName -notmatch 'backup|smoke' } | Select-Object -First 1 -ExpandProperty FullName; if ($path) { Write-Output $path }"`) do set "TEMPORAL_STATS=%%I"
set "TRACKER_STARTUP_TRACE=%STARTUP_TRACE%"
set "PATH=%BUILD%;%PY310%;%PY310%\Library\bin;%PY310%\DLLs;%PY310%\Scripts;%PATH%"
set "QT_PLUGIN_PATH=%PY310%\Library\lib\qt6\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%PY310%\Library\lib\qt6\plugins\platforms"

if exist "%STARTUP_TRACE%" del /q "%STARTUP_TRACE%"

if not exist "%TEMPORAL_MODEL%" (
    echo [error] temporal model not found: %TEMPORAL_MODEL%
    pause
    exit /b 1
)
if not exist "%TEMPORAL_STATS%" (
    echo [error] temporal stats not found: %TEMPORAL_STATS%
    pause
    exit /b 1
)

echo [0/4] Configuring preset %PRESET%...
cmake --preset "%PRESET%"
if errorlevel 1 (
    echo Configure failed.
    pause
    exit /b 1
)

echo [1/4] Building tracker with temporal prediction...
cmake --build "%BUILD%" --config Release
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
if not exist "%EXE%" (
    echo [error] tracker.exe not found: %EXE%
    pause
    exit /b 1
)

if not exist "%TEMPORAL_ASCII_DIR%" mkdir "%TEMPORAL_ASCII_DIR%"
copy /y "%TEMPORAL_MODEL%" "%TEMPORAL_MODEL_ASCII%" >nul
if errorlevel 1 (
    echo [error] failed to copy temporal model to ascii path
    pause
    exit /b 1
)
copy /y "%TEMPORAL_STATS%" "%TEMPORAL_STATS_ASCII%" >nul
if errorlevel 1 (
    echo [error] failed to copy temporal stats to ascii path
    pause
    exit /b 1
)
set "TEMPORAL_MODEL_ONNX=%TEMPORAL_MODEL_ASCII%"
set "TEMPORAL_STATS_JSON=%TEMPORAL_STATS_ASCII%"

echo [2/4] Starting YOLO GPU server... camera=%YOLO_CAMERA_INDEX% backend=%YOLO_CAMERA_BACKEND%
start "yolo_shm_server" cmd /k ""%PYTHON%" "%ROOT%tools\yolo_shm_server.py" --camera %YOLO_CAMERA_INDEX% --backend %YOLO_CAMERA_BACKEND%"
echo [wait] waiting for yolo_det_shm (timeout=%YOLO_SHM_WAIT_SEC%s)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(%YOLO_SHM_WAIT_SEC%); while((Get-Date)-lt $deadline){ try { $mmf=[System.IO.MemoryMappedFiles.MemoryMappedFile]::OpenExisting('yolo_det_shm'); if($mmf){ $mmf.Dispose(); exit 0 } } catch {}; Start-Sleep -Milliseconds 200 }; exit 1"
if errorlevel 1 (
    echo [tracker] yolo_det_shm was not ready in time.
    pause
    exit /b 1
)

echo [3/4] Temporal prediction model: %TEMPORAL_MODEL_ASCII%
echo [4/4] Starting tracker with X/Y motor output...
pushd "%ROOT%"
"%EXE%" --mode cam --config "%CONFIG%"
set "TRACKER_EXIT=%ERRORLEVEL%"
popd
if not "%TRACKER_EXIT%"=="0" (
    echo.
    echo [tracker] exited with code %TRACKER_EXIT%
    if exist "%STARTUP_TRACE%" (
        echo [tracker] startup trace:
        type "%STARTUP_TRACE%"
    ) else (
        echo [tracker] startup trace not found: %STARTUP_TRACE%
    )
    pause
    exit /b %TRACKER_EXIT%
)
endlocal
