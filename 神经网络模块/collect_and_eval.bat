@echo off
setlocal

set ROOT_DIR=%~dp0
set PROJECT_DIR=%ROOT_DIR%..

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

set LOG_SRC=%PROJECT_DIR%\logs\tracker_log.csv
set DATA_DIR=%ROOT_DIR%datasets
set METRIC_DIR=%ROOT_DIR%metrics
set LOG_DST=%DATA_DIR%\session_%TS%.csv
set METRIC_DST=%METRIC_DIR%\session_%TS%.json

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%METRIC_DIR%" mkdir "%METRIC_DIR%"

echo [1/3] build
pushd "%PROJECT_DIR%"
"C:\Program Files\CMake\bin\cmake.exe" --preset msvc-opencv-release
if errorlevel 1 (
  popd
  goto :err
)
"C:\Program Files\CMake\bin\cmake.exe" --build --preset msvc-opencv-release -j 8
set BUILD_RC=%ERRORLEVEL%
popd
if not "%BUILD_RC%"=="0" goto :err

echo [2/3] run tracker cam
set "PATH=C:\Users\Administrator\miniconda3\envs\py310\Library\bin;%PATH%"
pushd "%PROJECT_DIR%"
"%PROJECT_DIR%\build\msvc-opencv-release\tracker.exe" --mode cam --config "%PROJECT_DIR%\config\tracker.yaml"
set RUN_RC=%ERRORLEVEL%
popd
if not "%RUN_RC%"=="0" goto :err

echo [3/3] copy log and evaluate
if not exist "%LOG_SRC%" goto :err
for %%A in ("%LOG_SRC%") do set LOG_SIZE=%%~zA
if "%LOG_SIZE%"=="0" goto :err
copy /Y "%LOG_SRC%" "%LOG_DST%" >nul
python "%ROOT_DIR%\evaluate_log.py" --csv "%LOG_DST%" --out_json "%METRIC_DST%"
if errorlevel 1 goto :err

echo.
echo [OK] done
echo log:    %LOG_DST%
echo metric: %METRIC_DST%
goto :eof

:err
echo.
echo [ERROR] failed
exit /b 1
