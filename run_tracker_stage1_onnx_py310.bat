@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "ASCII_ROOT=D:\kun-data\kun-code-data\cpp_control_ascii"
set "PY310=C:\Users\Administrator\miniconda3\envs\py310"
set "BUILD_DIR=%ASCII_ROOT%\build\nmake-msvc-onnx-release"
set "STAGE1_ASSET_DIR=%ASCII_ROOT%\outputs\stage1_runtime_assets"

set "PATH=%PY310%;%PY310%\Library\bin;%PY310%\DLLs;%PY310%\Scripts;%BUILD_DIR%;%PATH%"
set "QT_PLUGIN_PATH=%PY310%\Library\lib\qt6\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%PY310%\Library\lib\qt6\plugins\platforms"
set "STAGE1_MODEL_ONNX=%STAGE1_ASSET_DIR%\dual_state_best.onnx"
set "STAGE1_META_JSON=%STAGE1_ASSET_DIR%\stage1_clean_meta.json"

pushd "%PROJECT_ROOT%"
if "%~1"=="" (
  "%BUILD_DIR%\tracker.exe" --mode cam --config "%PROJECT_ROOT%config\tracker.yaml"
) else (
  "%BUILD_DIR%\tracker.exe" %*
)
popd

endlocal
