@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PY310=C:\Users\Administrator\miniconda3\envs\py310"
set "PATH=%PY310%;%PY310%\Library\bin;%PY310%\DLLs;%PY310%\Scripts;%PATH%"
set "QT_PLUGIN_PATH=%PY310%\Library\lib\qt6\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%PY310%\Library\lib\qt6\plugins\platforms"
pushd "%PROJECT_ROOT%"
"%PROJECT_ROOT%build\msvc-opencv-release\tracker.exe" --mode cam --config "%PROJECT_ROOT%config\tracker_collect_nn.yaml"
popd
endlocal
