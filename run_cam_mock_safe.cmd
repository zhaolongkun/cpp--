@echo off
cd /d D:\kun-data\kun-code-data\反无\cpp智能控制
set "PATH=C:\Users\Administrator\miniconda3\envs\py310\Library\bin;%PATH%"
build\msvc-opencv-release\tracker.exe --mode cam --config config\tracker_cam_mock_paper.yaml
