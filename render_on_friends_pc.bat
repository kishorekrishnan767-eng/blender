@echo off
title LAST SEEN - final render
rem Keep this file, last_seen.blend and last_seen_master.py together in one folder, then double-click this file.
rem It uses the NVIDIA card, saves every finished frame in the "frames" folder and skips frames that already exist.
rem To pause: close this window. To resume: double-click it again. Nothing already rendered is lost.

set "BLENDER=C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
if not exist "%BLENDER%" (
    echo Blender 5.2 was not found at:
    echo   %BLENDER%
    echo Install Blender 5.2 LTS from https://www.blender.org/download/lts/ and run this again.
    pause
    exit /b 1
)

"%BLENDER%" -b "%~dp0last_seen.blend" -P "%~dp0last_seen_master.py" -- --final
if errorlevel 1 (
    echo The render stopped with an error. Double-click this file to continue from the last finished frame.
    pause
    exit /b 1
)

"%BLENDER%" -b "%~dp0last_seen.blend" -P "%~dp0last_seen_master.py" -- --encode
echo.
echo Finished. The video is last_seen.mp4 in this folder.
pause
