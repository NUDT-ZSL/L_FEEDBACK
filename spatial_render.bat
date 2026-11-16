@echo off
setlocal
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 -m spatial_audio_renderer %*
) else (
    python -m spatial_audio_renderer %*
)
exit /b %errorlevel%
