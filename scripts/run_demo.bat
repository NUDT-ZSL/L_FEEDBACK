@echo off
REM Deterministic end-to-end demo for Windows (cmd.exe).
REM Requires Python 3.9+ with numpy installed: py -m pip install -r requirements.txt
setlocal
set PY=py -3
if not exist .venv\Scripts\python.exe goto run
set PY=.venv\Scripts\python.exe
:run
set OUT=artifacts\demo
%PY% -m audio_sync simulate --output-dir %OUT% --streams 3 --frames 250 --seed 20261114 --loss-rate 0.12 --reorder-rate 0.06 --inject-faults --burst-loss
if errorlevel 1 exit /b 1
%PY% -m audio_sync run --input-dir %OUT% --wav %OUT%\mixed.wav --log %OUT%\frames.jsonl --buffer-depth 4
if errorlevel 1 exit /b 1
%PY% -m audio_sync verify --input-dir %OUT% --wav %OUT%\mixed.wav --log %OUT%\frames.jsonl
exit /b %errorlevel%
