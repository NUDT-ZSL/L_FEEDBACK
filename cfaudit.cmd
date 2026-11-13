@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
set PYTHONPATH=%~dp0;%PYTHONPATH%
py -3 -m cfaudit %*
