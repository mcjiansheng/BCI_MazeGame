@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-mi\Scripts\python.exe" goto missing
".venv-mi\Scripts\python.exe" -X utf8 scripts\run_lk_mi.py ui %*
if errorlevel 1 pause
exit /b %errorlevel%
:missing
echo Missing .venv-mi. See doc/LOCAL_MI_SETUP.md.
pause
exit /b 2
