@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-mi\Scripts\python.exe" goto missing
if "%~1"=="" goto demo
".venv-mi\Scripts\python.exe" -X utf8 scripts\run_lk_mi.py %*
exit /b %errorlevel%
:demo
".venv-mi\Scripts\python.exe" -X utf8 scripts\run_lk_mi.py demo
pause
exit /b
:missing
echo Missing .venv-mi. See doc/LOCAL_MI_SETUP.md.
exit /b 2
