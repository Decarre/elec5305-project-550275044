@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" demo\serve.py --open
if errorlevel 1 pause
