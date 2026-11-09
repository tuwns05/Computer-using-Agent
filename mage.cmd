@echo off
chcp 65001 >nul
set PYTHONUTF8=1
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0mage.py" %*
) else (
  python "%~dp0mage.py" %*
)
