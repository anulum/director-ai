@REM SPDX-License-Identifier: Apache-2.0
@REM Commercial license available
@REM Copyright 2020-2026 Miroslav Sotek
@REM ORCID: 0009-0009-3560-0851
@REM Contact: www.anulum.li | protoscience@anulum.li
@REM Director-Class AI - Windows Offline Licence Key Ceremony Launcher
@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

echo Director-AI SEC-1 offline key ceremony
echo This computer must be offline before you continue.
echo.

where py >nul 2>nul
if %errorlevel% equ 0 (
  set "PYTHON=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 goto :no_python
  set "PYTHON=python"
)

if exist ".ceremony-venv" rmdir /s /q ".ceremony-venv"
%PYTHON% verify_bundle.py
if errorlevel 1 goto :failed
%PYTHON% -m venv .ceremony-venv
if errorlevel 1 goto :failed

call ".ceremony-venv\Scripts\activate.bat"
if errorlevel 1 goto :failed
python -m pip install --disable-pip-version-check --no-index --require-hashes --find-links wheelhouse -r requirements-offline.txt
if errorlevel 1 goto :failed

echo.
python run_ceremony.py
if errorlevel 1 goto :failed

echo.
echo SUCCESS. Shut down the laptop before removing the PRIVATE vault medium.
echo Only PUBLIC_KEY_ONLY.txt may return to the online workstation.
pause
exit /b 0

:no_python
echo ERROR: Python 3 is not installed or not on PATH.
goto :failed

:failed
echo.
echo Ceremony stopped without overwriting any existing key file.
pause
exit /b 1
