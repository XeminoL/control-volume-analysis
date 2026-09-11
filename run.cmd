@echo off
setlocal

set PYTHONDONTWRITEBYTECODE=1
set ROOT=%~dp0

where python >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 or newer is required. Install it from https://python.org
  pause
  exit /b 1
)

python -c "import numpy, h5py" >nul 2>&1
if errorlevel 1 (
  echo Installing dependencies...
  python -m pip install --quiet --disable-pip-version-check numpy h5py
  if errorlevel 1 exit /b 1
)

if "%~1"=="" goto web
if /i "%~1"=="web" goto web
if /i "%~1"=="lint" goto lint
goto script

:web
pushd "%ROOT%"
python -B serve.py
popd
exit /b 0

:lint
pushd "%ROOT%"
for %%f in (*.py) do python -B -m pyflakes "%%f"
popd
echo lint clean
exit /b 0

:script
pushd "%ROOT%"
python -B "%~1" %2 %3 %4 %5
popd