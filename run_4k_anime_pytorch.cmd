@echo off
setlocal
set "BASE=%~dp0"
if "%BASE:~-1%"=="\" set "BASE=%BASE:~0,-1%"
set "PY=%BASE%\.venv\Scripts\python.exe"
set "SCRIPT=%BASE%\batch_realesrgan_exact_4k.py"

if not exist "%PY%" (
  echo Python environment not found:
  echo %PY%
  goto end
)

if not exist "%SCRIPT%" (
  echo Script not found:
  echo %SCRIPT%
  goto end
)

pushd "%BASE%"
"%PY%" "%SCRIPT%" --base "%BASE%" --mode anime
popd

:end
pause
