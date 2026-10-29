@echo off
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython
if exist ".venv\Scripts\python.exe" goto havevenv
echo Creating virtual environment .venv ...
%PY% -m venv .venv
if errorlevel 1 goto fail
:havevenv
call ".venv\Scripts\activate.bat"
echo Installing dependencies ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo Starting the simulator. Open http://localhost:8501 if the browser does not start by itself.
python -m streamlit run app.py
goto end
:nopython
echo Python 3.10 or newer was not found. Install it from python.org and tick the option to add Python to PATH.
goto end
:fail
echo Setup failed. Check the messages above.
:end
pause
