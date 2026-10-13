@echo off
echo ========================================================
echo Face Segmentation - Virtual Environment Installer
echo ========================================================
echo.

echo [1/3] Creating virtual environment 'venv'...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. Please ensure Python is installed and in your PATH.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/3] Upgrading pip and installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo Installation Complete!
echo ========================================================
echo.
echo To run the GUI application in the future, you can:
echo 1. Run the 'run.bat' file, OR
echo 2. Open a terminal and type:
echo    venv\Scripts\activate
echo    python gui.py
echo.
pause
