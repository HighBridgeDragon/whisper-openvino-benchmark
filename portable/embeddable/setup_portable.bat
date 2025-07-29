@echo off
REM Python Embeddable + uv Portable Environment Setup for Whisper Benchmark
REM This script creates a completely portable Python environment

echo ==========================================================
echo Whisper Benchmark Portable Environment Setup
echo ==========================================================
echo.

REM Set variables
set "PYTHON_VERSION=3.13.2"
set "PYTHON_DIR=python"
set "UV_VERSION=0.5.14"
set "PROJECT_NAME=whisper-benchmark"

REM Check if already set up
if exist "%PYTHON_DIR%\python.exe" (
    echo Portable environment already exists.
    echo Run run_benchmark.bat to execute the benchmark.
    pause
    exit /b 0
)

echo [1/5] Downloading Python Embeddable...
if not exist downloads mkdir downloads

REM Download Python embeddable if not exists
if not exist "downloads\python-%PYTHON_VERSION%-embed-amd64.zip" (
    echo Downloading Python %PYTHON_VERSION% embeddable...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip' -OutFile 'downloads\python-%PYTHON_VERSION%-embed-amd64.zip'"
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to download Python embeddable
        pause
        exit /b 1
    )
) else (
    echo Python embeddable already downloaded.
)

echo [2/5] Extracting Python...
powershell -Command "Expand-Archive -Path 'downloads\python-%PYTHON_VERSION%-embed-amd64.zip' -DestinationPath '%PYTHON_DIR%' -Force"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to extract Python
    pause
    exit /b 1
)

echo [3/5] Configuring Python paths...
REM Enable site-packages by uncommenting import site in python313._pth
powershell -Command "(Get-Content '%PYTHON_DIR%\python313._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python313._pth'"

REM Create get-pip.py if not exists
if not exist "%PYTHON_DIR%\get-pip.py" (
    echo Downloading get-pip.py...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'"
)

echo [4/5] Installing pip and uv...
cd %PYTHON_DIR%
python.exe get-pip.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install pip
    cd ..
    pause
    exit /b 1
)

REM Install uv
python.exe -m pip install uv==%UV_VERSION%
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install uv
    cd ..
    pause
    exit /b 1
)

cd ..

echo [5/5] Installing project dependencies...
REM Copy project files to portable environment  
copy /Y "..\..\main.py" "%PYTHON_DIR%\main.py"
if exist "..\..\pyproject.toml" copy /Y "..\..\pyproject.toml" %PYTHON_DIR%\
if exist "..\..\uv.lock" copy /Y "..\..\uv.lock" %PYTHON_DIR%\

cd %PYTHON_DIR%

REM Install dependencies from pyproject.toml using uv sync
echo Installing dependencies from pyproject.toml...
if exist "pyproject.toml" (
    echo Using uv sync for accurate dependency management...
    python.exe -m uv sync --no-dev
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to install dependencies using uv sync
        cd ..
        pause
        exit /b 1
    )
    echo Dependencies installed successfully using uv sync
) else (
    echo Error: pyproject.toml not found - cannot install dependencies
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo ==========================================================
echo Setup completed successfully!
echo ==========================================================
echo.
echo To run the benchmark:
echo   1. Place your Whisper model in 'models' directory
echo   2. Run: run_benchmark.bat
echo.
echo The portable environment is ready for distribution.
echo Total size: 
dir %PYTHON_DIR% /s /-c | find "bytes"
echo.
pause