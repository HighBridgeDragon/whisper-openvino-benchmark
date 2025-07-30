@echo off
REM Python Embeddable + uv Portable Environment Setup for Whisper Benchmark
REM This script creates a completely portable Python environment

echo ==========================================================
echo Whisper Benchmark Portable Environment Setup
echo ==========================================================
echo.

REM Set variables
set "PYTHON_DIR=python"
set "PROJECT_NAME=whisper-benchmark"

REM Read Python version from .python-version file
if exist "..\..\.python-version" (
    for /f "usebackq delims=" %%i in ("..\..\.python-version") do (
        set "PYTHON_BASE_VERSION=%%i"
        goto found_version
    )
) else (
    echo Warning: .python-version file not found, using default 3.13
    set "PYTHON_BASE_VERSION=3.13"
)

:found_version
REM Set Python version based on base version from .python-version
if "%PYTHON_BASE_VERSION%"=="3.13" (
    set "PYTHON_VERSION=3.13.2"
) else if "%PYTHON_BASE_VERSION%"=="3.12" (
    set "PYTHON_VERSION=3.12.8"
) else if "%PYTHON_BASE_VERSION%"=="3.11" (
    set "PYTHON_VERSION=3.11.11"
) else (
    echo Warning: Unsupported Python version %PYTHON_BASE_VERSION%, using 3.13.2
    set "PYTHON_VERSION=3.13.2"
)

echo Using Python version: %PYTHON_VERSION%

REM Check if already set up
if exist "python\python.exe" (
    if exist "python\Lib\site-packages\openvino_genai" (
        echo Dependencies already installed.
        echo Run run_benchmark.bat to execute the benchmark.
        pause
        exit /b 0
    )
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
REM Enable site-packages by uncommenting import site in python._pth file
for %%f in ("%PYTHON_DIR%\python*._pth") do (
    echo Configuring %%f...
    powershell -Command "(Get-Content '%%f') -replace '#import site', 'import site' | Set-Content '%%f'"
)

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

REM Install uv (latest version)
echo Installing latest uv version...
python.exe -m pip install uv --upgrade
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install uv
    cd ..
    pause
    exit /b 1
)

cd ..

echo [5/5] Installing project dependencies...
REM Copy project files
copy /Y "..\..\main.py" "main.py"
if exist "..\..\pyproject.toml" copy /Y "..\..\pyproject.toml" "pyproject.toml"

REM Install dependencies directly into Python embeddable using pip
echo Installing dependencies directly into Python environment...
python\python.exe -m pip install openvino-genai psutil librosa tabulate py-cpuinfo
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install dependencies using pip
    pause
    exit /b 1
)
echo Dependencies installed successfully

echo.
echo ==========================================================
echo Setup completed successfully!
echo ==========================================================
echo.
echo Python environment ready at: python\
echo.
echo To run the benchmark:
echo   1. Place your Whisper model in 'models' directory
echo   2. Run: run_benchmark.bat
echo.
echo The portable environment is ready for distribution.
echo.
pause