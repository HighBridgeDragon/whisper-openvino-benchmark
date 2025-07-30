@echo off
REM Test script for embeddable environment setup
echo Testing embeddable environment setup...
echo.

REM Check if setup completed
if not exist "python\python.exe" (
    echo Running setup...
    call setup_portable.bat
) else (
    echo Python embeddable exists.
    if not exist ".venv\Scripts\python.exe" (
        echo Virtual environment not found. Running setup again...
        call setup_portable.bat
    ) else (
        echo Virtual environment exists.
    )
)

echo.
echo Testing Python installation...
.venv\Scripts\python.exe --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not working properly
    pause
    exit /b 1
)

echo.
echo Testing package manager availability...
REM uv is managed by embeddable Python, not in venv
echo Checking uv in embeddable Python...
python\python.exe -m uv --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: uv is not working properly
    pause  
    exit /b 1
)

REM UV is managed by embeddable Python, not in venv

echo.
echo Testing OpenVINO import...
.venv\Scripts\python.exe -c "import openvino; print('OpenVINO version:', openvino.__version__)"
if %ERRORLEVEL% NEQ 0 (
    echo Error: OpenVINO import failed
    pause
    exit /b 1
)

echo.
echo Testing OpenVINO GenAI import...
.venv\Scripts\python.exe -c "import openvino_genai; print('OpenVINO GenAI imported successfully')"
if %ERRORLEVEL% NEQ 0 (
    echo Error: OpenVINO GenAI import failed  
    pause
    exit /b 1
)

echo.
echo Testing librosa import...
.venv\Scripts\python.exe -c "import librosa; print('librosa version:', librosa.__version__)"
if %ERRORLEVEL% NEQ 0 (
    echo Error: librosa import failed
    pause
    exit /b 1
)

echo.
echo ===========================================
echo All tests passed! Environment is ready.
echo ===========================================
echo.
echo Available commands:
echo - run_benchmark.bat --help
echo - create_distribution.bat
echo.

REM Show disk usage
echo Environment size:
for /f "tokens=3" %%a in ('dir .venv /s /-c ^| find "bytes"') do echo Virtual environment: %%a bytes

echo.
pause