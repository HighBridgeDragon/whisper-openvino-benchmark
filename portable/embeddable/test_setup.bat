@echo off
REM Test script for embeddable environment setup
echo Testing embeddable environment setup...
echo.

REM Check if setup completed
if not exist "python\python.exe" (
    echo Running setup...
    call setup_portable.bat
) else (
    echo Environment already exists.
)

echo.
echo Testing Python installation...
python\python.exe --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not working properly
    pause
    exit /b 1
)

echo.
echo Testing pip...
python\python.exe -m pip --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: pip is not working properly
    pause  
    exit /b 1
)

echo.
echo Testing uv...
python\python.exe -m uv --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: uv is not working properly
    pause
    exit /b 1
)

echo.
echo Testing OpenVINO import...
python\python.exe -c "import openvino; print('OpenVINO version:', openvino.__version__)"
if %ERRORLEVEL% NEQ 0 (
    echo Error: OpenVINO import failed
    pause
    exit /b 1
)

echo.
echo Testing OpenVINO GenAI import...
python\python.exe -c "import openvino_genai; print('OpenVINO GenAI imported successfully')"
if %ERRORLEVEL% NEQ 0 (
    echo Error: OpenVINO GenAI import failed  
    pause
    exit /b 1
)

echo.
echo Testing librosa import...
python\python.exe -c "import librosa; print('librosa version:', librosa.__version__)"
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
for /f "tokens=3" %%a in ('dir python /s /-c ^| find "bytes"') do echo Python environment: %%a bytes

echo.
pause