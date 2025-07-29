@echo off
REM Portable Whisper Benchmark Runner
REM This script runs the benchmark using the portable Python environment

setlocal EnableDelayedExpansion

echo ==========================================================
echo Whisper Benchmark - Portable Runner
echo ==========================================================
echo.

REM Check if portable environment exists
if not exist "python\python.exe" (
    echo Error: Portable environment not found.
    echo Please run setup_portable.bat first.
    pause
    exit /b 1
)

REM Set current directory as Python path
set "CURRENT_DIR=%~dp0"
set "PYTHON_DIR=%CURRENT_DIR%python"
set "PYTHONPATH=%PYTHON_DIR%;%PYTHON_DIR%\Lib\site-packages"

REM Add DLL directory for OpenVINO (similar to main.py fix)
set "PATH=%PYTHON_DIR%\Lib\site-packages\openvino_tokenizers\lib;%PATH%"

echo Portable Python environment: %PYTHON_DIR%
echo.

REM Check for model directory
if not exist "models" (
    echo Creating models directory...
    mkdir models
    echo.
    echo WARNING: No models found in 'models' directory.
    echo Please place your exported Whisper model in the models folder.
    echo Example: models\openai\whisper-large-v3-turbo-stateless\
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i "!continue!" NEQ "y" (
        echo Benchmark cancelled.
        pause
        exit /b 0
    )
)

REM Default parameters
set "MODEL_PATH="
set "ITERATIONS=5"
set "NUM_BEAMS=1"
set "DEVICE=CPU"
set "AUDIO_FILE="

REM Parse command line arguments
:parse_args
if "%~1"=="" goto run_benchmark
if "%~1"=="--model-path" (
    set "MODEL_PATH=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--iterations" (
    set "ITERATIONS=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--num-beams" (
    set "NUM_BEAMS=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--device" (
    set "DEVICE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--audio-file" (
    set "AUDIO_FILE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: run_benchmark.bat [options]
    echo.
    echo Options:
    echo   --model-path PATH     Path to Whisper model directory
    echo   --iterations N        Number of benchmark iterations (default: 5)
    echo   --num-beams N         Number of beams for decoding (default: 1)
    echo   --device DEVICE       Device to use: CPU, GPU (default: CPU)
    echo   --audio-file PATH     Path to audio file (if not specified, downloads default)
    echo   --help                Show this help message
    echo.
    echo Examples:
    echo   run_benchmark.bat --model-path models\whisper-large-v3-turbo-stateless
    echo   run_benchmark.bat --model-path models\whisper-base --iterations 10
    echo   run_benchmark.bat --audio-file my_audio.wav --iterations 3
    echo.
    pause
    exit /b 0
)
shift
goto parse_args

:run_benchmark

REM Auto-detect model if not specified
if "%MODEL_PATH%"=="" (
    echo Auto-detecting model...
    for /f "delims=" %%i in ('dir /b /s models\config.json 2^>nul ^| find /v "File Not Found"') do (
        set "CONFIG_FILE=%%i"
        for %%j in ("!CONFIG_FILE!") do set "MODEL_PATH=%%~dpj"
        set "MODEL_PATH=!MODEL_PATH:~0,-1!"
        echo Found model: !MODEL_PATH!
        goto model_found
    )
    
    echo Error: No Whisper model found in 'models' directory.
    echo.
    echo To export a model, use:
    echo   uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo --trust-remote-code --weight-format fp32 --disable-stateful models\whisper-large-v3-turbo-stateless
    echo.
    pause
    exit /b 1
)

:model_found

REM Display benchmark configuration
echo ==========================================================
echo Benchmark Configuration
echo ==========================================================
echo Model Path: %MODEL_PATH%
echo Iterations: %ITERATIONS%
echo Num Beams: %NUM_BEAMS%
echo Device: %DEVICE%
if not "%AUDIO_FILE%"=="" echo Audio File: %AUDIO_FILE%
echo ==========================================================
echo.

REM Change to python directory and run benchmark
cd python

echo Running benchmark...
if "%AUDIO_FILE%"=="" (
    python.exe main.py --model-path "%MODEL_PATH%" --iterations %ITERATIONS% --num-beams %NUM_BEAMS% --device %DEVICE%
) else (
    python.exe main.py --model-path "%MODEL_PATH%" --iterations %ITERATIONS% --num-beams %NUM_BEAMS% --device %DEVICE% --audio-file "%AUDIO_FILE%"
)

set "BENCHMARK_EXIT_CODE=%ERRORLEVEL%"

cd ..

echo.
if %BENCHMARK_EXIT_CODE% EQU 0 (
    echo ==========================================================
    echo Benchmark completed successfully!
    echo ==========================================================
) else (
    echo ==========================================================
    echo Benchmark failed with exit code: %BENCHMARK_EXIT_CODE%
    echo ==========================================================
    echo.
    echo Troubleshooting tips:
    echo 1. Ensure the model was exported with --disable-stateful flag
    echo 2. Check that all required model files exist
    echo 3. Verify OpenVINO GenAI version compatibility
    echo 4. Try running with --device CPU if using GPU
)

echo.
pause
exit /b %BENCHMARK_EXIT_CODE%