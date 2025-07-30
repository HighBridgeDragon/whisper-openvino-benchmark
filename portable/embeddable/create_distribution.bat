@echo off
REM Create distribution package for Whisper Benchmark Portable
REM This script packages the portable environment for easy distribution

echo ==========================================================
echo Creating Whisper Benchmark Distribution Package
echo ==========================================================
echo.

set "DIST_NAME=whisper-benchmark-portable"
set "DIST_DIR=dist"

REM Create timestamp for archive name (using time-based approach)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DATETIME=%%I"
set "TIMESTAMP=%DATETIME:~0,8%"
set "ARCHIVE_NAME=%DIST_NAME%-%TIMESTAMP%.zip"

REM Check if portable environment exists
if not exist "python\python.exe" (
    echo Error: Portable environment not found.
    echo Please run setup_portable.bat first.
    pause
    exit /b 1
)

echo [1/4] Creating distribution directory...
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
mkdir "%DIST_DIR%"
mkdir "%DIST_DIR%\%DIST_NAME%"

echo [2/4] Copying portable environment...
xcopy /E /I /H /Y python "%DIST_DIR%\%DIST_NAME%\python" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to copy Python environment
    pause
    exit /b 1
)

echo [3/4] Copying distribution files...
copy /Y run_benchmark.bat "%DIST_DIR%\%DIST_NAME%\"
copy /Y setup_portable.bat "%DIST_DIR%\%DIST_NAME%\"

REM Copy project files from parent directory
copy /Y "..\..\pyproject.toml" "%DIST_DIR%\%DIST_NAME%\"
copy /Y "..\..\main.py" "%DIST_DIR%\%DIST_NAME%\"

REM Create README for distribution
echo Creating README.txt...
(
echo ========================================================
echo Whisper Benchmark - Portable Distribution
echo ========================================================
echo.
echo This is a portable Whisper speech recognition benchmark tool
echo that runs without requiring Python installation.
echo.
echo QUICK START:
echo 1. Extract this package to any directory
echo 2. Run run_benchmark.bat
echo 3. Follow the on-screen instructions
echo.
echo SYSTEM REQUIREMENTS:
echo - Windows 10/11 x64
echo - At least 4GB RAM
echo - 2GB free disk space for models
echo.
echo USAGE:
echo   run_benchmark.bat [options]
echo.
echo OPTIONS:
echo   --model-path PATH     Path to Whisper model directory
echo   --iterations N        Number of benchmark iterations (default: 5)
echo   --num-beams N         Number of beams for decoding (default: 1)
echo   --device DEVICE       Device to use: CPU, GPU (default: CPU)
echo   --language LANG       Language code (default: ^<^|en^|^>)
echo   --help                Show help message
echo.
echo EXAMPLES:
echo   run_benchmark.bat --model-path models\whisper-large-v3-turbo-stateless
echo   run_benchmark.bat --iterations 10 --device CPU
echo.
echo MODEL PREPARATION:
echo To prepare a Whisper model, you need to export it in OpenVINO format:
echo.
echo 1. Install Python and uv on a development machine
echo 2. Run the following command:
echo    uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo \
echo        --trust-remote-code --weight-format fp32 --disable-stateful \
echo        models\whisper-large-v3-turbo-stateless
echo.
echo 3. Copy the models directory to this portable package
echo.
echo TROUBLESHOOTING:
echo - If you get "model not found" errors, ensure the model directory
echo   contains all required .xml and .bin files
echo - For OpenVINO GenAI 2025.2, use --disable-stateful flag when exporting
echo - For large models, ensure you have sufficient RAM
echo.
echo SUPPORT:
echo For issues and updates, visit:
echo https://github.com/openvinotoolkit/openvino
echo.
echo This package includes:
echo - Python 3.13 Embeddable
echo - uv package manager
echo - OpenVINO GenAI
echo - Required dependencies
echo.
echo Total package size: ~150MB
echo ========================================================
) > "%DIST_DIR%\%DIST_NAME%\README.txt"

REM Create models directory with example structure
mkdir "%DIST_DIR%\%DIST_NAME%\models"
(
echo Place your exported Whisper models in this directory.
echo.
echo Example structure:
echo models\
echo   └── openai\
echo       └── whisper-large-v3-turbo-stateless\
echo           ├── config.json
echo           ├── generation_config.json
echo           ├── openvino_encoder_model.xml
echo           ├── openvino_encoder_model.bin
echo           ├── openvino_decoder_model.xml
echo           ├── openvino_decoder_model.bin
echo           ├── openvino_tokenizer.xml
echo           ├── openvino_tokenizer.bin
echo           ├── openvino_detokenizer.xml
echo           └── openvino_detokenizer.bin
echo.
echo To export a model:
echo uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo \
echo     --trust-remote-code --weight-format fp32 --disable-stateful \
echo     models\whisper-large-v3-turbo-stateless
) > "%DIST_DIR%\%DIST_NAME%\models\README_MODELS.txt"

echo [4/4] Creating archive...
powershell -Command "Compress-Archive -Path '%DIST_DIR%\%DIST_NAME%' -DestinationPath '%DIST_DIR%\%ARCHIVE_NAME%' -Force"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create archive
    pause
    exit /b 1
)

REM Get archive size
for %%A in ("%DIST_DIR%\%ARCHIVE_NAME%") do set "ARCHIVE_SIZE=%%~zA"
set /a "ARCHIVE_SIZE_MB=%ARCHIVE_SIZE% / 1024 / 1024"

echo.
echo ==========================================================
echo Distribution package created successfully!
echo ==========================================================
echo.
echo Package: %DIST_DIR%\%ARCHIVE_NAME%
echo Size: %ARCHIVE_SIZE_MB% MB
echo.
echo The package contains:
echo - Portable Python environment with all dependencies
echo - Benchmark runner scripts
echo - Documentation and examples
echo - Models directory (ready for your Whisper models)
echo.
echo To distribute:
echo 1. Copy %ARCHIVE_NAME% to target machines
echo 2. Extract and run run_benchmark.bat
echo 3. Add Whisper models to the models directory
echo.
echo Note: Recipients do NOT need Python or any dependencies installed!
echo.
pause