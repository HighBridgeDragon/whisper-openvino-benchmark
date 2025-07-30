@echo off
REM Debug script to identify the exact error

echo ==========================================================
echo Debug Setup Script
echo ==========================================================
echo.

echo Current directory: %CD%
echo.

echo Checking Python executable...
if exist "python\python.exe" (
    echo Python found at: python\python.exe
    python\python.exe --version
) else (
    echo Python NOT found!
)
echo.

echo Checking uv installation...
python\python.exe -m uv --version
if %ERRORLEVEL% NEQ 0 (
    echo UV not installed or error occurred
) else (
    echo UV is installed correctly
)
echo.

echo Checking pyproject.toml...
if exist "pyproject.toml" (
    echo pyproject.toml found
) else (
    echo pyproject.toml NOT found!
)
echo.

echo Testing uv sync command directly...
echo Command: python\python.exe -m uv sync --no-dev
python\python.exe -m uv sync --no-dev
echo Exit code: %ERRORLEVEL%
echo.

pause