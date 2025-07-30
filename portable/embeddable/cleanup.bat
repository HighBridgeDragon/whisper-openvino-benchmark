@echo off
REM Cleanup script to remove .venv and force fresh setup
echo Cleaning up virtual environment...

if exist ".venv" (
    echo Removing .venv directory...
    rmdir /s /q ".venv"
    echo .venv directory removed.
) else (
    echo No .venv directory found.
)

echo.
echo Cleanup completed. Run setup_portable.bat to recreate the environment.
pause