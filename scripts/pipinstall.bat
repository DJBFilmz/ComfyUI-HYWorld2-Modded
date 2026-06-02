@echo off
setlocal

:: Script lives at: ComfyUI\custom_nodes\<module>\scripts\pipinstall.bat
:: python_embeded lives at: ComfyUI\python_embeded\python.exe
set "SCRIPT_DIR=%~dp0"
set "COMFYUI_ROOT=%SCRIPT_DIR%..\..\..\"

set "PYTHON_EXE=%COMFYUI_ROOT%python_embeded\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] python_embeded not found at: %PYTHON_EXE%
    echo Make sure this script is located in ComfyUI\custom_nodes\<module>\scripts\
    pause
    exit /b 1
)

echo [OK] Found Python: %PYTHON_EXE%
echo [*] Installing HY-World gsplat_maskgaussian fork...

"%PYTHON_EXE%" "%SCRIPT_DIR%build_gsplat.py"

pause
