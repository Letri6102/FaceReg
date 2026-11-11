@echo off
setlocal
cd /d "%~dp0"
echo === FaceReg: cai dat moi truong ===

set "PY="
py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY ( py -3.10 -c "import sys" >nul 2>&1 && set "PY=py -3.10" )
if not defined PY ( python -c "import sys; assert sys.version_info[:2] in ((3,10),(3,11))" >nul 2>&1 && set "PY=python" )
if not defined PY (
  echo [LOI] Can Python 3.10 hoac 3.11 ^(insightface chua ho tro tot ban moi hon^).
  echo       Tai tai https://www.python.org/downloads/windows/ va tick "Add python.exe to PATH".
  exit /b 1
)
echo Dung: %PY%

if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv || exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip wheel setuptools cython
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [LOI] Cai dat that bai.
  echo  - Neu loi khi build "insightface": cai "Microsoft C++ Build Tools"
  echo    https://visualstudio.microsoft.com/visual-cpp-build-tools/
  echo    chon "Desktop development with C++", khoi dong lai may roi chay lai setup.bat
  exit /b 1
)
if not exist ".env" (
  copy ".env.example" ".env" >nul
  python -c "import secrets,pathlib;p=pathlib.Path('.env');t=secrets.token_urlsafe(18);p.write_text(p.read_text(encoding='utf-8').replace('ADMIN_TOKEN=change-me','ADMIN_TOKEN='+t,1),encoding='utf-8');print('ADMIN_TOKEN da tao: '+t)"
)
echo.
echo === Xong. ADMIN_TOKEN nam trong file .env. Chay run.bat de khoi dong ===
endlocal
