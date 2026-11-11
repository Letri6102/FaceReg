@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua cai dat. Chay setup.bat truoc.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo FaceReg dang chay tai http://localhost:8000  (Ctrl+C de dung)
echo Lan dau chay se tai mo hinh buffalo_l (~280MB), vui long doi.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
