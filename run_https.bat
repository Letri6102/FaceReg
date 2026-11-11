@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua cai dat. Chay setup.bat truoc.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
if not exist "certs\cert.pem" (
  echo Chua co chung chi. Tao bang lenh:  python scripts\make_cert.py ^<IP-LAN-cua-may-nay^>
  exit /b 1
)
echo FaceReg HTTPS tai https://^<IP-may-nay^>:8443
python -m uvicorn app.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
