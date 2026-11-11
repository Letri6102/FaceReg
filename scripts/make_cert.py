"""Tạo chứng chỉ HTTPS tự ký để chạy trong mạng LAN phòng thi.

Trình duyệt chỉ cho mở webcam trên HTTPS (hoặc http://localhost). Khi máy thí sinh truy cập server qua IP
LAN (vd https://192.168.1.10:8443) cần HTTPS. Chứng chỉ tự ký sẽ hiện cảnh báo lần đầu, bấm "Tiếp tục".
Triển khai chính thức nên dùng chứng chỉ thật (Let's Encrypt / chứng chỉ của trường) qua nginx hoặc IIS.

    python scripts/make_cert.py 192.168.1.10
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

out = Path(__file__).resolve().parent.parent / "certs"
out.mkdir(exist_ok=True)

hosts = ["localhost", "127.0.0.1", *sys.argv[1:]]
san: list[x509.GeneralName] = []
for h in hosts:
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(h)))
    except ValueError:
        san.append(x509.DNSName(h))

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[-1])])
now = dt.datetime.now(dt.timezone.utc)
cert = (
    x509.CertificateBuilder()
    .subject_name(name)
    .issuer_name(name)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now - dt.timedelta(days=1))
    .not_valid_after(now + dt.timedelta(days=825))
    .add_extension(x509.SubjectAlternativeName(san), critical=False)
    .sign(key, hashes.SHA256())
)
(out / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                                serialization.NoEncryption()))
(out / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
print(f"Đã tạo {out / 'cert.pem'} và {out / 'key.pem'} cho: {', '.join(hosts)}")
