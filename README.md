# FaceReg – Xác thực khuôn mặt thí sinh (chống thi hộ)

Hệ thống web so khớp **1:1** khuôn mặt thí sinh trước camera với **ảnh 4x6** đã đăng ký, kèm **chống giả mạo** (dùng ảnh in, video) và **giám thị duyệt** các trường hợp không chắc chắn.

```
[Đăng ký]  Ảnh 4x6 ──► Kiểm tra chất lượng ──► Căn chỉnh 112x112 ──► Vector từng mô hình ──► SQLite
                        (1 mặt, nét, thẳng, đủ sáng)          (ảnh + ảnh lật)     (ArcFace, AdaFace...)

[Check-in] Nhập SBD ──► Webcam ──► Thử thách ngẫu nhiên: nhìn thẳng → quay trái/phải/ngẩng (x3) → nhìn thẳng
                                    │  server kiểm tra TỪNG khung hình: đúng tư thế, 1 người, không đổi mặt giữa chừng
                                    │  (+ liveness thụ động MiniFASNet nếu bật)
                                    ▼
                        So khớp với ảnh đăng ký (+ mẫu webcam các lần trước)
                        điểm = trung bình có trọng số cosine của các mô hình
                          ≥ ACCEPT  → ĐẠT, cấp token giám sát
                          giữa      → CHỜ DUYỆT (giám thị xem 2 ảnh cạnh nhau)
                          < REJECT  → KHÔNG ĐẠT

[Trong giờ thi] monitor.js gửi ảnh ngẫu nhiên ~30s/lần → cảnh báo vắng mặt / nhiều người / đổi người
```

## 1. Cấu trúc thư mục

```
FaceReg/
├── app/
│   ├── main.py          # FastAPI: trang web + REST API
│   ├── services.py      # Nghiệp vụ: đăng ký, check-in, duyệt, giám sát
│   ├── face_engine.py   # Phát hiện (SCRFD), căn chỉnh, nhiều mô hình nhận diện + gộp điểm, chất lượng ảnh
│   ├── liveness.py      # Thử thách chủ động + liveness thụ động ONNX
│   ├── quality.py       # Kiểm tra ảnh 4x6
│   ├── db.py            # SQLite
│   ├── config.py        # Đọc cấu hình từ .env
│   └── bootstrap.py
├── static/              # Giao diện: index, checkin, admin, js/monitor.js (nhúng vào trang thi)
├── scripts/
│   ├── bulk_enroll.py          # Đăng ký hàng loạt từ thư mục ảnh / CSV
│   ├── calibrate_threshold.py  # So sánh mô hình + hiệu chỉnh ngưỡng trên dữ liệu thật
│   ├── export_adaface_onnx.py  # Xuất AdaFace IR-101 sang ONNX
│   ├── reembed.py              # Tính bổ sung vector khi thêm mô hình mới
│   ├── export_antispoof_onnx.py# Xuất mô hình chống giả mạo MiniFASNet sang ONNX
│   └── make_cert.py            # Tạo chứng chỉ HTTPS tự ký cho mạng LAN
├── tests/test_services.py      # Kiểm thử luồng nghiệp vụ (engine giả lập)
├── setup.bat / run.bat / run_https.bat
├── .env.example
├── requirements.txt
└── Dockerfile
```

## 2. Cài đặt trên Windows

1. Cài **Python 3.10 hoặc 3.11** (python.org, tick *Add python.exe to PATH*). Gói `insightface` chưa ổn định trên Python mới hơn.
2. Cài **Microsoft C++ Build Tools** → chọn *Desktop development with C++* (để build `insightface`).
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Mở `cmd` tại thư mục này:
   ```bat
   setup.bat
   run.bat
   ```
   `setup.bat` tạo môi trường `.venv`, cài thư viện, tạo file `.env` với `ADMIN_TOKEN` ngẫu nhiên (in ra màn hình).
   Lần chạy đầu `run.bat` tự tải mô hình `buffalo_l` (~280MB) vào `models/`.
4. Mở trình duyệt:
   - http://localhost:8000/admin – quản trị (đăng nhập bằng `ADMIN_TOKEN` trong `.env`)
   - http://localhost:8000/checkin – thí sinh check-in
   - http://localhost:8000/docs – tài liệu API

Kiểm thử nhanh logic (không cần mô hình): `.venv\Scripts\python tests\test_services.py`

## 3. Sử dụng

### Đăng ký ảnh 4x6
- **Từng người:** trang Quản trị → *Đăng ký thí sinh*. Ảnh không đạt sẽ báo rõ lý do (mờ, nghiêng, quá nhỏ, nhiều mặt...).
- **Hàng loạt:**
  ```bat
  .venv\Scripts\python scripts\bulk_enroll.py --dir D:\anh_4x6                    :: tên file = SBD
  .venv\Scripts\python scripts\bulk_enroll.py --dir D:\anh_4x6 --csv ds.csv      :: CSV: code,full_name[,photo]
  ```
  Kết quả ghi vào `enroll_report.csv` (mở bằng Excel) để yêu cầu thí sinh nộp lại ảnh không đạt.

Yêu cầu ảnh: **file ảnh gốc** (không chụp lại bản in), chụp trong 6 tháng, nhìn thẳng, không kính râm/khẩu trang, mặt ≥ 112px.

### Check-in
Link cho thí sinh có thể điền sẵn: `http://<server>/checkin?code=2024001&exam=CK_HK1_2026`
(khi có `exam` trong link, ô mã kỳ thi sẽ ẩn).

### Giám thị duyệt
Trang Quản trị → *Chờ duyệt*: ảnh đăng ký bên trái, ảnh chụp lúc check-in bên phải, kèm điểm số và lý do. Bấm **Đúng người** hoặc **Không đúng**. Màn hình thí sinh tự cập nhật kết quả.

## 4. Tích hợp với hệ thống MOOC

**a. Chuyển hướng sau khi đạt** – trong `.env`:
```
CHECKIN_SUCCESS_URL=https://mooc.example.edu.vn/exam/start?sbd={code}&exam={exam}&ftoken={token}
```

**b. Server MOOC kiểm tra thí sinh đã check-in chưa** (gọi server-to-server, không để lộ ADMIN_TOKEN ra trình duyệt):
```
GET /api/admin/checkins?candidate_code=2024001&exam_code=CK_HK1_2026&status=ACCEPT
Header: X-Admin-Token: <ADMIN_TOKEN>
```

**c. Giám sát trong giờ thi** – nhúng vào trang làm bài:
```html
<script src="https://<facereg>/static/js/monitor.js"></script>
<script>
  const mon = FaceRegMonitor.start({
    server: "https://<facereg>", token: "<ftoken>", intervalSec: 30,
    onResult: (s) => { if (s !== "OK") console.warn("FaceReg:", s); }
  });
</script>
```
Thêm domain trang thi vào `CORS_ORIGINS`. Số cảnh báo hiển thị ở tab *Lịch sử check-in*.

## 5. Triển khai

- **Phòng thi (mạng LAN):** trình duyệt chỉ mở webcam trên `https://` hoặc `localhost`.
  ```bat
  .venv\Scripts\python scripts\make_cert.py 192.168.1.10
  run_https.bat
  ```
  Máy thí sinh vào `https://192.168.1.10:8443/checkin` (chấp nhận cảnh báo chứng chỉ tự ký lần đầu).
- **Chính thức:** đặt sau nginx/IIS có chứng chỉ thật, hoặc dùng `Dockerfile` trên Linux. Chạy **1 process** là đủ cho vài chục lượt check-in đồng thời trên CPU; nếu cần nhiều hơn dùng GPU (`USE_GPU=true`, cài `onnxruntime-gpu`) hoặc `--workers N` (dữ liệu phiên lưu trong SQLite nên chạy nhiều worker được).

## 6. Hiệu chỉnh ngưỡng (bắt buộc trước khi dùng thật)

Ngưỡng mặc định `THRESHOLD_ACCEPT=0.45`, `THRESHOLD_REJECT=0.28` chỉ là điểm xuất phát. Ảnh 4x6 in/scan so với webcam cho điểm thấp hơn ảnh cùng nguồn, nên cần đo trên dữ liệu của bạn:

1. Thu ~100 người trở lên (càng nhiều càng tốt): mỗi người 1 ảnh `ref.jpg` (ảnh 4x6) + vài ảnh webcam, xếp vào `calib/<SBD>/`.
2. `.venv\Scripts\python scripts\calibrate_threshold.py --dir calib --target-far 0.001`
3. Chép ngưỡng đề xuất vào `.env`.

Nguyên tắc: với kỳ thi, **nhận nhầm người khác nguy hiểm hơn từ chối nhầm** (đã có giám thị xử lý vùng REVIEW).

## 7. Nâng độ chính xác: gộp nhiều mô hình

Mặc định hệ thống dùng một mô hình (ArcFace R50 trong `buffalo_l`) + **flip TTA** (trích đặc trưng trên ảnh và ảnh lật ngang). Có thể thêm mô hình thứ hai và **gộp điểm** – hai mô hình khác nhau thường sai ở những trường hợp khác nhau, nên gộp lại thường giảm lỗi.

**Thêm AdaFace IR-101 (WebFace12M)** – mô hình mở mạnh nhất hiện có, đặc biệt tốt với ảnh chất lượng kém:
1. Xuất ONNX theo hướng dẫn đầu file `scripts/export_adaface_onnx.py` (cần PyTorch, làm một lần).
2. Trong `.env`:
   ```
   EXTRA_MODELS=adaface:models/adaface/adaface_ir101_webface12m.onnx
   FUSION_WEIGHTS=                 # trống = 50/50
   ```
3. Thí sinh đã đăng ký **không cần đăng ký lại**: hệ thống tự tính vector mới từ ảnh 4x6 đã lưu (hoặc chạy trước `python scripts\reembed.py`).

**Chọn cấu hình bằng dữ liệu, không đoán:** chạy `scripts/calibrate_threshold.py` – script in bảng so sánh từng mô hình và điểm gộp:
```
Cấu hình                   | EER    | FNMR @ FMR=0.01 | FNMR @ FMR=0.001 | ...
buffalo_l                  | ...    | ...             | ...
adaface_ir101_webface12m   | ...    | ...             | ...
GỘP (FUSION_WEIGHTS)       | ...    | ...             | ...
```
FNMR (tỷ lệ người thật bị từ chối) thấp hơn ở cùng FMR (tỷ lệ nhận nhầm) là tốt hơn. Script cũng gợi ý trọng số gộp tốt nhất. Muốn dùng **chỉ** AdaFace: `FUSION_WEIGHTS=buffalo_l:0`.

> Mỗi khi đổi `EXTRA_MODELS`, `FUSION_WEIGHTS` hoặc `FLIP_TTA`, phân bố điểm thay đổi → **phải chạy lại hiệu chỉnh ngưỡng**.
> Trang quản trị hiển thị cả điểm gộp và điểm từng mô hình cho mỗi lượt check-in.

Chi phí: mỗi mô hình thêm tốn khoảng vài trăm mili-giây CPU cho mỗi khung hình "nhìn thẳng" (chỉ 2 khung/lượt check-in); các bước quay đầu không bị ảnh hưởng.

## 8. Liveness thụ động (khuyến nghị bật)

Thử thách chủ động chặn được ảnh in và video quay sẵn thông thường, nhưng không chặn được người cầm điện thoại phát video thời gian thực. Nên bật thêm mô hình MiniFASNet:

1. Làm theo hướng dẫn đầu file `scripts/export_antispoof_onnx.py` để xuất `.onnx` vào `models/anti_spoof/`.
2. Trong `.env`:
   ```
   ANTISPOOF_MODELS=models/anti_spoof/2.7_80x80_MiniFASNetV2.onnx,models/anti_spoof/4_0_0_80x80_MiniFASNetV1SE.onnx
   ```
3. **Kiểm thử** với mặt thật, ảnh in, ảnh trên điện thoại/màn hình, xem cột *Liveness* trong *Lịch sử check-in*, rồi chỉnh `ANTISPOOF_THRESHOLD`. Nếu mặt thật luôn cho điểm thấp, thử `ANTISPOOF_INPUT_SCALE=0.00392157` (chuẩn hóa /255).

Khi nghi giả mạo, hệ thống **không tự từ chối** mà chuyển sang giám thị duyệt.

## 9. API chính

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| POST | `/api/checkin/start` | `{candidate_code, exam_code}` → phiên + chuỗi bước ngẫu nhiên |
| POST | `/api/checkin/{sid}/frame` | `{image: dataURL}` → phản hồi tư thế, tiến độ |
| POST | `/api/checkin/{sid}/finish` | Kết quả ACCEPT/REVIEW/REJECT (không trả điểm cho thí sinh) |
| GET | `/api/checkin/{sid}/status` | Kết quả hiện tại (dùng khi chờ duyệt) |
| POST | `/api/monitor` | `{token, image}` → OK/NO_FACE/MULTI_FACE/MISMATCH |
| POST | `/api/admin/candidates` | multipart `code, full_name, photo, replace` |
| GET/DELETE | `/api/admin/candidates[/{code}]` | Danh sách / xóa thí sinh |
| GET | `/api/admin/checkins` | Lọc theo `status, exam_code, candidate_code` (kèm điểm từng mô hình) |
| POST | `/api/admin/checkins/{id}/review` | `{decision: APPROVE/REJECT, note}` |

API admin yêu cầu header `X-Admin-Token`.

## 10. Bảo mật, pháp lý, giới hạn

- **Bản quyền mô hình:** mô hình huấn luyện sẵn của InsightFace (`buffalo_l`, `antelopev2`) chỉ được phép dùng cho **nghiên cứu phi thương mại**; muốn dùng chính thức cần mua giấy phép thương mại của InsightFace. Trọng số AdaFace (mã MIT) được huấn luyện trên WebFace – bộ dữ liệu chỉ cấp phép nghiên cứu. Khi triển khai chính thức, hãy dùng mô hình có giấy phép phù hợp (InsightFace thương mại, hoặc nhà cung cấp eKYC trong nước): chỉ cần viết một lớp `Recognizer` mới có hàm `embed_batch()` trong `face_engine.py`.
- **Dữ liệu sinh trắc học** là dữ liệu cá nhân nhạy cảm theo quy định bảo vệ dữ liệu cá nhân của Việt Nam: cần sự đồng ý của thí sinh, giới hạn người truy cập, sao lưu/mã hóa thư mục `data/`, và có chính sách xóa sau kỳ thi.
- **Gian lận từ khâu đăng ký** (nộp ảnh của người thi hộ ngay từ đầu) không thể phát hiện bằng so khớp. Nên đối chiếu ảnh 4x6 với ảnh CCCD/hồ sơ gốc khi tiếp nhận.
- Giám sát định kỳ (`/api/monitor`) **không có liveness**; mục đích là phát hiện vắng mặt/đổi người, các cảnh báo cần người xem lại.
- Không tự động loại thí sinh chỉ dựa vào điểm số: mọi trường hợp bất thường đều đi qua giám thị.
