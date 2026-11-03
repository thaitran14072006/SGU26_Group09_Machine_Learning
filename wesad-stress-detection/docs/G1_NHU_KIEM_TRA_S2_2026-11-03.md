# Kiểm tra mẫu S2 cho G1 ngày 03 tháng 11 năm 2026

Phạm vi: xác nhận môi trường máy Như và quy tắc cửa sổ bằng extractor thống kê raw có sẵn của Thái. Không triển khai lọc sinh lý, cửa sổ đa tần số hoặc đặc trưng ACC/EMG mới. Đây là chạy mẫu G1 trên một máy, chưa thay nghiệm thu bốn máy hoặc gói feature chính.

## Kết quả

- Môi trường `.venv`: Python 3.13.9, thư viện theo `requirements.lock.txt`.
- Bộ kiểm tra hạ tầng gốc: 14/14 đạt; gồm biên nhãn 90%, loại nhãn không dùng, không nối đoạn và bỏ phần cuối thiếu độ dài.
- Chạy lại S2 sau khi gỡ triển khai G2, dùng `configs/features_nhu_smoke.yaml`.
- Ngực ECG/EDA/Resp, 700 Hz; cửa sổ 60 giây, bước 5 giây, majority ít nhất 90%.
- Lượt chạy mẫu hoàn tất: 395 cửa sổ, gồm 219 Baseline, 113 Stress và 63 Amusement.
- Thời gian xử lý S2 do extractor ghi nhận: 3,894 giây; không gồm toàn bộ thời gian khởi động Python, chưa đo RAM đỉnh.
- Đầu ra gồm 8 metadata và 21 thống kê raw có sẵn; chưa phải 68 feature sinh lý trong schema nháp Mai.

Kết quả lưu riêng ngoài Git ở `data/features/nhu_g1_S2_chest_raw_stats_majority_v1/`. `complete` trong `run_status.json` chỉ áp dụng S2 đã chọn, không có nghĩa đủ 15 người hoặc hoàn tất G2.

## Cách chạy lại

Thiết lập `configs/local.yaml` chỉ chứa đường dẫn riêng `data_root`, trỏ tới thư mục chứa trực tiếp `S2/`. File này và dữ liệu lớn được Git bỏ qua.

```powershell
.\.venv\Scripts\python.exe scripts/extract_features.py --config configs/features_nhu_smoke.yaml --local configs/local.yaml --subjects S2 --resume
```

Source thay đổi làm đổi chữ ký resume. Khi thay source/config/môi trường, chọn thư mục output mới; không ghi đè hoặc sửa manifest để nhận kết quả từ source khác.

## Trạng thái G1

Đã chuẩn bị môi trường, thiết kế và chạy mẫu S2 trên máy Như. Còn đối chiếu thiết kế với Mai, thống nhất các điểm đang để đề xuất và xác nhận mẫu giữa các máy. Máy Thái trong bản bàn giao dùng Python 3.12.14, nên cần thống nhất môi trường trước nghiệm thu chung.

File gốc `S2/S2_readme.txt` ghi cảm biến nhiệt độ ngực không được gắn sát hoàn toàn trong suốt quá trình. Ghi nhận để thiết kế QA sau này, chưa tự sửa tín hiệu hoặc loại subject.

Chỉ tiếp tục triển khai G2 khi người dùng yêu cầu. Xem [thiết kế G1](G1_NHU_THIET_KE_TIEN_XU_LY.md).
