# Kiểm tra mẫu S2 cho G1 ngày 03 tháng 11 năm 2026

Phạm vi: xác nhận môi trường máy Như và quy tắc cửa sổ bằng extractor thống kê raw có sẵn của Thái. Không triển khai lọc sinh lý, cửa sổ đa tần số hoặc đặc trưng ACC/EMG mới. Đây là chạy mẫu G1 trên một máy, chưa thay nghiệm thu bốn máy hoặc gói feature chính.

## Kết quả

- Môi trường G1 `.venv-g1`: Python 3.12.14, thư viện theo `requirements.lock.txt`; môi trường `.venv` cũ 3.13.9 được giữ nguyên.
- Bộ kiểm tra: 19/19 đạt (14 hạ tầng gốc + 5 kiểm tra nghiệm thu G1); gồm biên nhãn 90%, loại nhãn không dùng, không nối đoạn, phần dư cuối và phát hiện gói đối chiếu sai dữ liệu/môi trường.
- Lượt mẫu gốc dùng cấu hình smoke riêng; ngày 04/11 đã gộp cấu hình đó vào `configs/features_smoke.yaml`. Dùng `--subjects S2` và `--output-dir` để chọn mẫu và lưu riêng.
- Ngực ECG/EDA/Resp, 700 Hz; cửa sổ 60 giây, bước 5 giây, majority ít nhất 90%.
- Lượt chạy mẫu hoàn tất: 395 cửa sổ, gồm 219 Baseline, 113 Stress và 63 Amusement.
- Thời gian xử lý S2 trên Python 3.12.14 do extractor ghi nhận: 5,437 giây; không gồm toàn bộ thời gian khởi động Python, chưa đo RAM đỉnh.
- Đầu ra gồm 8 metadata và 21 thống kê raw có sẵn; chưa phải 68 feature sinh lý trong schema nháp Mai.

Kết quả Python 3.12.14 lưu riêng ngoài Git ở `data/features/nhu_g1_S2_py312_raw_stats_majority_v1/`. Công cụ nghiệm thu ghi `passed` tại `outputs/g1/S2_py312_acceptance.json`, phạm vi `g1_raw_stats_single_package`, `four_machine_acceptance: false`. `complete` trong `run_status.json` chỉ áp dụng S2 đã chọn, không có nghĩa đủ 15 người hoặc hoàn tất G2.

Raw SHA256: `36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23`. Code SHA256 của lượt G1 mới: `d15840d9d97c2e430c72f0ba293253a9ffbad30befd7e73cd4ddcff981635f83`.

Đã so sánh trên cùng máy với lượt Python 3.13.9 trước đây: metadata khớp và sai khác tuyệt đối lớn nhất của 21 thống kê raw bằng 0. Đây là so sánh hai môi trường trên một máy, chưa phải đối chiếu với máy Thái/Mai/Nam.

Cập nhật 04/11: đã chạy cấu hình smoke chung với `--subjects S2 --output-dir data/features/g1_S2_shared_smoke_v1`. Gói mới đạt kiểm tra G1, giữ 395 cửa sổ và khớp chính xác toàn bộ bảng S2 cũ. Báo cáo lượt này ở `outputs/g1/S2_shared_smoke_acceptance.json`; các checksum và thời gian phía trên ghi nhận lượt 03/11, không thay thế provenance của gói mới. Bộ 19 kiểm tra vẫn đạt.

## Cách chạy lại

Thiết lập `configs/local.yaml` chỉ chứa đường dẫn riêng `data_root`, trỏ tới thư mục chứa trực tiếp `S2/`. File này và dữ liệu lớn được Git bỏ qua.

```powershell
.\.venv-g1\Scripts\python.exe scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S2 --output-dir data/features/g1_S2_shared_smoke_v1 --resume
.\.venv-g1\Scripts\python.exe scripts/check_g1_sample.py --features-dir data/features/g1_S2_shared_smoke_v1 --output outputs/g1/S2_py312_acceptance.json
```

Source thay đổi làm đổi chữ ký resume. Khi thay source/config/môi trường, chọn thư mục output mới; không ghi đè hoặc sửa manifest để nhận kết quả từ source khác.

## Trạng thái G1

Mai đã chọn thiết kế G1 tại commit `b5a0cb8`. Như đã chuẩn bị thiết kế bổ sung, môi trường Python 3.12.14 và chạy mẫu S2; cách đối chiếu các máy có trong [bàn giao G1](G1_NHU_BAN_GIAO.md). Còn nhận gói kết quả các máy khác và đối chiếu trước khi xác nhận nghiệm thu chung.

File gốc `S2/S2_readme.txt` ghi cảm biến nhiệt độ ngực không được gắn sát hoàn toàn trong suốt quá trình. Ghi nhận để thiết kế QA sau này, chưa tự sửa tín hiệu hoặc loại subject.

Chỉ tiếp tục triển khai G2 khi người dùng yêu cầu. Xem [thiết kế G1](G1_NHU_THIET_KE_TIEN_XU_LY.md).
