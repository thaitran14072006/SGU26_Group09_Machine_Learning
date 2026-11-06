> Cập nhật theo quyết định nhóm: G1 hiện nghiệm thu bộ 112 thống kê raw, 63 ngực + 49 cổ tay, phiên bản all_sensors_raw_stats_v1. Gói 21 trước đây chỉ là lịch sử chạy mẫu.

# Bàn giao G1 theo README mới

Người phụ trách: Phạm Quỳnh Như. Cập nhật ngày 06/11/2026, sau README và EDA bước 5 giây của Nam. Phạm vi chỉ gồm thiết kế, cấu hình hiện có, kiểm tra biên nhãn và một lượt mẫu S2 theo phân công G1.

## Đầu ra

| File | Nội dung |
| --- | --- |
| `docs/G1_NHU_THIET_KE_TIEN_XU_LY.md` | Đồng bộ ngực/cổ tay, bộ lọc và QA ở mức thiết kế; cửa sổ 60/5, nhãn >=90% |
| `configs/preprocessing_g1_design.yaml` | Hợp đồng thiết kế, runtime_ready false; không dùng để chạy extractor |
| `configs/features_smoke.yaml` | Config raw chung theo README Nam; không tạo smoke riêng |
| `tests/test_g1_windowing.py` | 6 test G1 trên cửa sổ 42.000 mẫu và mặc định CLI |
| `scripts/check_g1_sample.py`, `tests/test_g1_sample.py` | Kiểm tra gói raw S2 và đối chiếu gói từ máy khác |
| `docs/G1_NHU_KIEM_TRA_S2.md` | Kết quả S2 hiện tại và giới hạn nghiệm thu |

CLI mặc định đã sửa sang `features_smoke.yaml`, vì cấu hình cũ đã bị xóa. Hàm cửa sổ mặc định dùng majority >=90%, bước 5 giây. Không thêm thuật toán lọc, HRV/SCR, xử lý đa tần số hoặc feature ACC/EMG của G2.

## Môi trường và chạy mẫu

README dùng tên venv `.venv`; trên máy Như, môi trường Python 3.12.14 đã cài sẵn mang tên `.venv-g1`. Khác tên thư mục không làm đổi phương pháp. Dùng đúng Python trong `.python-version`, `requirements.lock.txt` và cài package editable; không dùng môi trường 3.13.9 cũ để nghiệm thu G1.

Chạy tại thư mục `wesad-stress-detection`, sau khi cấu hình `configs/local.yaml` chỉ chứa `data_root` thật:

```powershell
.\.venv-g1\Scripts\python.exe -m pip check
.\.venv-g1\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-g1\Scripts\python.exe scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S2 --output-dir data/features/g1_s2_all_sensors_raw_stats_v1 --resume
.\.venv-g1\Scripts\python.exe scripts/check_g1_sample.py --features-dir data/features/g1_s2_all_sensors_raw_stats_v1 --output outputs/g1/S2_all_sensors_acceptance.json
```

Đây là đúng cấu hình và đường dẫn mẫu S2 trong README mới, dùng Python 3.12.14 của máy Như. Nếu output đã có chữ ký code/config/môi trường khác, chọn tên thư mục mới bằng `--output-dir`, không sửa manifest để ép resume. Các gói cũ vẫn giữ ngoài Git.

## Bằng chứng đã có

- `pip check` đạt; 35/35 test đạt.
- Mẫu S2 complete: 395 cửa sổ, 219 Baseline, 113 Stress, 63 Amusement; 8 metadata + 112 thống kê raw, kiểm tra gói passed.
- 10 kênh S2 có shape đúng và thời lượng 6.079 giây; chưa khẳng định chất lượng sinh lý.
- Đã đối chiếu đủ bốn gói S2 của Như, Thái, Mai, Nam: CSV giống hệt nhau. Nhóm nghiệm thu theo kết quả đầu ra; phần G1 của Như hoàn tất. Xem [biên bản đối chiếu](G1_DOI_CHIEU_S2_BON_THANH_VIEN.md).

## Cách đối chiếu nghiêm ngặt khi cần tái lập

Nhận ngoài Git cả thư mục gói gồm `S2.csv`, `S2.json`, `config.json`, `feature_schema.json`, `run_status.json`. Các máy cần cùng raw/source, Python patch version và thư viện. Số cửa sổ bằng nhau chưa đủ nghiệm thu.

Đặt gói Mai, ví dụ, tại `data/features/g1_reference_mai/`, rồi chạy:

```powershell
.\.venv-g1\Scripts\python.exe scripts/check_g1_sample.py --features-dir data/features/g1_s2_all_sensors_raw_stats_v1 --reference-dir data/features/g1_reference_mai --output outputs/g1/compare_nhu_mai.json
```

So tên/thứ tự cột, khóa, nhãn và thời gian chính xác; 112 thống kê raw dùng rtol=1e-7, atol=1e-9 theo bàn giao G1. Không áp dung sai này cho 68 feature sinh lý. Gói từ source khác phải được chạy lại bằng source thống nhất trước khi so sánh.

Công cụ không xác thực danh tính máy, từ chối đối chiếu cùng thư mục và luôn giữ `four_machine_acceptance: false`. Cần kết quả/nguồn gói thực tế của các thành viên trước khi ký nghiệm thu chung. Ngày 06/11/2026 đã nhận đủ bốn gói và xác nhận đầu ra giống hệt nhau. Nhóm chấp nhận kết quả này để chốt đối chiếu G1, dù thông tin mã nguồn/môi trường khác nhau; không yêu cầu chạy lại. Quyết định này được ghi riêng trong biên bản, không đổi hành vi của công cụ nghiêm ngặt.
