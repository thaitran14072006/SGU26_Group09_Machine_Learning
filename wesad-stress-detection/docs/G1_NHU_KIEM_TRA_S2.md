# Kiểm tra S2 G1 với bộ 112 đã chốt

Cập nhật 05/11/2026 theo quyết định nhóm: dùng 112 thống kê raw ngực/cổ tay, phiên bản `all_sensors_raw_stats_v1`. Các lượt 21 trước đây là lịch sử chạy mẫu, không dùng để nghiệm thu bộ đang chọn.

## Kết quả mới trên máy Như

- Python 3.12.14 trong `.venv-g1`, thư viện theo requirements.lock.txt.
- Config `configs/features_smoke.yaml`, processor `wesad.multisensor:process_subject`.
- Cửa sổ 60 giây, bước 5 giây, majority >=90%; kênh giữ tần số gốc.
- Bảng S2: **395 dòng x 120 cột**, gồm 8 metadata và **112 feature**: 63 ngực, 49 cổ tay.
- Nhãn: 219 Baseline, 113 Stress, 63 Amusement.
- 1.204 cửa sổ ứng viên; loại 809 do quy tắc nhãn; không loại thêm vì thiếu mẫu hoặc NaN/Inf ở kênh yêu cầu; giữ 395.
- Thời gian extractor ghi nhận: 10.019 giây, không gồm toàn bộ khởi động Python; chưa đo RAM đỉnh.
- Công cụ kiểm tra G1: passed, đúng header/thứ tự feature, nhóm cảm biến, khóa, metadata, checksum và môi trường.
- Bộ **35/35 test đạt**: 14 hạ tầng, 7 nghiệm thu gói G1, 6 cửa sổ/CLI và 8 đa cảm biến. Có test từ chối gói 21 cũ và schema chia nhóm cảm biến sai.

Raw SHA256: `36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23`.
Code SHA256: `e6bf04894e04f54b1966bc3a34663aac068f9f2fde7d615f196600e1eb8d4e43`.

## Định nghĩa bộ đang dùng

Mỗi kênh có mean, std, min, max, RMS, IQR, range; ACC có x/y/z và magnitude. Tính trực tiếp trong cửa sổ 60 giây. Không áp dụng bộ lọc sinh lý, chuyển đơn vị hoặc công thức tổng hợp ACC/EMG 12 đoạn 5 giây lên gói raw này. Thiết kế bộ lọc trong hợp đồng G1 là tài liệu cho bước triển khai sau, chưa phải định nghĩa feature của bộ 112.

Schema nháp 68 feature của Mai ghi một hướng thiết kế sinh lý trước quyết định bộ 112; không dùng schema đó để kiểm tra bảng hiện tại. Dùng `feature_schema.json` sinh kèm gói 112. Ghi chú S2 về tiếp xúc cảm biến nhiệt độ ngực vẫn cần xem xét; tính hữu hạn không xác nhận chất lượng sinh lý.

## Đối chiếu Nam và giới hạn nghiệm thu

Báo cáo EDA mới của Nam tại `notebooks/reports/tables/eda_step5/` ghi 112 feature và 6.196 cửa sổ/15 người. Cấu hình và số lượng từng lớp S2 khớp lượt chạy này. Đây là so sánh ở mức tóm tắt; chưa có gói đầy đủ để đối chiếu từng khóa/feature giữa các máy.

`four_machine_acceptance` vẫn false. Chỉ chạy S2 trên máy Như trong lượt này, không nhận kết quả 15 người của Nam làm kết quả chạy trên máy Như.

## Kết quả lưu ngoài Git

- `data/features/g1_s2_all_sensors_raw_stats_v1/`: S2.csv, S2.json, all_subjects.csv, config.json, schema, checksum và quality report.
- `outputs/g1/S2_all_sensors_acceptance.json`: nghiệm thu gói đơn.
- `outputs/g1/compare_nam_112_summary.json`: đối chiếu tóm tắt với Nam.

Lệnh chạy lại và nhận gói đối chiếu có trong [bàn giao G1](G1_NHU_BAN_GIAO.md). Các gói cũ giữ nguyên ngoài Git.
