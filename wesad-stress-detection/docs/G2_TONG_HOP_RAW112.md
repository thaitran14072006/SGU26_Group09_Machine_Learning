# Tổng hợp bộ dữ liệu G2 raw112 chung — 07/11/2026

Mai kiểm tra, đối chiếu và tổng hợp bộ dữ liệu chung từ các gói bàn giao. Báo cáo ghi rõ vai trò kiểm tra/schema/hợp nhất và giữ thông tin nguồn chạy trong provenance; không thay tác giả các lần trích xuất. Phạm vi là `all_sensors_raw_stats_v1`: thống kê raw 60 giây/bước 5 giây, majority >=90%; chưa áp bộ lọc G2 hoặc tổng hợp ACC/EMG 5 giây.

## Kết luận kiểm tra

**Đủ 15 subject, 6.196 cửa sổ, 112 feature + 8 metadata.** Không thiếu subject; không NaN/Inf, không trùng (subject_id,window_id). Nhãn: **3.377 Baseline / 1.848 Stress / 971 Amusement**. Mỗi subject nhận một lần; S2 chỉ có 395 cửa sổ, không lặp khi đối chiếu các gói.

Đã kiểm tra processing/schema/thứ tự cột, manifest, chữ ký nội bộ, checksum, tổng dòng/nhãn, ID và lưới cửa sổ; raw hash của 15 người khớp dữ liệu local. all_subjects của gói nhận được kiểm tra so với CSV từng người. Khi xuất bộ chung chỉ nối CSV từng subject một lần, không nối thêm all_subjects vào cùng bảng.

| Subject | Cửa sổ | Baseline | Stress | Amusement |
| --- | ---: | ---: | ---: | ---: |
| S2 | 395 | 219 | 113 | 63 |
| S3 | 401 | 218 | 118 | 65 |
| S4 | 404 | 222 | 117 | 65 |
| S5 | 414 | 230 | 119 | 65 |
| S6 | 412 | 226 | 121 | 65 |
| S7 | 411 | 228 | 118 | 65 |
| S8 | 412 | 224 | 124 | 64 |
| S9 | 411 | 227 | 119 | 65 |
| S10 | 426 | 226 | 135 | 65 |
| S11 | 417 | 227 | 126 | 64 |
| S13 | 417 | 226 | 124 | 67 |
| S14 | 417 | 226 | 126 | 65 |
| S15 | 418 | 226 | 128 | 64 |
| S16 | 415 | 226 | 125 | 64 |
| S17 | 426 | 226 | 135 | 65 |

## Các gói bàn giao và bằng chứng đối chiếu

| Gói | Vai trò trong tổng hợp |
| --- | --- |
| all_sensors_raw_stats_v1.rar do Nam cung cấp | Gói đủ 15 subject đã kiểm tra toàn bộ để xác nhận tính đầy đủ và tính nhất quán nội bộ |
| g2_nhu_raw112_20261107 của Như | Đối chiếu S4/S9/S11/S17, tổng 1.658 cửa sổ |
| g2_raw112_S5_S6_S10_20261107 của Mai | Đối chiếu S5/S6/S10, tổng 1.252 cửa sổ |
| Gói S2 raw112 G1 đã lưu | Tham chiếu bổ sung S2, không nhận thêm một bản S2 vào bảng chung |

Cả tám subject đối chiếu (bảy từ hai lô G2 và S2 tham chiếu cũ) có metadata/khóa và toàn bộ giá trị feature khớp, sai lệch lớn nhất **0**. Các CSV từng subject dùng trong đối chiếu cũng giống từng byte.

Để không trộn mã/môi trường khác nhau vào cùng bộ manifest, bộ xuất dùng một tập manifest/schema nhất quán đã kiểm tra đủ 15 người; các lô bàn giao trùng được dùng xác nhận giá trị, không cộng thêm vào bảng. Nguồn ban đầu và kết quả kiểm tra chéo được lưu trong `config.json -> merge_provenance`, không sửa snapshot/manifest của các gói đã nhận.

## Giới hạn nguồn chạy và nghiệm thu

Snapshot trích xuất của tập manifest đầy đủ ghi **Python 3.13.5, NumPy 2.5.3, pandas 3.0.6**; imbalanced-learn không được ghi nhận. Các gói Như/Mai có source/môi trường khác. Vì vậy đây là kiểm tra đầy đủ và tương đương nội dung raw112, **không phải xác nhận mọi máy cùng source/Python 3.12.14/thư viện khóa**. Không đổi thông tin môi trường hoặc chữ ký để ép đạt kiểm tra nghiêm ngặt.

Không ghép gói 68 sinh lý hoặc bộ thử 175 feature vào bộ raw112. Đủ 15 người raw112 không thay việc triển khai/nghiệm thu bộ feature sinh lý theo proposal. Chưa ghi toàn bộ G2 nhóm hoàn thành và không chạy mô hình.

## Đầu ra chung ngoài Git

- Thư mục: `data/features/g2_raw112_group15_20261107/`.
- Bảng tổng: `data/features/g2_raw112_group15_20261107/all_subjects.csv`.
- ZIP chia sẻ: `data/features/g2_raw112_group15_20261107.zip`.
- Schema, quality report, checksum và manifest từng người nằm cùng thư mục.
- JSON tổng hợp: `outputs/g2_raw112_full_20261107/team_report.json`.
- JSON kiểm tra gói nhận: `outputs/g2_raw112_full_20261107/received_full_audit.json`.
- Đối chiếu các gói và provenance: `outputs/g2_raw112_full_20261107/consolidation.json`.
- SHA-256 bảng tổng: `de34cf16aa21e5e15bc61f5c4ac22b859fc7237c048e82fd1b9e63ddef662b91`.
- SHA-256 ZIP chung: `d3bae0c5e1688e4d9490bbe142bc42d770485f6151b2a9a7553e22ed66e0a808`.

Công cụ cơ bản: `scripts/check_merge_raw112.py`, `src/wesad/raw_feature_packages.py`; đọc ZIP/RAR hoặc thư mục, kiểm tra trước khi ghép, chặn phiên bản/định nghĩa khác và subject xung đột. Tám test đạt, gồm đủ 15 người, chống lặp S2/all_subjects, hỏng checksum, NaN, lệch phiên bản/provenance và đường dẫn archive không hợp lệ. File dữ liệu lớn được .gitignore bỏ qua. Không tự commit/push.

## TODO sau bản cơ bản

- Nhóm thống nhất điều kiện source/môi trường cho lượt tái lập chung; nếu cần điều kiện nghiêm ngặt thì chạy lại đồng bộ thay vì sửa metadata.
- Khi tích hợp tiền xử lý mới, dùng phiên bản/schema riêng và kiểm tra lại; không đổi công thức raw112 này.
- Các gói đến sau tiếp tục so subject trùng và raw hash, không append thêm bản subject đã có.
