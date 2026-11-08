# Experiment Log G1–G3 của nhóm 09

Cập nhật theo bằng chứng lưu đến ngày 08/11/2026. Không chạy lại mô hình khi tổng hợp log.

## 1. Cấu hình chung của G3 chính thức

- Dữ liệu: physiological_68_v1, 6.189 cửa sổ, 68 feature (37 ngực + 31 cổ tay), đủ 15 người.
- Cửa sổ 60 giây, bước 5 giây, nhãn đa số ít nhất 90%.
- LOSO: mỗi lượt 14 người train, 1 người test, đủ 15 lượt cho từng model/task.
- Median imputer trong train. LR/KNN/SVM/MLP có StandardScaler trong train. RF/Dummy theo pipeline chung không thêm scaler.
- Không tuning, selection, class balancing hoặc resampling. Seed 42 ở các estimator hỗ trợ và đã khai báo.
- Macro-F1 theo nhãn cố định, zero_division=0. Balanced Accuracy theo các lớp có thật trong test. SD là độ lệch chuẩn mẫu qua 15 người (ddof=1).
- Python 3.12.14, scikit-learn 1.9.1 và thư viện theo file khóa. Producer giữ thông tin build và git_dirty thật.

## 2. Log 12 thí nghiệm G3 chính thức

| Mã | Người | Mô hình | Bài toán | Fold | Dự đoán OOF | Macro-F1 (TB ± SD) | Balanced Accuracy (TB ± SD) | Nhận xét |
|---|---|---|---|---:|---:|---:|---:|---|
| G3-01 | Mai | Dummy | Ba lớp | 15 | 6.189 | 0.2350 ± 0.0023 | 0.3333 ± 0.0000 | Mốc majority để so với các mô hình học từ feature. |
| G3-02 | Mai | Logistic Regression | Ba lớp | 15 | 6.189 | 0.5894 ± 0.1484 | 0.6315 ± 0.1394 | Baseline tuyến tính. Giữ tham số mặc định. |
| G3-03 | Nam | KNN | Ba lớp | 15 | 6.189 | 0.5555 ± 0.1680 | 0.5753 ± 0.1588 | KNN mặc định. Chưa tuning số láng giềng. |
| G3-04 | Nam | SVM RBF | Ba lớp | 15 | 6.189 | 0.6177 ± 0.1647 | 0.6595 ± 0.1559 | SVM RBF mặc định. Chưa tuning C/gamma. |
| G3-05 | Như | Random Forest | Ba lớp | 15 | 6.189 | 0.6480 ± 0.1297 | 0.6700 ± 0.1204 | F1 trung bình gần MLP. Chưa kiểm định chênh lệch. |
| G3-06 | Như | MLP | Ba lớp | 15 | 6.189 | 0.6462 ± 0.1144 | 0.6815 ± 0.1050 | F1 trung bình gần RF. Không ghi cảnh báo hội tụ. |
| G3-07 | Mai | Dummy | Nhị phân | 15 | 6.189 | 0.4123 ± 0.0035 | 0.5000 ± 0.0000 | Mốc majority để so với các mô hình học từ feature. |
| G3-08 | Mai | Logistic Regression | Nhị phân | 15 | 6.189 | 0.8103 ± 0.1938 | 0.8304 ± 0.1752 | Baseline tuyến tính. Giữ tham số mặc định. |
| G3-09 | Nam | KNN | Nhị phân | 15 | 6.189 | 0.7258 ± 0.1823 | 0.7541 ± 0.1650 | KNN mặc định. Chưa tuning số láng giềng. |
| G3-10 | Nam | SVM RBF | Nhị phân | 15 | 6.189 | 0.8424 ± 0.1805 | 0.8663 ± 0.1575 | SVM RBF mặc định. Chưa tuning C/gamma. |
| G3-11 | Như | Random Forest | Nhị phân | 15 | 6.189 | 0.8735 ± 0.1440 | 0.8865 ± 0.1143 | F1 trung bình cao nhất trong bảng mặc định này. |
| G3-12 | Như | MLP | Nhị phân | 15 | 6.189 | 0.7986 ± 0.1846 | 0.8225 ± 0.1671 | Điểm thay đổi giữa subject. Không ghi cảnh báo hội tụ. |

Tổng 180 fold và 74.268 dự đoán ngoài mẫu. Mỗi dòng là một tổ hợp mô hình/bài toán, không phải một fold. Nhóm không ghi cảnh báo trong ba gói mới theo báo cáo Thái. Điểm trung bình gần nhau chưa chứng minh khác biệt có ý nghĩa thống kê. Nguồn điểm: S1/S2, kiểm tra cập nhật: S3.

## 3. Tham số chính đã dùng

| Mô hình | Chủ trì | Tham số chính |
|---|---|---|
| Dummy | Mai | strategy=most_frequent |
| Logistic Regression | Mai | solver=lbfgs, C=1, max_iter=100, random_state=42 |
| KNN | Nam | n_neighbors=5, weights=uniform, metric=minkowski, p=2 |
| SVM RBF | Nam | kernel=rbf, C=1, gamma=scale, probability=false |
| Random Forest | Như | n_estimators=100, max_depth=None, max_features=sqrt, random_state=42, n_jobs=1 |
| MLP | Như | hidden_layer_sizes=(100), activation=relu, solver=adam, alpha=0.0001, max_iter=200, random_state=42 |

Các tham số không liệt kê giữ mặc định của phiên bản thư viện khóa. Tham số đầy đủ Dummy/LR ở S14, RF/MLP ở S15. KNN/SVM theo nguồn/giao thức chung trong S3/S16.

## 4. Log tiền xử lý và các lượt thử

| Mã | Mốc ghi nhận | Giai đoạn | Phạm vi | Cấu hình | Kiểm tra / cách chia | Kết quả | Giới hạn | Nguồn |
|---|---|---|---|---|---|---|---|---|
| G1-01 | 2026-11-06 | G1 | S2 trên bốn máy | 112 raw, cửa sổ 60s, bước 5s, nhãn ≥90% | Đối chiếu CSV cùng S2 | 395 cửa sổ, 219/113/63, bốn CSV giống từng byte | Nhóm chấp nhận theo đầu ra. Nguồn/môi trường khác được ghi trong biên bản. | S4 |
| G2-01 | 2026-11-07 | G2 | Raw112 đủ 15 người | Thống kê raw 60/5, chưa lọc sinh lý | Kiểm tra gói và subject trùng | 6.196 cửa sổ, 112 feature + 8 metadata | Lịch sử raw112. Không dùng làm đầu vào G3 hiện tại. | S5 |
| G2-02 | 2026-11-07 | G2 | Lọc và kiểm tra lô Như | EDA/RESP, tần số gốc, ACC/EMG 12 đoạn 5s | S2 và S4/S9/S11/S17 | S2 395. Lô Như 1.658 cửa sổ. Bảng chẩn đoán ACC/EMG riêng 72 cột. | Bảng chẩn đoán không thay bộ feature chính thức. | S6 |
| G2-03 | 2026-11-07 | G2 | S2 sinh lý68 | Provider lọc/calibrate, hợp đồng physiological_68_v1 | Kiểm tra tích hợp/cấu trúc | 395 cửa sổ, 68 feature + 8 metadata | 3.046 ô NaN có lý do, BVP thiếu nhóm nhịp ở 336/395 cửa sổ. | S7 |
| G2-04 | 2026-11-07 | G2 | Lô sinh lý Như | Cùng provider/hợp đồng 68 | S4, S9, S11, S17 | 1.658 cửa sổ. Nhãn 902/497/259. | Giữ NaN và cờ chất lượng. Chưa impute toàn bảng. | S8 |
| G2-05 | 2026-11-07 | G2 | Lô sinh lý Mai | Cùng phương pháp sinh lý68 | S5, S6, S10 | 1.252 cửa sổ. Nhãn 682/375/195. | BVP thiếu nhiều, gồm S6 thiếu cả nhóm nhịp ở 412/412 cửa sổ. | S9 |
| G2-06 | 2026-11-07 | G2 | Bộ sinh lý68 chung | 68 feature: ngực 37, cổ tay 31. Cùng khóa cửa sổ. | Đủ 15 người. Kiểm tra QA/checksum/schema. | 17.202 ứng viên. Loại 11.006 do nhãn, 7 do biên lọc. Giữ 6.189. | Nhãn 3.370/1.848/971. NaN 37.460 ô. BVP thiếu nhóm nhịp ở 5.312 cửa sổ. | S10 |
| G3-Q01 | 2026-11-08 | Quick | Kiểm thử RF/MLP bằng dữ liệu giả | Mặc định, seed 42 | Train hai subject giả, test S4 giả | Bốn lượt fit cho hai mô hình và hai bài toán | Kiểm tra luồng mã. Không dùng điểm dữ liệu giả làm kết quả đồ án. | S11 |
| G3-Q02 | 2026-11-08 | Quick | WESAD sinh lý68, lô bốn người | RF/MLP mặc định, factory chuẩn bị riêng | Train S9/S11/S17, test S4 | 1.658 cửa sổ đầu vào, 404 test, bốn lượt fit | Quick 3 train / 1 test. Không đưa vào bảng main. | S12 |
| G3-Q03 | 2026-11-08 | Quick | Bảng sinh lý68 đủ 15 người | RF/MLP mặc định, factory chuẩn bị riêng | 14 train, test S4 | 6.189 cửa sổ đầu vào, 404 test, bốn lượt fit | Quick một outer subject. Không phải trung bình LOSO 15 người. | S13 |
| G3-M01 | 2026-11-08 | Main | Sáu mô hình G3 | Mặc định, physiological68, chưa tuning | 15 lượt LOSO cho mỗi model/task | 180 fold, 74.268 dự đoán ngoài mẫu | Thái kiểm tra gói Nam mới theo giao thức chung. Điểm khớp bảng đã công bố. | S1, S2, S3 |

Ngày là mốc trong nguồn bàn giao, không suy đoán timestamp hoặc thời lượng chạy. Các lượt quick, dữ liệu giả và raw112 không được trộn vào bảng điểm sinh lý68 chính thức.

## 5. Ghi chú về cập nhật và giới hạn

Báo cáo Thái S3 xác nhận gói Nam mới theo giao thức chung và các metric khớp bảng đã công bố. S1 vẫn giữ provenance của lượt hợp nhất cũ, nên log dùng S1/S2 cho điểm số và S3 cho trạng thái nguồn mới nhất. Không sửa báo cáo producer cũ để giả cùng runtime. Phạm vi nguồn chung gồm bảy file G3 đã khai báo, không khẳng định mọi file repo hoặc build Python giống nhau.

Bộ G2 có 37.460 ô NaN có lý do, trong đó 5.312/6.189 cửa sổ thiếu nhóm sáu feature nhịp BVP. Nhóm giữ tập cửa sổ chung và xử lý NaN trong train fold. Kết quả G3 không chứng minh đã giải quyết hết chất lượng BVP. Chưa có tuning G4 hoặc kiểm định G5 trong log này.

## 6. Nguồn truy vết

CSV G3 khóa SHA-256: `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba`.

- **S1: Điểm tổng hợp G3** — [reports/g3_group/summary.json](../reports/g3_group/summary.json).
- **S2: 180 dòng metric theo subject** — [reports/g3_group/metrics_by_subject.csv](../reports/g3_group/metrics_by_subject.csv).
- **S3: Kiểm tra giao thức chung mới của Thái** — [reports/g3_group/thai_common_merge_verification.json](../reports/g3_group/thai_common_merge_verification.json).
- **S4: Biên bản đối chiếu G1** — [docs/G1_DOI_CHIEU_S2_BON_THANH_VIEN.md](../docs/G1_DOI_CHIEU_S2_BON_THANH_VIEN.md).
- **S5: Tổng hợp raw112** — [docs/G2_TONG_HOP_RAW112.md](../docs/G2_TONG_HOP_RAW112.md).
- **S6: Kiểm tra tiền xử lý Như** — [reports/g2_preprocessing_nhu/verification.json](../reports/g2_preprocessing_nhu/verification.json).
- **S7: S2 provider sinh lý** — [reports/g2_preprocessing_nhu/S2_provider_basic_verification.json](../reports/g2_preprocessing_nhu/S2_provider_basic_verification.json).
- **S8: Lô sinh lý Như** — [reports/g2_preprocessing_nhu/batch_physiological68_verification.json](../reports/g2_preprocessing_nhu/batch_physiological68_verification.json).
- **S9: Lô sinh lý Mai** — [reports/g2_physiological68/S5_S6_S10_verification.json](../reports/g2_physiological68/S5_S6_S10_verification.json).
- **S10: Nghiệm thu G2** — [reports/g2_physiological68/G2_acceptance.json](../reports/g2_physiological68/G2_acceptance.json).
- **S11: Kiểm tra dữ liệu giả** — [reports/g3_nhu/preparation_check.json](../reports/g3_nhu/preparation_check.json).
- **S12: Quick lô bốn người** — [reports/g3_nhu/physiological68_local4_quick_check.json](../reports/g3_nhu/physiological68_local4_quick_check.json).
- **S13: Quick bảng đủ 15 người** — [reports/g3_nhu/physiological68_group15_quick_check.json](../reports/g3_nhu/physiological68_group15_quick_check.json).
- **S14: Tham số Dummy/LR** — [reports/g3_baselines/results.json](../reports/g3_baselines/results.json).
- **S15: Tham số RF/MLP** — [reports/g3_nhu/main_results.json](../reports/g3_nhu/main_results.json).
- **S16: Mặc định KNN/SVM trong giao thức** — [src/wesad/models.py](../src/wesad/models.py).
