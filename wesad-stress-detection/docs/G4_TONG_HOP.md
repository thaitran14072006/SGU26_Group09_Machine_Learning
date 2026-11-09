# Tổng hợp các gói tuning G4 đã nhận

Ngày 09/11/2026: đã kiểm tra và hợp nhất **76/120 outer fold**, **31.388 dự đoán ngoài mẫu**, **76 bộ best_params**. Chưa chốt bảng tuning G4 đầy đủ: còn thiếu 44 fold RF/MLP. Không chạy lại tuning trong lượt nhận gói này.

## Phạm vi kết quả

| Gói | Mô hình | Outer test | Fold hai bài toán | Kết luận |
| --- | --- | --- | ---: | --- |
| `g4_nam_knn_svm_phys68_main_v1.rar` | KNN/SVM | Đủ 15 người | 60 | Đạt kiểm tra, đã ghép |
| `g4_nhu_rf_mlp_outer4_handoff.zip` | RF/MLP | S4, S9, S11, S17 | 16 | Đạt kiểm tra đúng lô, đã ghép |

**Cần bổ sung RF/MLP cho S2, S3, S5, S6, S7, S8, S10, S13, S14, S15, S16**, cả ba lớp và nhị phân: 11 người × hai mô hình × hai bài toán = 44 outer fold. Không cần gửi lại các fold KNN/SVM đã đủ hoặc bốn người RF/MLP đã nhận. Thái điều phối các lô còn lại theo phân công nhóm; không suy ra Như phải tự chạy cả 11 người.

## Đã kiểm tra

- Cùng bảng G2 sinh lý68, 6.189 cửa sổ/15 subject, checksum `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba`. Không metadata trong X, không dùng raw112.
- Python patch 3.12.14 và thư viện ghi nhận khớp; giữ chuỗi build/provenance từng producer, kể cả Git metadata null, không sửa để ép giống nhau.
- Snapshot từng run khớp chữ ký và phạm vi source đã khai báo. Lõi train số học chung của bốn file data/evaluation/models/utils có hash `ae97afb79dbbaea35a4e5d33ae3a97c633f1510d8dc716afc73ae18dec3fcfd5`. Wrapper RF/MLP và KNN/SVM khác nhau được giữ đúng nguồn gốc.
- Outer 14 train/1 test; ba inner folds chỉ trong outer train, không giao subject và validation inner phủ đủ 14 người một lần.
- Mỗi model có bốn ứng viên khác nhau, trong miền proposal và thay đổi các tham số đã nêu. Điểm mean CV khớp ba split scores; best_params theo điểm inner CV, không chọn theo outer test. Export best_params khớp từng details.json.
- Checksum từng checkpoint, khóa OOF/nhãn thật và sensor/normalization khớp; không trùng fold/dự đoán, không nhận quick. Tính lại Macro-F1, Balanced Accuracy, confusion matrix từ dự đoán.
- Giữ riêng main nested không cân bằng với baseline LR `class_weight=balanced`. Không fit imputer/scaler hoặc sampler trên toàn bảng.

## So sánh mặc định và tuning trên cùng người

Đã ghép 76 dòng metric với G3 theo task/model/subject/sensor/normalization, kiểm tra cùng số cửa sổ test. Bảng đầy đủ và delta từng người được lưu trong CSV; số dưới đây là trung bình delta Macro-F1 qua đúng những người có kết quả. Chưa kiểm định thống kê, không chọn lại grid để ép điểm tăng.

| Bài toán | Mô hình | Số người | Delta Macro-F1 (G4 − G3) |
| --- | --- | ---: | ---: |
| Ba lớp | KNN | 15 | -0.0054 |
| Ba lớp | SVM | 15 | +0.0122 |
| Ba lớp | RF | 4 | -0.0048 |
| Ba lớp | MLP | 4 | +0.0489 |
| Nhị phân | KNN | 15 | +0.0604 |
| Nhị phân | SVM | 15 | +0.0088 |
| Nhị phân | RF | 4 | -0.0305 |
| Nhị phân | MLP | 4 | +0.0054 |

**Không xếp hạng mean RF/MLP bốn người với mean KNN/SVM 15 người.** Sau khi nhận đủ 44 fold còn lại mới lập bảng main đủ 15 người cho cả bốn mô hình.

## Bàn giao

- Gói tổng hợp hiện tại ngoài Git: `outputs/g4_group_tuning_received_20261109/` và ZIP cùng tên. Có predictions, metrics, best_params, summary với danh sách thiếu, so sánh G3, provenance và checksum. Gói này ghi `waiting_for_tuned_packages`, không giả là main hoàn tất.
- Các gói gốc được giữ trong `outputs/received_g4/`; không sửa chữ ký hoặc chạy lại mô hình của thành viên khác.
- Báo cáo nhỏ: [summary.json](../reports/g4_group/summary.json), [metrics_by_subject.csv](../reports/g4_group/metrics_by_subject.csv), [best_params.json](../reports/g4_group/best_params.json), [default_vs_tuned_by_subject.csv](../reports/g4_group/default_vs_tuned_by_subject.csv), [received_packages.json](../reports/g4_group/received_packages.json).

Khi có lô bổ sung, dùng `scripts/merge_g4_results.py --runs <các gói model gốc> --output <thư mục mới>`. Không ghép đồng thời bảng tổng hợp này và các gói nguồn vì sẽ lặp fold. Công cụ chặn khác lõi train, khác candidate grid/base params giữa các shard của cùng mô hình, sai inner CV và khóa trùng.

Phần baseline/audit của Mai đã hoàn thành trước đó; lượt này đã làm phần kiểm tra/hợp nhất cho mọi gói thực nhận. Còn bước nhận và ghép 44 fold phụ thuộc nhóm. Chưa làm G5, không thêm RAM, không tự commit/push; dự đoán chi tiết và gói nén giữ ngoài Git.
