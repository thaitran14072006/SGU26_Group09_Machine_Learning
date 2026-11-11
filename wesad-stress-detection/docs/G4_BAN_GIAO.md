# Bàn giao G4 phần Mai

Ngày 09/11/2026. Theo phân công, Mai phụ trách baseline cân bằng lớp, kiểm tra resampling và hợp nhất best_params/dự đoán hai bài toán. Các kết quả baseline/audit dưới đây được giữ riêng; ngày 10/11 đã hoàn thành thêm lượt chạy bổ sung và hợp nhất main tuning.

## Đã thực hiện

- Chạy **Logistic Regression với `class_weight=balanced`**, cả ba lớp và nhị phân, đủ 15 fold LOSO: **30 fold, 12.378 dự đoán ngoài mẫu**. Dùng đúng bảng sinh lý68 đã chốt, 6.189 cửa sổ; checksum được khóa trong `configs/g4_balanced_baseline.yaml`.
- Giữ LR mặc định, `max_iter=100`, seed 42; chỉ thêm class_weight. Không tuning, selection hoặc resampling cho LR. Imputer và scaler chỉ fit trên train của từng fold. Không có cảnh báo hội tụ trong lượt chạy.
- So sánh ghép theo đúng subject với LR mặc định G3; tái sử dụng Dummy G3 làm tham chiếu, không chạy lại G3. Lưu trọng số lớp tính từ train của từng fold.
- Kiểm tra **240 lượt resampling**: 15 outer × hai bài toán × bốn scope train (ba inner và một refit) × KNN/MLP. Chỉ fit preprocessing và RandomOverSampler để audit, không huấn luyện classifier KNN/MLP trên dữ liệu thật trong lượt audit này.
- Sample indices chỉ trỏ tới train, nhãn resample đúng, số lượng lớp sau ROS bằng nhau; **số hàng validation/test không tăng**. Xác nhận pipeline LR/SVM/RF dùng class_weight khi bật nhánh cân bằng.
- Công cụ hợp nhất G4 đã có: kiểm tra OOF, nhãn, metric, inner splits chỉ trong outer train, bốn ứng viên đã khóa, best_params theo điểm inner CV, checksum và source snapshot. Nhận full run hoặc lô outer-subject được khai báo rõ; không nhận quick làm main, không nhận trùng fold hoặc trộn grid khác nhau cho cùng mô hình.

## Kết quả baseline

Trung bình qua 15 người, không phải metric fit trên toàn bảng. SD mẫu và điểm từng người được lưu trong báo cáo.

| Bài toán | LR | Macro-F1 | Balanced Accuracy |
| --- | --- | ---: | ---: |
| Ba lớp | Mặc định G3 | 0.58939 | 0.63147 |
| Ba lớp | class_weight balanced | 0.58527 | 0.63037 |
| Nhị phân | Mặc định G3 | 0.81027 | 0.83036 |
| Nhị phân | class_weight balanced | 0.81028 | 0.83188 |

Cân bằng lớp không cải thiện đều các chỉ số. Giữ nguyên kết quả giảm; không đổi tham số theo outer test và chưa thực hiện kiểm định ý nghĩa thống kê.

## Tệp và nguồn chạy

- Gói ngoài Git: `outputs/g4_balanced_lr_physiological68_v1/` và ZIP cùng tên. Có dự đoán, metric/summary, splits, checkpoint, config/environment/log, class weights theo fold, so sánh G3, audit resampling và checksum.
- Báo cáo nhỏ: [results.json](../reports/g4_baselines/results.json), [comparison_by_subject.csv](../reports/g4_baselines/comparison_by_subject.csv) (30 dòng), [resampling_audit.csv](../reports/g4_baselines/resampling_audit.csv) (240 dòng).
- Python 3.12.14 và thư viện khóa. Manifest ghi nguồn lúc fit `1a7b337`, dirty; source snapshot runtime được giữ nguyên. Module kiểm tra/nhận lô được hoàn thiện sau lượt fit và lưu riêng trong `verification_source.zip`, không sửa chữ ký producer để giả cùng nguồn.
- Bộ kiểm thử tổng: **108/108 đạt**. Sau khi bổ sung chặn trộn grid giữa các shard, kiểm thử G4 riêng **9/9 đạt**. Log: `outputs/g4_all_tests_final.txt` và `outputs/g4_final_checks.txt`.

## Chạy lại

Chọn output mới nếu code/settings thay đổi. Resume chỉ tái sử dụng checkpoint đúng chữ ký và checksum; không sửa manifest của lượt cũ để ép dùng lại.

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_g4_baselines.py --quick-subject S2 --output outputs/g4_balanced_quick_lan2
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_g4_baselines.py --output outputs/g4_balanced_main_lan2 --resume
```

## Tổng hợp main tuning

Ngày 10/11/2026: đã hoàn thành lượt chạy bổ sung 44 fold RF/MLP và hợp nhất với 76 fold đã nhận. Bảng main đủ **120/120 fold, 49.512 dự đoán, 120 bộ best_params**, bốn mô hình × hai bài toán × 15 người. Không còn chờ các fold main tuning.

Đã kiểm tra source snapshot/checksum, cùng lõi train và grid giữa các shard, OOF/nhãn, inner CV, best_params và metric tính lại. So sánh mặc định–tuning trên đủ 15 người được lưu riêng. Xem [tổng hợp main G4](G4_TONG_HOP.md) và `outputs/g4_group_tuning_complete_20261110.zip`; gói này giữ cả checkpoint và bằng chứng của từng nguồn trong `source_runs`.

Ngày 11/11/2026 đã hoàn thành thêm so sánh cân bằng lớp cho KNN/SVM/RF/MLP trên 120 fold, giữ best_params của từng fold main. Kết quả và cách chạy lại nằm trong [tổng hợp G4](G4_TONG_HOP.md#so-sánh-cân-bằng-lớp). Các so sánh nhóm cảm biến/feature, ablation và chuẩn hóa theo subject còn lại.
