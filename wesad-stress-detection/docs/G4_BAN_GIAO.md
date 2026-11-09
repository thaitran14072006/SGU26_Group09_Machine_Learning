# Bàn giao G4 phần Mai

Ngày 09/11/2026. Theo phân công, Mai phụ trách baseline cân bằng lớp, kiểm tra resampling và hợp nhất best_params/dự đoán hai bài toán. Đã hoàn thành các việc độc lập dưới đây; không thay thế phần tuning và các biến thể của chủ mô hình.

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

## Phần còn phụ thuộc nhóm

Như đã commit báo cáo **16 fold RF/MLP trên S4/S9/S11/S17**. Đây là lô bốn outer test, không phải đủ 15 người. Máy Mai chưa nhận ZIP dự đoán/fold của lô này nên chưa đưa vào số fold được kiểm tra độc lập. Không kết luận Như chưa chạy.

Khi nhận các gói tuning, dùng `scripts/merge_g4_results.py --runs <các thư mục kết quả đã giải nén> --output <thư mục mới>`. Chạy `--help` để xem tham số. Kết quả main cần đủ bốn mô hình × hai bài toán × 15 outer fold = **120 fold**, mỗi fold có best_params riêng và bằng chứng inner CV. Các wrapper có thể khác nhau nhưng snapshot phải khớp chữ ký từng producer và cùng lõi train số học; cùng mô hình phải dùng cùng bốn ứng viên/base params trên các shard.

Đã thử luồng chưa nhận gói: `outputs/g4_group_tuned_pending_20261109/summary.json` ghi `waiting_for_tuned_packages`, chưa tạo dự đoán giả hoặc gọi main hoàn tất. Baseline LR cân bằng không thay một trong bốn mô hình tuned. So sánh cân bằng lớp cho SVM/RF/KNN/MLP cần kết quả từ chủ mô hình; kiểm tra sampler sẵn sàng để họ dùng, chưa chạy thay các biến thể đó.

Phạm vi lượt này là phần Mai ở mục G4 và cơ chế cân bằng liên quan. Chưa làm đóng góp/ablation nhóm feature RF ở mục so sánh phụ, chuẩn hóa theo subject, G5 hoặc phần RAM. Không gọi toàn bộ G4 của nhóm đã hoàn tất. Chưa commit/push.
