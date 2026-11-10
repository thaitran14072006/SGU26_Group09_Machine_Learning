# Tổng hợp main tuning G4

Ngày 10/11/2026: đã kiểm tra và hợp nhất **120/120 outer fold**, **49.512 dự đoán ngoài mẫu**, **120 bộ best_params**. KNN, SVM, RF và MLP đều đủ 15 người cho hai bài toán; không còn thiếu fold main tuning.

## Nguồn và lượt chạy bổ sung

- Giữ nguyên hai gói đã nhận: `g4_nam_knn_svm_phys68_main_v1.rar` (60 fold KNN/SVM) và `g4_nhu_rf_mlp_outer4_handoff.zip` (16 fold RF/MLP trên S4/S9/S11/S17).
- **Lượt chạy bổ sung** hoàn thành 44 fold RF/MLP trên S2, S3, S5, S6, S7, S8, S10, S13, S14, S15, S16. Không chạy lại 76 fold đã có.
- Dùng `configs/g4_rf_mlp_additional.yaml`: cùng bốn ứng viên RF/MLP, seed 42, nested LOSO, ba inner GroupKFold và normalization `global_train`. Ba tiến trình xử lý các outer subject không giao nhau; checkpoint hợp lệ được resume, sau cùng tổng hợp tuần tự.
- Provenance, chữ ký, môi trường và snapshot từng nguồn được giữ đúng thực tế, kể cả Git metadata null của gói nhận; không ép thành cùng chữ ký wrapper.

## Kiểm tra trước khi hợp nhất

- Cùng bảng G2 sinh lý68: 6.189 cửa sổ/15 người, checksum `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba`. X có 68 feature, không đưa metadata hoặc raw112 vào mô hình.
- Python 3.12.14 và thư viện khóa khớp. Snapshot từng run khớp chữ ký; lõi train chung data/evaluation/models/utils có hash `ae97afb79dbbaea35a4e5d33ae3a97c633f1510d8dc716afc73ae18dec3fcfd5`.
- Outer 14 train/1 test; ba inner fold chỉ thuộc outer train, không giao subject, validation phủ đủ 14 người một lần. Imputer/scaler chỉ fit trong pipeline train.
- Mỗi model có bốn ứng viên khác nhau. Mean CV khớp ba split scores; best_params lấy theo inner Macro-F1 và khớp export, không chọn theo outer test. Cùng mô hình dùng cùng grid/base params trên các shard.
- Checksum checkpoint đạt; khóa OOF, nhãn và số cửa sổ khớp bảng nguồn; không trùng fold/dự đoán và không nhận quick làm main. Đã tính lại Macro-F1, Balanced Accuracy và confusion matrix từ dự đoán.
- Main nested không cân bằng được giữ riêng với baseline LR `class_weight=balanced` và audit resampling.

## Kết quả trên đủ 15 người

Mean và SD mẫu qua 15 người, không phải điểm fit trên toàn bảng. So sánh G3–G4 ghép đúng task/model/subject/sensor/normalization và cùng số cửa sổ test; không kiểm định thống kê hoặc chỉnh grid theo outer test.

| Bài toán | Mô hình | Số người | Macro-F1 mean ± SD | Balanced Accuracy mean ± SD | Delta Macro-F1 G4 − G3 |
| --- | --- | ---: | ---: | ---: | ---: |
| Nhị phân | KNN | 15 | 0.78621 ± 0.12697 | 0.79662 ± 0.12598 | +0.06036 |
| Nhị phân | MLP | 15 | 0.78191 ± 0.18688 | 0.80516 ± 0.16820 | -0.01669 |
| Nhị phân | RF | 15 | 0.86548 ± 0.14122 | 0.87822 ± 0.11412 | -0.00805 |
| Nhị phân | SVM | 15 | 0.85115 ± 0.18556 | 0.87206 ± 0.16330 | +0.00876 |
| Ba lớp | KNN | 15 | 0.55009 ± 0.11489 | 0.56396 ± 0.10727 | -0.00536 |
| Ba lớp | MLP | 15 | 0.61867 ± 0.13381 | 0.65915 ± 0.11044 | -0.02749 |
| Ba lớp | RF | 15 | 0.65396 ± 0.13877 | 0.67314 ± 0.13483 | +0.00601 |
| Ba lớp | SVM | 15 | 0.62987 ± 0.16321 | 0.66396 ± 0.14947 | +0.01217 |

Có 0 fold ghi nhận cảnh báo trong nguồn chạy; chi tiết được giữ trong `reports/g4_group/received_packages.json` và `source_runs/*/folds/*/*/*/details.json` của ZIP. Cảnh báo không bị xóa hoặc che bằng cách tự tăng số vòng lặp.

## Bàn giao

- Bộ chung ngoài Git: `outputs/g4_group_tuning_complete_20261110/` và ZIP cùng tên. Có predictions, metrics, best_params, so sánh G3, cấu hình/checkpoint/snapshot của cả ba nguồn trong `source_runs`, provenance và checksum.
- Báo cáo nhỏ: [summary.json](../reports/g4_group/summary.json), [metrics_by_subject.csv](../reports/g4_group/metrics_by_subject.csv), [best_params.json](../reports/g4_group/best_params.json), [default_vs_tuned_by_subject.csv](../reports/g4_group/default_vs_tuned_by_subject.csv), [received_packages.json](../reports/g4_group/received_packages.json).
- Gói tổng hợp 76 fold ngày 09/11 là lịch sử trung gian; dùng bộ `complete_20261110` cho bảng main hiện tại. Không nối bảng tổng hợp với các gói nguồn vì sẽ lặp fold. Dữ liệu và ZIP giữ ngoài Git.

Chạy lại lượt bổ sung với output mới (hoặc dùng `--resume` khi chữ ký không đổi):

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_experiment.py --config configs/g4_rf_mlp_additional.yaml --outer-subjects S2 S3 S5 S6 S7 S8 S10 S13 S14 S15 S16 --resume
```

Đã đủ **main tuning 120 fold** và phần baseline/audit/hợp nhất. Các so sánh phụ về cân bằng cho mô hình tuned, nhóm cảm biến/feature, ablation và chuẩn hóa theo subject chưa được thực hiện trong lượt này; không gọi toàn bộ các thí nghiệm G4 đã hoàn thành. Chưa làm G5, không thêm phần RAM, không tự commit/push.
