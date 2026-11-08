# Bàn giao G3 Dummy và Logistic Regression

Cập nhật nhận đủ gói 08/11/2026: đã kiểm tra/hợp nhất Dummy/LR/KNN/SVM/RF/MLP, **180 fold, 74.268 dự đoán**; phần tổng hợp của Mai đã hoàn thành. Xem [báo cáo cuối](G3_TONG_HOP.md) về review runner cũ của Nam và giới hạn snapshot runtime, không tuyên bố mọi code hash giống nhau. Các trạng thái chờ bên dưới là lịch sử trước khi nhận đủ gói.

Ngày 08/11/2026: đã hoàn thành Dummy majority và Logistic Regression mặc định cho hai bài toán, mỗi mô hình/bài toán đủ 15 fold LOSO. Phần kiểm tra nhãn, metric theo subject và công cụ hợp nhất dự đoán đã có. Tổng hợp đủ sáu mô hình còn chờ Nam gửi KNN/SVM và Như gửi RF/MLP; không ghi G3 cả nhóm hoàn tất.

## Dữ liệu và quy tắc chạy

- Chỉ dùng `data/features/g2_physiological68_group15_20261107/all_subjects.csv`: 6.189 cửa sổ, 68 feature, đủ 15 người. SHA-256 `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba` được khóa trong `configs/defaults.yaml`.
- X gồm đúng 68 cột schema sinh lý; không có subject, thời gian, nhãn hoặc quality flag. Không đổi detector hoặc tính lại G2.
- Mỗi fold giữ một người test, 14 người train. Dùng median imputer trong pipeline, `keep_empty_features=True`; LR có StandardScaler học trên train. Không fit trước trên toàn bảng, không loại hàng do BVP thiếu.
- Dummy dùng `most_frequent`. LR giữ tham số mặc định của thư viện, `max_iter=100`, `solver=lbfgs`, chỉ đặt `random_state=42`. Không cân bằng lớp, tuning, selection hoặc resampling; không tự tăng max_iter.
- Ba lớp theo thứ tự `[1,2,3]`; nhị phân `[0,1]`, stress là 1. Macro-F1 dùng nhãn cố định và `zero_division=0`. Fold thiếu lớp train dừng báo lỗi; nếu test thiếu lớp, Macro-F1 vẫn dùng tập nhãn cố định, Balanced Accuracy theo các lớp thực có trong test. Dữ liệu thực tế hiện đủ lớp ở cả 15 fold.
- Chạy trên nhóm kết hợp đủ 68 feature; các thử nghiệm so sánh ngực/cổ tay hoặc chuẩn hóa khác là lượt riêng về sau, không trộn vào bảng mặc định này.

## Kết quả ngoài mẫu

Số dưới đây là trung bình ± độ lệch chuẩn mẫu (`ddof=1`) qua 15 subject, không phải metric fit trên toàn bảng. Confusion matrix gộp dự đoán ngoài mẫu được lưu trong summary/results JSON.

| Bài toán | Mô hình | Macro-F1 | Balanced Accuracy |
| --- | --- | ---: | ---: |
| Ba lớp | Dummy | 0.2350 ± 0.0023 | 0.3333 ± 0.0000 |
| Ba lớp | LR | 0.5894 ± 0.1484 | 0.6315 ± 0.1394 |
| Nhị phân | Dummy | 0.4123 ± 0.0035 | 0.5000 ± 0.0000 |
| Nhị phân | LR | 0.8103 ± 0.1938 | 0.8304 ± 0.1752 |

Đủ **60 fold và 24.756 dòng dự đoán** (6.189 × hai mô hình × hai bài toán). Mỗi model/task có đủ 15 subject và đúng tập cửa sổ đầu vào. Không trùng khóa model/task/sensor/normalization/subject/window, không sai nhãn tham chiếu; train/test không giao subject. Đã tính lại metric và confusion matrix từ dự đoán để kiểm tra các bản lưu, kiểm tra checksum fold và nguồn snapshot. Không ghi nhận cảnh báo ở 60 fold. Bộ kiểm thử: **84/84 đạt**, gồm 8 test G3; log ngoài Git `outputs/g3_all_tests_final.txt`.

Đã thử S2 trước khi chạy chính thức; gói quick cuối cùng là `outputs/g3_baselines_physiological68_v1_quick_final/`, chỉ 4 fold/1.580 dự đoán, gắn nhãn quick và không đưa vào bảng chính. Các lượt thử trước trong `outputs/` chỉ là lịch sử; lấy gói `_v1` dưới đây làm kết quả bàn giao.

## Tệp bàn giao

- Gói chính: `outputs/g3_baselines_physiological68_v1/` và ZIP cùng tên, ngoài Git.
- Có `predictions.csv`, `metrics_by_subject.csv`, `verified_metrics_by_subject.csv`, `summary.json`, `splits.json`, `config.yaml`, `config.json`, `environment.txt/json`, `model_parameters.json`, `run.log`, verification và fold checkpoint/checksum.
- `source_snapshot.zip` chứa bảy file mã G3 thực thi đã khai báo. Hash nguồn được tính trên byte chuẩn hóa LF và đường dẫn tương đối dùng `/`; không bị lệch chỉ vì CRLF/LF hoặc có thêm script chẩn đoán G2. Thay nội dung nguồn thực vẫn bị phát hiện. Git nguồn lúc chạy là `8b1e1cd`, `git_dirty=true` do các file G3 đang được triển khai; giữ đúng bằng chứng này, không giả là nguồn sạch. Snapshot đi kèm để tái tạo sau commit.
- Báo cáo nhỏ đưa Git: [results.json](../reports/g3_baselines/results.json) và [metrics_by_subject.csv](../reports/g3_baselines/metrics_by_subject.csv), chỉ 60 dòng metric; dự đoán chi tiết và ZIP giữ ngoài Git.
- Hạn chế BVP của G2 còn nguyên; điểm số là kết quả với median imputation học trong train, không phải bằng chứng đã sửa chất lượng BVP.

## Chạy lại và chia công việc

Tại project, dùng Python 3.12.14 và thư viện khóa. Quick và main có chữ ký/thư mục riêng, không dùng quick làm kết quả chính. Chọn output mới nếu code/settings/data thay đổi; `--resume` chỉ dùng lại checkpoint có cùng chữ ký và checksum.

```powershell
# Mai: thử S2; sau đó chạy đủ hai mô hình và hai bài toán
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_g3_baselines.py --quick-subject S2 --output outputs/g3_baselines_quick_lan2
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_g3_baselines.py --output outputs/g3_baselines_main_lan2 --resume

# Nam và Như có thể chạy song song bằng cùng code/config/bảng 68
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_g3_baselines.py --models knn svm --output outputs/g3_knn_svm --resume
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_g3_baselines.py --models rf mlp --output outputs/g3_rf_mlp --resume
```

Khung LOSO của Thái được tái sử dụng; chỉ bổ sung nhận diện phạm vi mã G3, không đổi thuật toán chia fold. Trong bản runner gốc, không dùng code hash provenance tổng quát ghi đè source hash đã khóa của run. Các lời gọi không khai báo phạm vi nguồn giữ cơ chế hash cũ.

## Hợp nhất dự đoán còn lại

Đã kiểm tra công cụ hợp nhất bằng gói Dummy/LR: kết quả tại `outputs/g3_group_predictions_v1_partial/` báo `waiting_for_models`, còn `knn, svm, rf, mlp`. Đây là trạng thái chờ dữ liệu, không phải thiếu triển khai phần chạy của Mai. Không chạy thay các mô hình của thành viên khác.

Khi Nam/Như gửi thư mục kết quả đầy đủ, Mai dùng:

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/merge_g3_predictions.py --runs outputs/g3_baselines_physiological68_v1 outputs/g3_knn_svm outputs/g3_rf_mlp --output outputs/g3_group_predictions_complete
```

Công cụ kiểm tra cùng checksum bảng, feature/schema, quy tắc mặc định, code G3, Python patch/thư viện, khóa/nhãn dự đoán, checksum fold, đúng 14 train/1 test và metric tính lại; từ chối quick, fold thiếu, khóa trùng và nguồn khác chưa xác minh. Các build Python khác nhau được lưu provenance, không giả cùng chuỗi build. Chỉ báo `complete` khi nhận đủ sáu mô hình, hai bài toán và 15 subject mỗi tổ hợp. Không nhận kết quả tuning hay raw112 vào bảng này.

Phần Dummy/LR đã push trong `a5e9099`. Các cập nhật hợp nhất cuối mới chưa commit/push; thay đổi G1 và tài liệu dọn cũ không thuộc lượt này.
