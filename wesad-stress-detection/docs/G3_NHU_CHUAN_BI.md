# Chuẩn bị G3 phần Như — 08/11/2026

Theo phân công: RF và MLP mặc định, seed 42 khi hỗ trợ; chạy ba lớp và nhị phân bằng LOSO; ghi thời gian, cảnh báo hội tụ và trực quan sơ bộ. Lượt này chỉ chuẩn bị module/cấu hình và lệnh quick một fold; chưa chọn bảng đầu vào thay nhóm, chưa train WESAD hoặc làm G4.

## Bản cơ bản đã có

| File | Nội dung |
| --- | --- |
| `src/wesad/models_rf_mlp.py` | Factory RF/MLP riêng cho phần Như, gọi qua cơ chế model_factories của runner Thái |
| `configs/g3_nhu_defaults.yaml` | Chỉ RF/MLP, hai bài toán, seed 42; features=null chờ đầu vào cụ thể |
| `scripts/run_nhu_g3_quick.py` | Yêu cầu đường dẫn bảng/schema và subject test; chỉ chạy một outer subject, lưu quick_check.json |
| `tests/test_models_rf_mlp.py` | 5 test mới về tham số mặc định, xử lý thiếu, train-only preprocessing và LOSO/resume |
| `reports/g3_nhu/preparation_check.json` | Bằng chứng kiểm tra bằng dữ liệu giả, không công bố điểm mô hình giả như kết quả đồ án |

Tận dụng `run_experiment.py`/`evaluation.py` hiện có để lưu dự đoán, metrics, splits, warnings và confusion matrix dạng số. Không viết một bộ LOSO khác hoặc thay đổi code RF/MLP fallback chung của nhóm.

## Cấu hình mặc định

- RF dùng `RandomForestClassifier(random_state=42, n_jobs=1)`; các tham số estimator khác giữ mặc định của thư viện khóa.
- MLP dùng `MLPClassifier(random_state=42)`; giữ hidden_layer_sizes=(100,), max_iter=200 và các mặc định khác, không bật early stopping hoặc tự tăng max_iter.
- Cả hai pipeline có median imputer và StandardScaler, fit trên train của đúng fold. Giữ cột train trống bằng `keep_empty_features=True`; nếu cả cột train thiếu, fallback của imputer là 0 và phải diễn giải giới hạn này khi chọn bộ sinh lý. Không lấy thông tin test để điền thiếu.
- Không dùng selection, class balancing, resampling hoặc nested tuning trong bản G3 mặc định. Factory từ chối các tùy chọn đó và model_params override để tránh nhầm với mặc định.
- X do loader chung chọn từ schema, không lấy subject_id, window_id, thời gian, nhãn hoặc quality_flag làm feature.

## Đã kiểm tra, chưa phải kết quả G3

Toàn checkout: **81/81 test đạt**, có 5 test mới RF/MLP. Đã chạy CLI quick thật trên bảng giả có ba subject S2/S3/S4: train S2+S3, test S4, hai mô hình × hai bài toán = bốn fold hoàn tất. Có 72 dòng dự đoán riêng biệt theo task/model/window, status `partial` và nhãn quick, chưa đủ 15 lượt LOSO.

Fixture chỉ kiểm tra đường đi mã và giao diện; tên subject là mã giả tương thích loader, không phải tín hiệu WESAD của các subject này. Các điểm trong output giả không được dùng làm kết quả đồ án. Fixture/output ở ngoài Git tại `outputs/g3_nhu_synthetic_check_20261108/`; báo cáo nhỏ chỉ ghi kiểm tra, không ghi điểm F1 từ dữ liệu giả.

Test fit MLP mặc định trên một mẫu giả nhỏ có phát sinh cảnh báo chưa hội tụ ở 200 vòng; không sửa max_iter để làm mất cảnh báo. Khi chạy WESAD, xem warnings của từng fold và báo đúng trường hợp không hội tụ.

## Khi nhận bảng dữ liệu nhóm chốt

Chưa quyết định thay nhóm giữa raw112 và sinh lý68. Nhận đúng `all_subjects.csv` cùng schema đi kèm; ghi rõ sensor_set theo gói thực tế. Dùng thư mục output mới cho mỗi lần thay dữ liệu/code/cấu hình.

Ví dụ cú pháp sau khi có bảng raw112 chung trên máy (thay đường dẫn nếu nhóm chọn gói khác):

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_nhu_g3_quick.py --features data/features/g2_raw112_group15_20261107/all_subjects.csv --test-subject S4 --sensor-set combined_raw112 --output outputs/g3_nhu_raw112_quick_lan1
```

Đây là hướng dẫn cho gói sẽ nhận, chưa phải lệnh đã chạy trên dữ liệu thật. Quick cho phép bảng thiếu subject để kiểm tra luồng; nếu dùng bảng lô bốn người thì train chỉ có ba người và vẫn không phải LOSO 14-train/1-test chính thức. Lệnh không tự chốt đầu vào, không chạy toàn bộ test subjects và không có resume; chọn output mới để giữ bằng chứng.

Xem sau khi chạy:

- `quick_check.json`: phạm vi quick, thời gian fit+predict và warnings của bốn fold.
- `folds/<task>/<model>/<subject>/details.json`: cảnh báo, train/test subjects, confusion matrix theo thứ tự [1,2,3] hoặc [0,1].
- `metrics_by_subject.csv`, `predictions.csv`, `splits.json`, `summary.json`: giữ trạng thái partial; không đưa vào bảng kết quả G3 chính thức.

Runner hiện ghi thời gian fit và predict chung, chưa tách riêng. Các định nghĩa metric kế thừa runner: Macro-F1 theo tập nhãn cố định với zero_division=0; Balanced Accuracy dùng các lớp thật có mặt trong test fold. Trường hợp thiếu lớp cần Thái/Mai xác nhận chính sách trước báo cáo chính thức.

## Các đợt tiếp theo

1. Sau khi nhóm chốt dữ liệu: thử một fold trên WESAD thật, kiểm tra feature/NaN, thời gian và hội tụ.
2. Sau khi lượt thử đạt yêu cầu: chạy đủ 15 outer subjects bằng runner chung, lưu dự đoán và điểm hai bài toán. Không tự bỏ `features=null` hoặc thay đầu vào trước khi có gói đã chọn.
3. Từ dự đoán thật: làm confusion matrix và biểu đồ sơ bộ. Bản chuẩn bị này chưa thêm biểu đồ.

Tuning RF/MLP thuộc G4. Giữ bản cơ bản để cập nhật theo từng đợt; không cố tình tạo lỗi hoặc thay đổi tham số mặc định nhằm làm kết quả đẹp hơn. Người dùng tự commit và push.
