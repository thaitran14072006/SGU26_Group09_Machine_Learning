# WESAD – Chia cửa sổ và trích xuất đặc trưng

**Phần Mai đang thực hiện G1 theo phân công:** xem [bàn giao Mai G1](docs/BAN_GIAO.md), [hợp đồng feature G1](docs/G1_HOP_DONG_FEATURE.md) và [schema nháp](configs/feature_schema.json). Schema cần thống nhất tiền xử lý với Như trước khi khóa G2.

Chương trình đọc dữ liệu WESAD lần lượt theo từng người, chia tín hiệu ngực thành các cửa sổ 60 giây với bước 60 giây, tính đặc trưng thống kê và lưu kết quả trong `data/features/`.

## 1. Phạm vi xử lý

- Thiết bị: cảm biến ngực (`chest`).
- Tín hiệu: ECG, EDA và Resp.
- Tần số lấy mẫu: 700 Hz.
- Cửa sổ: 60 giây, tương ứng 42.000 điểm đo.
- Bước trượt: 60 giây, không chồng lấp.
- Nhãn giữ lại:
  - `1`: Baseline – bình thường.
  - `2`: Stress – căng thẳng.
  - `3`: Amusement – vui thích.

Chỉ giữ cửa sổ có toàn bộ nhãn giống nhau và thuộc ba lớp trên. Bỏ cửa sổ chứa NaN/Inf trong các tín hiệu được chọn và phần cuối không đủ 60 giây.

Các cửa sổ được cắt trên trục thời gian gốc, không nối các đoạn rời nhau sau khi loại nhãn.

> Phiên bản hiện tại trích xuất thống kê từ tín hiệu thô, chưa lọc nhiễu sinh lý hoặc trích xuất HRV, SCR và nhịp thở.

## 2. Cấu trúc thư mục

```text
wesad-stress-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── features_chest.yaml
├── scripts/
│   └── extract_features.py
├── src/
│   └── wesad/
│       ├── __init__.py
│       └── preprocessing.py
└── data/
    ├── raw/
    │   └── WESAD/
    │       ├── S2/
    │       │   └── S2.pkl
    │       ├── S3/
    │       │   └── S3.pkl
    │       └── ...
    └── features/
```

Dữ liệu gốc có thể đặt ngoài dự án. Khi đó, khai báo đường dẫn tuyệt đối trong cấu hình, không cần sao chép dữ liệu.

## 3. Cài đặt thư viện

Mở terminal tại thư mục `wesad-stress-detection`, chạy:

```powershell
python -m pip install -r requirements.txt
```

Các thư viện cần thiết:

```text
numpy
pandas
PyYAML
```

## 4. Cấu hình

File `configs/features_chest.yaml`:

```yaml
# Thư mục chứa trực tiếp S2/, S3/, ...
data_root: "data/raw/WESAD"

device: chest
sensors: [ECG, EDA, Resp]
sampling_rate_hz: 700
labels: [1, 2, 3]

window_seconds: 60
step_seconds: 60
label_rule: homogeneous

# null: xử lý tất cả người tìm thấy trong data_root
subjects: null

feature_version: raw_stats_v1
output_dir: "data/features/chest_60s_step60_raw_stats_v1"
```

Nếu dữ liệu nằm trên Windows ở vị trí khác:

```yaml
data_root: "/WESAD/WESAD"
```

Dùng dấu `/` trong đường dẫn. Thư mục được chỉ định phải chứa trực tiếp các thư mục người tham gia.

Đường dẫn tương đối trong cấu hình được tính từ thư mục gốc `wesad-stress-detection`.

### Chạy thử một người

Thay hai dòng:

```yaml
subjects: [S2]
output_dir: "data/features/S2_chest_60s_step60_raw_stats_v1/S2"
```

### Chạy toàn bộ dữ liệu

Dùng:

```yaml
subjects: null
output_dir: "data/features/chest_60s_step60_raw_stats_v1"
```

Chương trình tự tìm các file `S*/S*.pkl`. Cần kiểm tra dữ liệu trên máy có đủ người tham gia; chương trình không tự tải các file còn thiếu.

## 5. Chạy chương trình

Tại thư mục `wesad-stress-detection`, chạy:

```powershell
python scripts/extract_features.py --config configs/features_chest.yaml
```

Không cần chạy riêng `src/wesad/preprocessing.py`; script tự import và gọi hàm chia cửa sổ.

Nếu file cấu hình vẫn đang mang tên `fearture_chest.yaml`, đổi tên thành `features_chest.yaml` hoặc truyền đúng tên file sau `--config`.

Quy trình:

1. Đọc một file `.pkl`.
2. Lấy ECG, EDA, Resp và nhãn.
3. Chia cửa sổ và kiểm tra điều kiện giữ lại.
4. Tính đặc trưng cho từng cửa sổ.
5. Lưu CSV của người đó.
6. Giải phóng dữ liệu trước khi xử lý người tiếp theo.
7. Ghép các bảng đặc trưng thành `all_subjects.csv`.

Chỉ mở file pickle từ nguồn tin cậy.

## 6. Kết quả đầu ra

```text
data/features/chest_60s_step60_raw_stats_v1/
├── config.yaml
├── S2.csv
├── S3.csv
├── ...
├── all_subjects.csv
├── quality_report.csv
└── run_status.json
```

| File | Nội dung |
| --- | --- |
| `S2.csv`, `S3.csv`… | Đặc trưng từng người |
| `all_subjects.csv` | Ghép đặc trưng của những người trong lượt chạy |
| `config.yaml` | Cấu hình thực tế và hash code |
| `quality_report.csv` | Số cửa sổ giữ lại theo người/lớp và thời gian chạy |
| `run_status.json` | Trạng thái lượt chạy: `running` hoặc `complete` |

Chương trình không lưu mỗi cửa sổ thành một file tín hiệu riêng. Mỗi cửa sổ hợp lệ trở thành một dòng trong CSV.

### Các cột dữ liệu

Có 6 cột quản lý:

```text
subject_id
window_id
start_s
end_s
label_original
quality_flag
```

Mỗi tín hiệu có 7 đặc trưng:

```text
mean, std, min, max, rms, iqr, range
```

Ba tín hiệu tạo tổng cộng 21 đặc trưng, ví dụ:

```text
ecg_mean, ecg_std, ...
eda_mean, eda_std, ...
resp_mean, resp_std, ...
```

Tổng cộng bảng có 27 cột với cấu hình mặc định.

Khi huấn luyện:

- `X`: 21 cột đặc trưng.
- `y`: `label_original`.
- `subject_id`: dùng chia tập theo người.
- Không đưa các cột quản lý vào `X`.

## 7. Kiểm tra sau khi chạy

- `run_status.json` phải có `"status": "complete"`.
- Có CSV của tất cả người đã chọn.
- Nhãn chỉ thuộc `{1, 2, 3}`.
- Mỗi dòng có `end_s - start_s = 60`.
- Các đặc trưng không chứa NaN/Inf.
- Kiểm tra số cửa sổ theo người và lớp trong `quality_report.csv`.

Mốc thời gian có thể có khoảng trống vì chương trình bỏ các cửa sổ không hợp lệ.

`quality_flag` hiện chỉ thể hiện việc kiểm tra giá trị hữu hạn, chưa bảo đảm tín hiệu đạt chất lượng sinh lý. Tín hiệu phẳng vẫn cần kiểm tra bổ sung.

## 8. Chạy lại và thay đổi cấu hình

- Chạy lại cùng cấu hình sẽ xử lý lại và ghi đè kết quả cùng tên.
- Chương trình chưa tự bỏ qua người đã xử lý.
- Khi đổi cảm biến, cửa sổ, bước trượt hoặc code, chọn `output_dir` mới.
- Không sử dụng kết quả của lượt chạy bị ngắt như một lượt hoàn chỉnh.

Ví dụ khi đổi bước trượt thành 5 giây:

```yaml
step_seconds: 5
output_dir: "data/features/chest_60s_step5_raw_stats_v1"
```

## 9. Công việc tiếp theo

Các CSV hiện tại dùng làm baseline từ thống kê tín hiệu thô.

Để lọc nhiễu hoặc phát hiện đỉnh tim, cần xử lý tín hiệu từ `.pkl` rồi trích xuất lại đặc trưng. Không thể khôi phục toàn bộ tín hiệu từ các cột thống kê CSV.

Khi huấn luyện mô hình:

- Tách train/test theo người tham gia.
- Chỉ fit imputer, scaler và bước chọn đặc trưng trên tập train của từng fold.
- Giữ cùng cách chia dữ liệu khi so sánh các mô hình.

## 10. Không đưa dữ liệu lớn lên Git

Thêm vào `.gitignore`:

```gitignore
/data/raw/
/data/features/
/outputs/

.venv/
__pycache__/
.ipynb_checkpoints/
.env
```

Mỗi thành viên tải WESAD và cấu hình `data_root` trên máy của mình.

Nguồn dữ liệu:
https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

## 11. Bổ sung hạ tầng và hướng dẫn mới ngày 02 tháng 11 năm 2026

Các mục 1–10 giữ nội dung README ban đầu. Bảng dưới ghi các thay đổi của phiên bản mới:

| Nội dung trong bản gốc | Cập nhật trong mã mới |
| --- | --- |
| Chỉ hỗ trợ homogeneous, bước 60 giây | Config cũ vẫn giữ nguyên; config features_smoke.yaml thêm majority 90%, bước 5 giây |
| Bảng có 6 metadata và 21 feature, tổng 27 cột | Extractor mới thêm label_binary và label_fraction: 8 metadata, 21 feature, tổng 29 cột |
| Lưu config.yaml | Extractor mới lưu config.json, schema, checksum và manifest từng subject |
| Chưa tự bỏ qua subject đã chạy | Thêm --resume để kiểm tra mã băm và bỏ qua kết quả còn hợp lệ |
| Chưa có bộ chạy huấn luyện | Thêm run_experiment.py cho LOSO và nested CV theo người |
| Thư viện numpy, pandas, PyYAML | Bổ sung thư viện đánh giá trong requirements.txt và phiên bản đã kiểm tra trong requirements.lock.txt |

Tệp dữ liệu cũ vẫn được mô tả ở các mục gốc; không trộn chúng với đầu ra phiên bản mới. Các ví dụ đường dẫn trong mục 4 là minh họa, cần thay bằng đường dẫn raw thật trên máy; cấu hình riêng của mã mới nằm trong local.yaml.

Hạ tầng dùng chung để đọc từng subject, trích feature tuần tự, chạy LOSO và nested CV theo người. Python từ 3.10; môi trường đã kiểm tra trên Windows dùng Python 3.12.14.

**Bàn giao hôm nay:** [Công việc Thái đã hoàn thành và đầu việc nối tiếp](docs/BAN_GIAO_CONG_VIEC_THAI_2026-11-02.md).

**Phần G1 của Như:** [thiết kế tiền xử lý](docs/G1_NHU_THIET_KE_TIEN_XU_LY.md) và [kiểm tra mẫu S2](docs/G1_NHU_KIEM_TRA_S2_2026-11-03.md). Phần bàn giao này chỉ gồm thiết kế, môi trường và chạy mẫu bằng hạ tầng có sẵn; chưa triển khai G2.

### Phạm vi bổ sung

- Đọc và xác minh một PKL mỗi lần; cấu hình riêng theo máy, CLI chọn subject.
- Giữ tương thích quy tắc nhãn đồng nhất cũ; thêm nhãn đa số ít nhất 90% theo proposal.
- Extractor đi kèm chỉ tính 21 thống kê raw ECG/EDA/Resp. Chưa phải HRV, SCR, tần số thở hoặc feature cổ tay cuối cùng. Mai/Như nối thuật toán qua `subject_processor`.
- Ghi CSV nguyên tử, resume kiểm tra hash raw/CSV, code, config, thư viện và schema.
- LOSO, nested GroupKFold 3 lượt, checkpoint từng outer fold, hai task; sáu mô hình có pipeline khởi đầu và chỗ thay thế của từng thành viên.
- Scaler/imputer, selection tùy chọn và resampling nằm trong pipeline của từng fold. Bảng chính dùng chuẩn hóa bằng train.
- Chỉ báo `complete` khi có đủ 15 người và tất cả fold của các mô hình/task yêu cầu; các chạy thử được ghi `partial`.

## 12. Cài đặt

Mở terminal tại thư mục chứa README này, không phải Git root phía trên:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

`requirements.txt` là danh sách phụ thuộc; `requirements.lock.txt` là phiên bản đã dùng kiểm tra. `.venv` hiện có trên máy Thái, không cần tạo lại. Máy khác tạo môi trường riêng, không sao chép `.venv`.

```powershell
Copy-Item configs/local.example.yaml configs/local.yaml
```

Sửa `data_root` trong `local.yaml` trỏ tới nơi chứa trực tiếp S2, S3… Trên máy Thái là `C:/Users/Home/Desktop/final_ML/WESAD/WESAD`. Đường dẫn tương đối của các lệnh/config được tính từ thư mục dự án này. File local chỉ được chứa `data_root`, không thay phương pháp thực nghiệm.

## 13. Chạy nhanh và trích lô raw của Thái

Kiểm tra S2 trước:

```powershell
.\.venv\Scripts\python.exe scripts/extract_features.py `
  --config configs/features_smoke.yaml --local configs/local.yaml `
  --subjects S2 --resume
```

Trích lô Thái bằng cùng mã:

```powershell
.\.venv\Scripts\python.exe scripts/extract_features.py `
  --config configs/features_smoke.yaml --local configs/local.yaml `
  --subjects S2 S7 S13 S15 --resume
```

`features_smoke.yaml` dùng 60 giây, bước 5 giây, majority 90%. Config cũ `features_chest.yaml` vẫn dùng homogeneous, bước 60 giây. Không trộn hai thư mục kết quả; không áp mốc số cửa sổ homogeneous cho majority.

Đầu ra trong `data/features/chest_raw_stats_majority_v1/`:

- `<subject>.csv` và `<subject>.json`: bảng và manifest từng người.
- `all_subjects.csv`: chỉ hợp nhất các subject được chọn trong lệnh vừa chạy.
- `config.json`, `feature_schema.json`, `checksums.json`, `quality_report.csv`, `run_status.json`, `run.log`.

Mỗi CSV có 8 metadata và 21 feature. Metadata: subject_id, window_id, start_s, end_s, label_original, label_binary, label_fraction, quality_flag. Không đưa metadata vào X. NaN feature có thể được impute trong train; Inf và metadata thiếu bị từ chối.

Resume vẫn đọc raw để xác minh SHA256, nhưng không nạp PKL/trích lại subject đã hoàn chỉnh. Đổi mã, định nghĩa feature hoặc thư viện phải dùng output_dir phiên bản mới. Chạy lại với danh sách subject ngắn hơn sẽ tạo lại bảng tổng của danh sách đó; muốn giữ gói bốn người, dùng đủ bốn subject trong lệnh cuối.

## 14. Huấn luyện từ feature

### Kiểm tra một outer fold trên gói hiện có

Gói Thái chỉ gồm bốn người nên phải khai báo chạy thử:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py `
  --config configs/experiment.yaml --models dummy lr `
  --outer-subjects S2 --allow-partial `
  --output outputs/experiments/quick_dummy_lr
```

Kết quả này là kiểm tra tích hợp, không dùng làm điểm đồ án. Sau đó dùng `--resume` với cùng lệnh để tiếp tục mà không fit lại fold hợp lệ.

### Đánh giá đủ 15 người

Khi Mai đã bàn giao feature chung và schema, dùng một config có sensor_set/feature_version đúng. Với đường dẫn ví dụ của gói cuối, lệnh là:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py `
  --config configs/experiment.yaml `
  --features data/features/FEATURE_VERSION/all_subjects.csv `
  --schema data/features/FEATURE_VERSION/feature_schema.json `
  --output outputs/experiments/FEATURE_VERSION_defaults --resume
```

Thay `FEATURE_VERSION` bằng thư mục thật. Không dùng `--allow-partial` cho lượt chính. `--task multiclass`, `--task binary` hoặc `--task both` chọn bài toán; mặc định config hiện tại chạy cả hai. Binary ánh xạ Stress 2 → 1; Baseline 1 và Amusement 3 → 0.

### Nested tuning

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py `
  --config configs/experiment.yaml --models knn svm rf mlp --nested `
  --features data/features/FEATURE_VERSION/all_subjects.csv `
  --output outputs/experiments/FEATURE_VERSION_nested --resume
```

Mỗi outer fold có một test subject; chỉ các subject train vào inner GroupKFold. Lưới bốn ứng viên mỗi mô hình trong YAML là cấu hình khởi đầu để các chủ mô hình chốt trước chạy chính. Tham số dùng tên bước `model__...`. Không sửa lưới theo outer test.

Đặt `balance: true` trong bản config thực nghiệm riêng khi so sánh xử lý mất cân bằng: LR/SVM/RF dùng class_weight; KNN/MLP dùng RandomOverSampler chỉ trong fit. Lượt mặc định giữ `balance: false`. `variance_threshold` tùy chọn chạy selection trong fold. Chọn theo tương quan/RF và nhánh chuẩn hóa theo subject chờ tích hợp; runner hiện từ chối tên normalization khác global_train để tránh dùng giả định khác mà không công bố.

Outer folds có thể chia qua `--outer-subjects`; mỗi máy vẫn cần đủ bảng 15 người. Không ghép tùy tiện dự đoán từ mã/config/hash khác nhau.

## 15. Đầu ra thực nghiệm

Mỗi output_dir riêng có `config.json`, `environment.json`, `run.log` và checkpoint trong `folds/<task>/<model>/<subject>/`. Mỗi fold có dự đoán, metric, chi tiết splits/warnings/best_params, bảng inner CV nếu tuning và manifest hash.

Các file tổng: `predictions.csv`, `metrics_by_subject.csv`, `summary.json`, `splits.json`, `best_params.json`, `run_status.json`. Macro-F1 trung bình theo người; std mẫu ddof=1, null khi chỉ có một người. Class report và confusion matrix có thứ tự nhãn cố định. Một run `partial` không được dùng như kết quả đủ 15 người.

## 16. Giao diện để các thành viên nối tiếp

### Mai và Như

Thêm hàm trong package, ví dụ `wesad.features:process_subject`, rồi đặt:

```yaml
subject_processor: wesad.features:process_subject
```

Hàm có chữ ký `process_subject(path: Path, config: dict) -> pandas.DataFrame`; trả metadata bắt buộc và các feature số. Hạ tầng tự xác minh, ghi manifest/hash, resume và ghép bảng. Đặt feature_version/output_dir mới. Majority hiện tính tỷ lệ nhãn trên toàn cửa sổ gốc, kể cả nhãn không dùng trong mẫu số; không nối các đoạn đã bỏ nhãn.

### Nam, Như và Mai

`build_pipeline(name, config)` trong `models.py` trả pipeline chưa fit. Có thể thay bằng factory của thành viên trong YAML:

```yaml
model_factories:
  svm: wesad.models_neighbors_svm:build_pipeline
```

Factory nhận `name`, `config` và trả estimator/Pipeline có thể clone. Lưới hiện tại dùng tên bước `model`; resampling dùng Pipeline của imbalanced-learn.

## 17. Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm tra bao gồm threshold nhãn, schema/metadata, resume, đầy đủ outer predictions, tách nhóm outer/inner, scaler chỉ fit train, bốn mô hình nested và sampler không thay số dự đoán test. Dữ liệu giả của test nằm trong thư mục tạm tự xóa.

## 18. Git và bàn giao

Git root là `DOAN/SGU26_Group09_Machine_Learning`, chứa cả weeks/ và Challenge/. Không git init trong dự án con. Add từng đường dẫn cần thiết, không commit raw, feature, outputs, local config hoặc môi trường. File mẫu `local.example.yaml` vẫn được quản lý bằng Git.

Các kết quả thật trên bốn người chỉ là bằng chứng chạy hạ tầng; xem file bàn giao để biết phần nào đã làm và phần nào còn chờ nhóm. Dữ liệu và output cần chia sẻ riêng, không nằm trong repository.
