# WESAD – Chia cửa sổ và trích xuất đặc trưng

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