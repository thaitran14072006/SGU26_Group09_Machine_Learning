# WESAD — Chia cửa sổ và trích xuất đặc trưng ngực, cổ tay

## 1. Phạm vi hiện tại

Chương trình đọc dữ liệu WESAD theo từng người, chia các tín hiệu ngực và
cổ tay thành cửa sổ cùng khoảng thời gian, trích xuất đặc trưng rồi lưu
thành bảng CSV.

Cấu hình chính:

- Cửa sổ: 60 giây.
- Bước trượt: 5 giây.
- Nhãn: majority, tỷ lệ nhãn đa số ít nhất 90%.
- Đối tượng: S2–S11 và S13–S17, tổng cộng 15 người.
- Đầu ra: 112 đặc trưng thống kê raw và 8 cột metadata.

Các cảm biến được xử lý:

| Thiết bị | Tín hiệu | Tần số lấy mẫu gốc |
| --- | --- | --- |
| Ngực | ECG, EDA, EMG, Resp, Temp, ACC | 700 Hz |
| Cổ tay | BVP | 64 Hz |
| Cổ tay | EDA, TEMP | 4 Hz |
| Cổ tay | ACC | 32 Hz |

Các kênh được cắt theo cùng khoảng thời gian nhưng dùng tần số lấy mẫu
riêng. Không nội suy toàn bộ tín hiệu lên 700 Hz.

### Giới hạn

Bộ hiện tại tính thống kê trực tiếp trên tín hiệu thô. Chưa triển khai
lọc nhiễu sinh lý, HRV, SCR, chuyển đổi đơn vị hoặc cách tính ACC/EMG
trên 12 đoạn 5 giây rồi gộp theo thiết kế đề tài.

Không coi bộ 112 thống kê raw là bộ 68 đặc trưng sinh lý trong schema nháp.

## 2. Các file chính

| File | Chức năng |
| --- | --- |
| `scripts/extract_features.py` | Chạy trích xuất theo config, subject và thư mục đầu ra |
| `configs/features_smoke.yaml` | Cấu hình toàn bộ cảm biến ngực và cổ tay |
| `src/wesad/multisensor.py` | Cắt cửa sổ theo tần số gốc, kiểm tra và tính đặc trưng |
| `src/wesad/extraction.py` | Điều phối từng người, lưu CSV, ghép bảng, ghi log và resume |
| `src/wesad/preprocessing.py` | Tạo lưới cửa sổ và xác định nhãn |
| `src/wesad/data.py` | Đọc PKL/CSV và kiểm tra cấu trúc dữ liệu |
| `src/wesad/utils.py` | Đọc config, xử lý đường dẫn, ghi file và tính hash |
| `scripts/run_experiment.py` | Chạy mô hình và đánh giá LOSO |
| `configs/experiment.yaml` | Cấu hình huấn luyện trên bảng feature |
| `tests/test_multisensor.py` | Kiểm thử xử lý nhiều cảm biến bằng dữ liệu giả lập |

## 3. Chuẩn bị môi trường

Các lệnh dưới đây chạy trong PowerShell tại thư mục
`wesad-stress-detection`, nơi chứa `scripts/`, `configs/` và `src/`.

Nếu chưa có môi trường:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
python -m pip install -e .
```

Nếu đã có môi trường của nhóm, kích hoạt đúng môi trường đó thay vì tạo lại.

Kiểm tra Python đang sử dụng:

```powershell
python -c "import sys; print(sys.executable)"
```

Dùng phiên bản Python thống nhất theo `.python-version`.

Để chạy notebook EDA, cài thêm nếu chưa có:

```powershell
python -m pip install matplotlib ipykernel
```

## 4. Chuẩn bị dữ liệu và đường dẫn

Thư mục dữ liệu phải chứa trực tiếp các thư mục subject:

- `S2/S2.pkl`
- `S3/S3.pkl`
- ...
- `S11/S11.pkl`
- `S13/S13.pkl`
- ...
- `S17/S17.pkl`

Chỉ đọc file PKL từ nguồn WESAD tin cậy.

Nếu chưa có `configs/local.yaml`, sao chép file mẫu:

```powershell
Copy-Item configs/local.example.yaml configs/local.yaml
```

Sửa `configs/local.yaml`, ví dụ:

```yaml
data_root: "D:/Datasets/WESAD/WESAD"
```

Nếu đường dẫn file là `D:/Datasets/WESAD/WESAD/S2/S2.pkl`,
thì `data_root` phải là `D:/Datasets/WESAD/WESAD`.

`local.yaml` chỉ chứa `data_root`. Không commit đường dẫn riêng
của máy hoặc dữ liệu thô lên Git.

## 5. Cấu hình trích xuất toàn bộ cảm biến

File: `configs/features_all_sensors.yaml`

```yaml
data_root: data/raw/WESAD/WESAD
device: combined

# Tần số lưới nhãn, không phải tần số chung của mọi cảm biến.
sampling_rate_hz: 700

sensors:
  chest: [ECG, EDA, EMG, Resp, Temp, ACC]
  wrist: [BVP, EDA, TEMP, ACC]

subject_processor: wesad.multisensor:process_subject

labels: [1, 2, 3]
window_seconds: 60
step_seconds: 5
label_rule: majority
min_label_fraction: 0.90

feature_version: all_sensors_raw_stats_v1
subjects: [S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17]
output_dir: data/features/all_sensors_raw_stats_v1
```

Danh sách subject được khai báo tường minh để phát hiện khi thiếu PKL.

Các tham số dòng lệnh:

- `--config`: chọn cấu hình trích xuất.
- `--local`: ghi đè đường dẫn `data_root`.
- `--subjects`: ghi đè danh sách người cần chạy.
- `--output-dir`: ghi đè thư mục lưu kết quả.
- `--resume`: kiểm tra và dùng lại kết quả theo người còn hợp lệ.

## 6. Cách chia cửa sổ và gán nhãn

Các cửa sổ lần lượt là:

- `[0, 60)` giây.
- `[5, 65)` giây.
- `[10, 70)` giây.
- ...

Hai cửa sổ liên tiếp chồng lấn 55 giây.

Với mỗi cửa sổ:

1. Tính tỷ lệ nhãn trên toàn bộ cửa sổ ở lưới 700 Hz.
2. Chỉ nhận nếu nhãn đa số thuộc 1/2/3 và chiếm ít nhất 90%.
3. Cắt đoạn cùng khoảng thời gian của mọi cảm biến.
4. Loại cửa sổ nếu một kênh được yêu cầu thiếu mẫu hoặc có NaN/Inf.
5. Tính đặc trưng và lưu thành một dòng.

Quy ước nhãn:

| Nhãn | Trạng thái |
| --- | --- |
| 1 | Baseline |
| 2 | Stress |
| 3 | Amusement |

Ví dụ, cửa sổ gồm 54 giây Stress và 6 giây Baseline được giữ
và gán Stress vì Stress chiếm 90%.

Nhãn không dùng vẫn nằm trong mẫu số khi tính tỷ lệ. Không xóa nhãn
rồi nối các đoạn tín hiệu rời nhau. Phần cuối bản ghi không đủ
60 giây không tạo cửa sổ.

Thiếu hẳn cảm biến hoặc sai số trục khiến chương trình báo lỗi;
chương trình không âm thầm bỏ qua cảm biến đó.

## 7. Chạy thử S2

Chạy thử một người trước và lưu vào thư mục riêng:

```powershell
python scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S2 --output-dir data/features/all_sensors_raw_stats_S2 --resume
```

Kiểm tra:

- Có `S2.csv`, `S2.json`, `all_subjects.csv` và schema đi kèm.
- Bảng có 112 feature và 8 metadata, tổng cộng 120 cột.
- Độ dài cửa sổ là 60 giây.
- `label_fraction` nằm trong khoảng từ 0.90 đến 1.
- Có các cột mang tiền tố `chest_` và `wrist_`.
- Xem số cửa sổ giữ/loại trong `quality_report.csv`.

Không dùng `scripts/check_g1_sample.py` cho gói này.
Script đó chỉ kiểm tra gói S2 gồm 21 thống kê raw ngực cũ.

## 8. Chạy đủ 15 người

Sau khi kiểm tra S2, chạy:

```powershell
python scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --resume
```

Chương trình xử lý từng PKL tuần tự, lưu bảng từng người rồi ghép thành:

`data/features/all_sensors_raw_stats_v1/all_subjects.csv`

Không cần tự tạo 15 bộ train/test. LOSO được thực hiện sau,
dựa vào cột `subject_id`.

### Chỉ chạy lô của Nam

```powershell
python scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S3 S8 S14 S16 --output-dir data/features/nam_all_sensors_raw_stats_v1 --resume
```

### Lưu ý khi chạy theo lô

`all_subjects.csv` chỉ chứa những subject được chọn trong lần chạy đó.
Chương trình không tự cộng dồn tất cả CSV có sẵn trong thư mục.

Nếu chạy riêng từng người liên tiếp, bảng tổng hợp sẽ được tạo lại
theo danh sách của lần chạy cuối.

Để ghép đủ bằng runner hiện tại trên một máy, cần đủ 15 PKL và chạy
cùng config/output với danh sách đủ 15 người, sử dụng `--resume`.

Nếu trích xuất trên nhiều máy, cần kiểm tra và hợp nhất các gói riêng.
Hiện chưa có CLI hợp nhất gói nhiều máy.

Không chạy nhiều tiến trình cùng ghi vào một `output_dir`.
Không ghép cả CSV từng người lẫn bảng tổng hợp của cùng lô,
vì sẽ gây trùng dữ liệu.

## 9. Đặc trưng đầu ra

Mỗi tín hiệu đơn có 7 thống kê:

- Mean.
- Standard deviation, ddof=0.
- Min.
- Max.
- RMS.
- IQR.
- Range.

ACC được tính riêng cho x/y/z và magnitude. Giữ nguyên scale
và thành phần trọng lực ghi trong PKL.

| Nhóm | Số đặc trưng |
| --- | --- |
| Ngực | 63 |
| Cổ tay | 49 |
| Kết hợp | 112 |

Ví dụ tên cột:

- `chest_ecg_mean`
- `chest_emg_rms`
- `chest_acc_x_std`
- `chest_acc_magnitude_mean`
- `wrist_bvp_mean`
- `wrist_eda_std`
- `wrist_temp_mean`
- `wrist_acc_z_range`

Tám cột metadata:

| Cột | Ý nghĩa |
| --- | --- |
| `subject_id` | Người tham gia |
| `window_id` | Mã người và chỉ số bắt đầu trên lưới nhãn 700 Hz |
| `start_s` | Thời điểm bắt đầu, tính bằng giây |
| `end_s` | Thời điểm kết thúc, không bao gồm điểm cuối |
| `label_original` | Nhãn 1/2/3 |
| `label_binary` | Stress = 1; Baseline/Amusement = 0 |
| `label_fraction` | Tỷ lệ nhãn đa số |
| `quality_flag` | Phạm vi kiểm tra chất lượng |

Không đưa metadata vào X khi huấn luyện.

## 10. Các file được sinh ra

| File | Nội dung |
| --- | --- |
| `S2.csv`, `S3.csv`, ... | Bảng đặc trưng theo người |
| `S2.json`, `S3.json`, ... | Hash nguồn/CSV, số cửa sổ, thời gian và QA |
| `all_subjects.csv` | Bảng ghép của các subject trong lần chạy |
| `feature_schema.json` | Tên/thứ tự feature và nhóm ngực/cổ tay/kết hợp |
| `quality_report.csv` | Số cửa sổ, phân bố nhãn và lý do loại |
| `config.json` | Cấu hình xử lý, mã và môi trường |
| `checksums.json` | Hash các bảng và schema |
| `run_status.json` | Trạng thái và phạm vi lần chạy |
| `run.log` | Nhật ký xử lý |

Các số đếm QA:

- `n_candidate_windows`: số cửa sổ đầy đủ theo lưới thời gian.
- `n_label_valid_windows`: số cửa sổ đạt quy tắc nhãn.
- `n_rejected_label`: số cửa sổ bị loại do nhãn.
- `n_rejected_incomplete_signal`: số cửa sổ bị loại do thiếu mẫu tín hiệu.
- `n_rejected_nonfinite_signal`: số cửa sổ bị loại do NaN/Inf.
- `n_kept_windows`: số cửa sổ cuối cùng được giữ.

Nếu một cửa sổ có nhiều lỗi tín hiệu, chỉ đếm theo lý do đầu tiên
được phát hiện. Không cộng trùng một cửa sổ vào nhiều lý do loại.

`quality_flag = finite_only_not_physiological_QA` không chứng minh
tín hiệu đã đạt chất lượng sinh lý hoặc hoàn toàn không có nhiễu.

`status: complete` chỉ áp dụng cho danh sách subject được chọn.
Kiểm tra đủ 15 người trước khi kết luận hoàn tất bộ dữ liệu chung.

## 11. Chạy tiếp và quản lý phiên bản

`--resume` kiểm tra cấu hình, mã, môi trường, hash PKL,
hash CSV và manifest trước khi dùng lại kết quả.

Nếu chương trình báo:

```text
Output contains different processing/code/environment. Choose a new output_dir.
```

hãy dùng thư mục đầu ra mới:

```powershell
python scripts/extract_features.py --config configs/features_all_sensors.yaml --local configs/local.yaml --output-dir data/features/all_sensors_raw_stats_v2 --resume
```

Không sửa chữ ký hoặc manifest để ép dùng lại kết quả cũ.

Không trộn:

- Bộ 21 feature ngực và bộ 112 feature toàn bộ cảm biến.
- Các phiên bản mã/cấu hình trích xuất khác nhau.
- Bảng cửa sổ 60/60 và bảng cửa sổ 60/5.

## 12. EDA và chọn nhóm cảm biến

Sau khi có bảng feature:

1. Kiểm tra đủ subject và khóa cửa sổ không trùng.
2. Thống kê phân bố nhãn, NaN/Inf và điểm bất thường.
3. Vẽ phân bố đặc trưng, tương quan và khác biệt theo người.
4. Viết nhận xét bằng số liệu thực tế.
5. Chuyển sang huấn luyện và đánh giá LOSO.

Dùng `feature_schema.json` sinh cùng thư mục CSV.
Không dùng `configs/feature_schema.json` nháp 68 feature cho bảng này.

Schema đầu ra có:

- `feature_columns`: toàn bộ 112 feature.
- `sensor_groups.chest`: 63 feature ngực.
- `sensor_groups.wrist`: 49 feature cổ tay.
- `sensor_groups.combined`: toàn bộ feature.

Để so sánh nhóm cảm biến công bằng, chọn các nhóm cột trên cùng
bảng combined và giữ cùng khóa cửa sổ/folds.

Notebook EDA cũ chọn cố định 21 feature chưa tương thích bảng mới.
Cần sửa đường dẫn, cách đọc feature từ schema, tên cột minh họa
và lưới subplot cho bộ 112 feature.

## 13. Cấu hình và chạy mô hình

Dùng một file `configs/experiment.yaml` cho bộ dữ liệu hiện tại.

Cập nhật ba trường sau trong file; giữ các tham số mô hình,
search_spaces và thiết kế thực nghiệm còn lại của nhóm:

```yaml
features: data/features/all_sensors_raw_stats_v1/all_subjects.csv
output_dir: outputs/experiments/all_sensors_raw_stats_defaults
sensor_set: combined_raw_statistics
```

Các lệnh dưới đây không phụ thuộc `experiment_all_sensors.yaml`.

Chạy KNN/SVM mặc định cho cả ba lớp và nhị phân:

```powershell
python scripts/run_experiment.py --config configs/experiment.yaml --models knn svm --task both --resume
```

Đảm bảo `nested: false` khi chạy cấu hình mặc định.

Chạy cả sáu mô hình:

```powershell
python scripts/run_experiment.py --config configs/experiment.yaml --models dummy lr knn svm rf mlp --task both --resume
```

Thử một outer fold trước để đo thời gian:

```powershell
python scripts/run_experiment.py --config configs/experiment.yaml --models knn svm --task both --outer-subjects S2 --output outputs/experiments/all_sensors_quick --resume
```

Lượt này vẫn cần bảng đủ 15 người; chỉ chạy outer test S2.
Đây là kiểm tra nhanh, không phải kết quả LOSO đủ 15 lượt.

Bật tuning KNN/SVM, dùng thư mục riêng:

```powershell
python scripts/run_experiment.py --config configs/experiment.yaml --models knn svm --task both --nested --output outputs/experiments/all_sensors_knn_svm_tuned --resume
```

Tuning dùng search_spaces trong config và inner GroupKFold
trên subject thuộc outer train.

Không chuẩn hóa hoặc chọn đặc trưng trên toàn bảng trước LOSO.
Imputer, scaler, selection và resampling phải học trong train
của đúng fold. Không chia ngẫu nhiên các dòng cửa sổ train/test.

## 14. Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Bản cập nhật đã đạt 33 kiểm thử trên môi trường kiểm tra,
gồm 8 kiểm thử mới cho nhiều cảm biến và 25 kiểm thử có sẵn.

Các kiểm thử dùng dữ liệu giả lập. Kết quả này không thay thế
việc chạy và kiểm tra S2 thật trên môi trường của nhóm.

## 15. Lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
| --- | --- |
| Không tìm thấy `Sx/Sx.pkl` | Kiểm tra `data_root` chứa trực tiếp các thư mục Sx |
| Báo thiếu cảm biến | Kiểm tra PKL gốc; tên kênh phân biệt hoa/thường |
| Báo ACC sai shape | ACC phải có ba trục, không làm phẳng thành một kênh |
| Không còn cửa sổ hợp lệ | Xem thống kê QA trong thông báo lỗi |
| Output khác chữ ký | Dùng output mới hoặc đúng phiên bản xử lý ban đầu |
| Bảng tổng hợp chỉ có vài người | Kiểm tra `--subjects` và lần chạy cuối |
| Notebook báo thiếu cột `ecg_mean` | Notebook vẫn dùng tên cột của bộ 21 feature cũ |
| Schema đòi feature sinh lý chưa có | Đang dùng nhầm schema nháp 68 feature |
| Runner báo khác dữ liệu/config/mã | Chọn thư mục kết quả thực nghiệm mới |
| Gói S2 mới không qua check_g1_sample | Script đó chỉ dùng cho bộ 21 raw-stat ngực cũ |