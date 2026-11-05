> **Trích xuất toàn bộ ngực + cổ tay:** dùng `configs/features_all_sensors.yaml` (mặc định mới của CLI), xem [hướng dẫn chi tiết](docs/ALL_SENSORS_RAW_STATS.md). Bộ này có 112 thống kê raw, không phải schema nháp 68 feature sinh lý. Config mô hình tương ứng là `configs/experiment_all_sensors.yaml`.
>
> Các hướng dẫn và cấu hình `features_smoke.yaml` dưới đây vẫn dành cho bộ **21 thống kê raw ngực** cũ khi truyền `--config` tường minh. Không trộn hai phiên bản dữ liệu.

# chia cửa sổ và trích xuất dữ liệu WESAD

## 1. Bộ G1 cũ: 21 thống kê ngực

Pipeline đọc dữ liệu PKL theo từng người, chia tín hiệu thành cửa sổ,
trích xuất đặc trưng, lưu CSV theo người và ghép thành all_subjects.csv.

Cấu hình bộ 21 feature trong phần hướng dẫn này:

- Thiết bị: cảm biến ngực.
- Tín hiệu: ECG, EDA, Resp.
- Tần số lấy mẫu: 700 Hz.
- Độ dài cửa sổ: 60 giây.
- Bước trượt: 5 giây.
- Nhãn hợp lệ: 1 Baseline, 2 Stress, 3 Amusement.
- Chấp nhận cửa sổ nếu nhãn chiếm đa số đạt ít nhất 90%.
- Mỗi tín hiệu có 7 thống kê: mean, std, min, max, RMS, IQR, range.
- Tổng cộng 21 đặc trưng và 8 cột metadata.

Đây là thống kê tín hiệu thô; chưa có lọc nhiễu sinh lý,
HRV, SCR, xử lý cổ tay hoặc ACC đa trục trong processor G1 cũ.

Pipeline không lưu mỗi cửa sổ thành một file tín hiệu thô riêng.
Mỗi cửa sổ hợp lệ trở thành một dòng trong bảng đặc trưng.

## 2. Chuẩn bị môi trường

Mở PowerShell tại thư mục wesad-stress-detection,
nơi có scripts/, configs/, src/ và requirements.txt.

Tạo môi trường:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Dự án ghi phiên bản Python thống nhất trong .python-version.
Công cụ kiểm tra gói G1 yêu cầu phiên bản patch khớp file này.

Nếu chỉ cài requirements.txt, các phiên bản có thể khác giữa máy.
Không coi các kết quả đó đã được nghiệm thu cùng môi trường.

Để chạy notebook EDA, cần cài thêm matplotlib và kernel nếu chưa có:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib ipykernel
```

## 3. Chuẩn bị đường dẫn dữ liệu

data_root phải trỏ tới thư mục chứa trực tiếp các thư mục:

- S2/S2.pkl
- S3/S3.pkl
- ...
- S11/S11.pkl
- S13/S13.pkl
- ...
- S17/S17.pkl

Tổng cộng 15 người: S2–S11 và S13–S17.

Sao chép cấu hình đường dẫn riêng:

```powershell
Copy-Item configs/local.example.yaml configs/local.yaml
```

Sửa configs/local.yaml, ví dụ:

```yaml
data_root: "D:/Datasets/WESAD/WESAD"
```

Chỉ đặt data_root trong local.yaml.
Không đặt sensors, subjects hoặc output_dir trong file này.
Không commit đường dẫn riêng của máy lên Git.

Chỉ đọc PKL từ nguồn WESAD tin cậy.

## 4. Cấu hình trích xuất

File: configs/features_smoke.yaml

```yaml
data_root: data/raw/WESAD/WESAD
device: chest
sensors: [ECG, EDA, Resp]
sampling_rate_hz: 700
labels: [1, 2, 3]

window_seconds: 60
step_seconds: 5
label_rule: majority
min_label_fraction: 0.90

feature_version: chest_raw_stats_majority_v1
subjects: null
output_dir: data/features/chest_raw_stats_majority_v1
```

subjects: null nghĩa là tự tìm các thư mục subject có trên máy.
Nó không bảo đảm dữ liệu đã có đủ 15 người.

--subjects trên dòng lệnh được ưu tiên hơn subjects trong YAML.
--local ghi đè data_root.
--output-dir ghi đè thư mục kết quả.

Script mặc định dùng configs/features_all_sensors.yaml. Các lệnh cho bộ 21 feature bên dưới vẫn
truyền --config rõ ràng để người chạy dễ đối chiếu cấu hình.

## 5. Cách chia cửa sổ và gán nhãn

Các khoảng thời gian bắt đầu là:

- [0, 60) giây
- [5, 65) giây
- [10, 70) giây
- ...

Mỗi cửa sổ có 42.000 mẫu; mỗi bước dịch 3.500 mẫu.
Hai cửa sổ liên tiếp chồng lấn 55 giây.

Với mỗi cửa sổ:

1. Đếm các nhãn trên toàn bộ 42.000 mẫu.
2. Tìm nhãn có số mẫu lớn nhất.
3. Chỉ giữ nếu nhãn đó thuộc 1/2/3 và chiếm ít nhất 90%.
4. Loại cửa sổ nếu tín hiệu yêu cầu có NaN hoặc Inf.
5. Tính 21 thống kê và lưu thành một dòng.

Ví dụ:

- 57 giây Stress và 3 giây Baseline: giữ, gán Stress.
- 54 giây Stress và 6 giây Baseline: giữ, gán Stress.
- 50 giây Stress và 10 giây Baseline: loại.

Không xóa nhãn ngoài 1/2/3 trước khi tính tỷ lệ.
Không nối các đoạn rời nhau sau khi loại nhãn.
Phần cuối bản ghi không đủ 60 giây không tạo cửa sổ.

## 6. Chạy thử một người

Nên dùng thư mục riêng để kết quả thử không thay bảng tổng hợp chính:

```powershell
.\.venv\Scripts\python.exe scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S2 --output-dir data/features/g1_s2_raw_stats --resume
```

Kiểm tra gói S2:

```powershell
.\.venv\Scripts\python.exe scripts/check_g1_sample.py --features-dir data/features/g1_s2_raw_stats --output outputs/g1/S2_acceptance.json
```

Công cụ này kiểm tra gói raw-stat S2 theo hợp đồng G1.
Kết quả passed chưa có nghĩa đã đối chiếu đủ bốn máy
hoặc đã hoàn thành feature chính của G2.

## 7. Chạy đủ 15 người và ghép bảng

Chỉ định đủ danh sách để phát hiện ngay khi thiếu file:

```powershell
.\.venv\Scripts\python.exe scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17 --resume
```

Chương trình đọc từng PKL tuần tự, lưu kết quả theo người,
sau đó tự ghép thành:

data/features/chest_raw_stats_majority_v1/all_subjects.csv

Không cần tự chia thành 15 bộ train/test.

Nếu chỉ chạy lô của Nam:

```powershell
.\.venv\Scripts\python.exe scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S3 S8 S14 S16 --output-dir data/features/nam_raw_stats --resume
```

Quan trọng: all_subjects.csv chỉ chứa các subject được chọn
trong lần chạy đó, không tự cộng dồn mọi CSV sẵn có trong thư mục.

Chạy từng subject riêng liên tiếp trong cùng thư mục sẽ tạo lại
all_subjects.csv theo lần chạy cuối.

Nếu mọi PKL có trên một máy, chạy lại với danh sách đủ 15 người
và --resume để kiểm tra, dùng lại các CSV hợp lệ và ghép đủ bộ.

Nếu chia lô trên nhiều máy, cần bàn giao và kiểm tra gói trước
khi hợp nhất. Mã hiện tại chưa có CLI hợp nhất gói nhiều máy.
Không ghép cả all_subjects.csv của lô và CSV từng người,
vì sẽ gây trùng dữ liệu.

## 8. Các file đầu ra

| File | Nội dung |
|---|---|
| S2.csv, S3.csv, ... | Các cửa sổ và đặc trưng của từng người |
| S2.json, S3.json, ... | Hash nguồn/CSV, số cửa sổ, thời gian và thông tin hoàn tất |
| all_subjects.csv | Bảng ghép của các subject được chọn trong lần chạy |
| quality_report.csv | Số cửa sổ và số lượng từng nhãn theo người |
| feature_schema.json | Danh sách 21 feature thực tế của gói này |
| config.json | Cấu hình xử lý, chữ ký, mã Git và môi trường |
| checksums.json | Hash của bảng, schema và các CSV theo người |
| run_status.json | Trạng thái và phạm vi lần chạy |
| run.log | Nhật ký trích xuất |

quality_report.csv hiện chưa có đầy đủ số cửa sổ trước/sau loại
và nguyên nhân loại. Không diễn giải nó như báo cáo QA sinh lý đầy đủ.

status: complete chỉ có nghĩa hoàn tất các subject được chọn.
Phải kiểm tra danh sách subject trước khi kết luận đủ 15 người.

## 9. Cấu trúc bảng đặc trưng

Tám cột metadata:

- subject_id: mã người.
- window_id: mã người kết hợp chỉ số bắt đầu trên lưới 700 Hz.
- start_s: thời điểm bắt đầu.
- end_s: thời điểm kết thúc, không bao gồm điểm cuối.
- label_original: 1/2/3.
- label_binary: Stress = 1; Baseline/Amusement = 0.
- label_fraction: tỷ lệ nhãn đa số.
- quality_flag: phạm vi kiểm tra chất lượng đã thực hiện.

Có 21 cột feature: 7 thống kê cho mỗi ECG, EDA và Resp.

Không đưa metadata vào X khi huấn luyện.
subject_id được dùng để chia nhóm trong LOSO.

quality_flag = finite_only_not_physiological_QA không chứng minh
tín hiệu không có nhiễu hoặc đạt chất lượng sinh lý.

Không dùng configs/feature_schema.json nháp 68 feature cho bảng này.
Dùng feature_schema.json được sinh cùng thư mục CSV.

## 10. Chạy tiếp và thay phiên bản

--resume kiểm tra cấu hình, mã, môi trường, hash PKL,
hash CSV và manifest trước khi dùng lại kết quả theo người.

Nếu có lỗi:

Output contains different processing/code/environment

hãy chọn thư mục output mới bằng --output-dir.
Không sửa hoặc xóa chữ ký trong manifest để ép dùng lại dữ liệu.

Đổi code, môi trường hoặc tham số xử lý có thể làm chữ ký thay đổi.

## 11. EDA và huấn luyện

Sau khi có bảng đủ 15 người:

1. Cập nhật notebook EDA sang bộ feature 60/5.
2. Kiểm tra số người, cửa sổ, nhãn, NaN/Inf và khóa trùng.
3. Cập nhật biểu đồ và nhận xét bằng kết quả mới.
4. Chạy LOSO theo subject_id.

Notebook EDA 60/60 cũ cần sửa cả kiểm tra bước trượt,
quy tắc nhãn và đường dẫn đầu ra; không chỉ đổi đường dẫn CSV.

Chạy KNN/SVM mặc định cho hai bài toán:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/experiment.yaml --models knn svm --task both --resume
```

Lệnh này dùng nested: false trong config hiện tại.
Muốn bật tuning phải chọn --nested và dùng thư mục output riêng.

Imputer, scaler, selection và resampling phải học trong train
của từng fold. Không chuẩn hóa toàn bảng trước LOSO.

## 12. Lỗi thường gặp

| Hiện tượng | Kiểm tra |
|---|---|
| Lệnh cũ trỏ tới cấu hình đã bỏ | Cập nhật main và dùng configs/features_smoke.yaml |
| Không tìm thấy Sx/Sx.pkl | data_root phải chứa trực tiếp các thư mục Sx |
| all_subjects.csv chỉ có một vài người | Xem --subjects và phạm vi lần chạy cuối |
| Schema thiếu các cột chest_ecg_... | Có thể đang dùng nhầm schema nháp 68 feature |
| Số cửa sổ khác bộ 60/60 | Do đổi bước trượt và quy tắc nhãn |
| Notebook báo ngoài lưới 60 giây | Notebook vẫn kiểm tra bộ 60/60 cũ |
| Resume báo khác chữ ký | Dùng output mới hoặc khôi phục đúng phiên bản xử lý cũ |

## 13. Bàn giao G1 phần Như

Đã cập nhật theo README này: [thiết kế đồng bộ và bộ lọc](docs/G1_NHU_THIET_KE_TIEN_XU_LY.md), [hướng dẫn bàn giao](docs/G1_NHU_BAN_GIAO.md) và [kiểm tra mẫu S2](docs/G1_NHU_KIEM_TRA_S2.md).

Bộ G1 cũ dùng `--config configs/features_smoke.yaml` tường minh; CLI hiện mặc định dùng `features_all_sensors.yaml`; hàm cửa sổ mặc định dùng 60 giây, bước 5 giây, majority >=90%. Trên máy Như, môi trường Python 3.12.14 hiện có tên `.venv-g1`; khi làm theo các lệnh README, thay `.venv` bằng `.venv-g1` nếu dùng môi trường đã cài này. Mẫu S2 đạt 395 cửa sổ và bộ 25 test đạt. Cấu hình/số cửa sổ S2 khớp bảng EDA bước 5 giây của Nam; nghiệm thu đầy đủ giữa các máy vẫn chờ gói đối chiếu.

`configs/preprocessing_g1_design.yaml` chỉ là hợp đồng thiết kế G1, `runtime_ready: false`; không truyền file đó vào extractor. Phần triển khai bộ lọc và feature sinh lý G2 chưa được thực hiện.

