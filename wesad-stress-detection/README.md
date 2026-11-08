> **G3 hiện tại dùng bộ sinh lý 68 đã nghiệm thu**, cấu hình `configs/defaults.yaml`; xem [bàn giao G3 Dummy/LR](docs/G3_BAN_GIAO.md). Hướng dẫn raw112 bên dưới được giữ để truy vết G1/G2 cũ.

> Nhóm chốt bộ 112 thống kê raw ngực/cổ tay. Dùng `configs/features_smoke.yaml` và `configs/experiment.yaml`; hướng dẫn đầy đủ tại [CHIA_NHO_DATA.md](docs/CHIA_NHO_DATA.md).

# chia cửa sổ và trích xuất dữ liệu WESAD

Cập nhật G2 sinh lý: **đã chốt G2 bản cơ bản** theo [biên bản nghiệm thu](docs/G2_NGHIEM_THU.md). Bộ chung đủ 15 người, 6.189 cửa sổ, 68 feature + 8 metadata, cùng nguồn của Như và Python 3.12.14; đã kiểm tra schema, khóa, manifest, checksum và quality report. Giữ cùng tập cửa sổ cho ba nhóm cảm biến, giữ NaN có lý do và công bố hạn chế BVP; không đổi phương pháp để ép hết thiếu. Gói ngoài Git: `data/features/g2_physiological68_group15_20261107.zip`; xem [bàn giao](docs/G2_BAN_GIAO.md). Không thay đổi bộ raw112 trong hướng dẫn dưới đây hoặc thực hiện mô hình G3 trong lượt chốt này.

Phần G2 của Như có [bản tiền xử lý cơ bản và kết quả lô S4/S9/S11/S17](docs/G2_NHU_BAN_CO_BAN.md): lọc EDA/RESP, căn cửa sổ theo tần số gốc và kiểm tra ACC/EMG theo đoạn 5 giây. Bảng chẩn đoán G2 được lưu riêng; bộ 112 raw trong hướng dẫn dưới đây giữ nguyên định nghĩa.

Cập nhật sau phần Mai `f24cafc`: đã nối provider tiền xử lý với extractor sinh lý68 và thử S2 (395 cửa sổ, 68 feature + 8 metadata; 68 test đạt). Đây là nhánh thử riêng dùng `configs/features_g2.yaml`, còn nhiều giá trị thiếu BVP cần rà với Mai; chưa nghiệm thu bộ sinh lý hoặc thay thế raw112. Chi tiết và lệnh chạy ở đầu tài liệu G2 của Như nêu trên.

Theo yêu cầu bàn giao tiếp theo, đã chạy lô sinh lý S4/S9/S11/S17: 1.658 cửa sổ; ZIP tại `data/features/g2_nhu_physiological68_20261107.zip` ngoài Git để người dùng gửi Mai đồng bộ. Giá trị thiếu/cờ chất lượng được giữ nguyên, chưa xác nhận nghiệm thu sinh lý 15 người. Xem bảng kết quả và kiểm tra gói trong tài liệu G2 của Như.

Đã nhận và kiểm tra ZIP sinh lý S5/S6/S10 của Mai sau `4b31246`. Công cụ `scripts/check_g2_bvp_quality.py` kiểm tra gói và tổng hợp thiếu nhịp BVP theo subject/nhãn cho hai lô: 7 subject, 2.910 cửa sổ; giữ nguyên thuật toán v1, chưa ghép bảng huấn luyện. Bảng chất lượng và giới hạn được ghi ở đầu tài liệu G2 của Như; toàn bộ 72 test đạt.

## 1. Bộ G1 hiện tại: 112 thống kê ngực và cổ tay

Pipeline đọc dữ liệu PKL theo từng người, chia tín hiệu thành cửa sổ,
trích xuất đặc trưng, lưu CSV theo người và ghép thành all_subjects.csv.

Cấu hình bộ 112 feature trong phần hướng dẫn này:

- Thiết bị: cảm biến ngực và cổ tay.
- Ngực: ECG, EDA, EMG, Resp, Temp và ACC.
- Cổ tay: BVP, EDA, TEMP và ACC.
- Lưới nhãn/ngực: 700 Hz; cổ tay giữ tần số gốc 64/32/4 Hz.
- Độ dài cửa sổ: 60 giây.
- Bước trượt: 5 giây.
- Nhãn hợp lệ: 1 Baseline, 2 Stress, 3 Amusement.
- Chấp nhận cửa sổ nếu nhãn chiếm đa số đạt ít nhất 90%.
- Mỗi kênh có 7 thống kê: mean, std, min, max, RMS, IQR, range. ACC tách x/y/z và magnitude.
- Tổng cộng 112 đặc trưng (63 ngực + 49 cổ tay) và 8 cột metadata: 120 cột.

Đây là thống kê tín hiệu thô; chưa có lọc nhiễu sinh lý,
HRV hoặc SCR. Processor hiện giữ tần số gốc từng kênh và tách ACC x/y/z/magnitude; chưa lọc hoặc tổng hợp ACC/EMG theo đoạn 5 giây.

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
# Raw statistics for every recorded chest/wrist sensor. Not the 68-feature physiology draft.
data_root: data/raw/WESAD/WESAD
device: combined
sampling_rate_hz: 700  # label grid only; sensors retain their native sample rates
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

subjects: null nghĩa là tự tìm các thư mục subject có trên máy.
Nó không bảo đảm dữ liệu đã có đủ 15 người.

--subjects trên dòng lệnh được ưu tiên hơn subjects trong YAML.
--local ghi đè data_root.
--output-dir ghi đè thư mục kết quả.

Script mặc định dùng configs/features_smoke.yaml. Các lệnh vẫn truyền --config rõ ràng để đối chiếu cấu hình.

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
5. Cắt cùng khoảng thời gian ở từng kênh theo tần số gốc; tính 112 thống kê và lưu một dòng.

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
.\.venv\Scripts\python.exe scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S2 --output-dir data/features/g1_s2_all_sensors_raw_stats_v1 --resume
```

Kiểm tra gói S2:

```powershell
.\.venv\Scripts\python.exe scripts/check_g1_sample.py --features-dir data/features/g1_s2_all_sensors_raw_stats_v1 --output outputs/g1/S2_all_sensors_acceptance.json
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

data/features/all_sensors_raw_stats_v1/all_subjects.csv

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
| feature_schema.json | Danh sách 112 feature thực tế và nhóm ngực/cổ tay/kết hợp |
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

Có 112 cột feature: 7 thống kê cho mỗi kênh; ACC có x/y/z và magnitude. Nhóm ngực có 63 cột, cổ tay có 49 cột.

Không đưa metadata vào X khi huấn luyện.
subject_id được dùng để chia nhóm trong LOSO.

quality_flag = finite_only_not_physiological_QA không chứng minh
tín hiệu không có nhiễu hoặc đạt chất lượng sinh lý.

Schema sinh lý 68 feature trong configs/feature_schema.json là hợp đồng riêng, không dùng để kiểm tra bảng raw 112 này.
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
| Schema thiếu các cột chest_ecg_... | Kiểm tra đúng phiên bản schema raw 112 hoặc hợp đồng sinh lý 68 |
| Số cửa sổ khác bộ 60/60 | Do đổi bước trượt và quy tắc nhãn |
| Notebook báo ngoài lưới 60 giây | Notebook vẫn kiểm tra bộ 60/60 cũ |
| Resume báo khác chữ ký | Dùng output mới hoặc khôi phục đúng phiên bản xử lý cũ |

## 13. Bàn giao G1 phần Như theo bộ 112

Đã cập nhật [thiết kế G1](docs/G1_NHU_THIET_KE_TIEN_XU_LY.md), [bàn giao](docs/G1_NHU_BAN_GIAO.md) và [kết quả S2](docs/G1_NHU_KIEM_TRA_S2.md) theo bộ 112 nhóm đã chốt. Công cụ check_g1_sample.py kiểm tra đúng 112 feature, 63 ngực/49 cổ tay, phiên bản all_sensors_raw_stats_v1; từ chối gói 21 cũ.

Máy Như dùng Python 3.12.14 trong .venv-g1; thay .venv bằng .venv-g1 trong các lệnh nếu dùng môi trường đã cài này. Gói S2 hiện tại có 395 cửa sổ và 120 cột; 35 test đạt. Ngày 06/11/2026 đã đối chiếu đủ bốn gói S2, CSV giống hệt nhau; nhóm chốt phần G1 của Như theo tiêu chí đầu ra khớp nhau. Khác biệt nguồn chạy/môi trường vẫn được ghi nhận trong [biên bản đối chiếu](docs/G1_DOI_CHIEU_S2_BON_THANH_VIEN.md).

Thiết kế bộ lọc trong configs/preprocessing_g1_design.yaml chưa áp dụng lên bộ raw 112. Bộ đang chọn dùng thống kê trực tiếp trong cửa sổ 60 giây, bước 5 giây; không tự đổi định nghĩa feature khi nghiệm thu mẫu.

## 14. Bàn giao G1 phần Mai

Đầu ra chính: [schema](configs/feature_schema.json), [danh mục feature](docs/G1_DANH_MUC_FEATURE.csv), [hợp đồng](docs/G1_HOP_DONG_FEATURE.md), [bàn giao và kết quả tự kiểm tra S2](docs/BAN_GIAO.md). Bằng chứng đơn vị được gộp trong schema/hợp đồng. Hợp đồng 68 feature sinh lý đã chốt thiết kế, chưa triển khai/kiểm chứng extractor; bộ raw hiện tại vẫn có 112 feature theo cấu hình nhóm.

Cả bốn chạy mẫu S2 theo phân công. Mai tự kiểm tra gói máy mình; Như nhận ZIP và tổng hợp đối chiếu theo cách nhóm đã xác định. Máy Mai: Python 3.12.14, source fa7b694, 395 cửa sổ, 112 feature, kiểm tra một gói passed; lượt kiểm tra gói đơn này có trước biên bản đối chiếu đủ bốn gói ngày 06/11/2026 nêu trên. Schema raw lấy từ file sinh cùng CSV, không tạo thêm danh mục/schema trùng lặp trong Git. Gói dữ liệu gửi ngoài Git, không gửi môi trường Python.

## 15. Tổng hợp chốt G1 — 06/11/2026

Mai đã tổng hợp [bàn giao và kết luận G1](docs/BAN_GIAO.md) dựa trên đầu ra bốn người và [biên bản đối chiếu của Như](docs/G1_DOI_CHIEU_S2_BON_THANH_VIEN.md). G1 được chốt theo tiêu chí đầu ra S2 giống nhau mà nhóm đã chấp nhận: cả bốn CSV giống từng byte, 395 cửa sổ, 112 feature + 8 metadata; sai lệch feature lớn nhất 0.

Khác biệt source/môi trường vẫn được ghi rõ, kiểm tra tái lập nghiêm ngặt chưa đạt; không sửa manifest/JSON cũ để ép đạt. Lần chạy chung tiếp theo cần thống nhất source sạch và môi trường khóa. Kết luận này chỉ chốt phạm vi G1, không xác nhận đã triển khai/nghiệm thu bộ sinh lý68 của G2 hoặc đã hoàn tất các lượt review cá nhân.
