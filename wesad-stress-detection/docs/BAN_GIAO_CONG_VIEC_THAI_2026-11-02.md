# Bàn giao công việc Thái đã hoàn thành ngày 02 tháng 11 năm 2026

Người phụ trách: Trần Trung Thái. Dự án: WESAD nhóm 09. Bản bàn giao dành cho Nam, Như, Mai tiếp tục triển khai trong ngày.

## 1. Kết quả đã làm

Đã bổ sung phần đọc raw, lệnh chạy, resume, LOSO và nested CV. Phân công mô hình theo proposal: Mai làm Dummy/LR, Nam làm KNN/SVM, Như làm RF/MLP.

| Phần việc | Kết quả thực tế |
| --- | --- |
| Môi trường | Tạo `.venv`, cài package editable; khóa thư viện trong `requirements.lock.txt`; `pip check` không có phụ thuộc bị lỗi |
| Package và cấu hình | Có `pyproject.toml`; config chung và `configs/local.yaml` riêng theo máy; local chỉ được ghi data_root |
| Đọc dữ liệu | Đọc một PKL mỗi lần; kiểm tra subject, khóa, shape; kiểm tra bảng feature và không đưa metadata vào X |
| Cửa sổ | Giữ homogeneous cũ; thêm majority 90%, tính cả nhãn không dùng trong mẫu số; cửa sổ theo thời gian gốc |
| Extractor | Chọn subject bằng CLI; ghi file nguyên tử, manifest và checksum; resume chỉ nhận file khớp raw/code/config/schema/thư viện |
| Đánh giá | LOSO theo người, hai task; imputer/scaler trong pipeline; metric và dự đoán ngoài mẫu |
| Tuning | Inner GroupKFold 3 lượt nằm trong outer LOSO; hỗ trợ KNN/SVM/RF/MLP; lưu best_params và splits từng fold |
| Mất cân bằng | Có lựa chọn class_weight cho LR/SVM/RF; RandomOverSampler trong fit của KNN/MLP |
| Tích hợp mô hình | Pipeline khởi đầu cho 6 mô hình; có factory để chủ mô hình thay thế mà không sửa evaluation |
| Resume thực nghiệm | Checkpoint từng task/model/outer subject; kiểm tra hash trước khi bỏ qua; fold hỏng được chạy lại |
| Tài liệu | README đã cập nhật bằng giao diện lệnh thật, hướng dẫn API và giới hạn hiện tại |

Mã và tài liệu nằm trong repository chung. Dữ liệu, môi trường và output chia sẻ riêng.

## 2. Kiểm tra đã thực hiện

### Kiểm tra tự động

**14 kiểm tra đều đạt**, gồm:

- Nhãn đúng ở ranh giới 90%; loại nhãn không dùng; không nối các đoạn; bỏ phần cuối thiếu độ dài.
- Loại metadata khỏi feature, schema đúng thứ tự; phát hiện trùng khóa, Inf, nhãn và thời gian không hợp lệ.
- Local config không được sửa phương pháp.
- Resume extractor: bỏ qua bảng đúng; trích lại bảng bị hỏng; từ chối config khác trong cùng thư mục.
- LOSO không lẫn subject; scaler chỉ fit train; dữ liệu giả đủ 15 người mới chuyển complete.
- Nested CV của cả bốn mô hình: inner train/validation không có outer test; kiểm tra resume và phục hồi fold hỏng.
- Sampler không làm tăng số dự đoán trên test.

Lệnh chạy lại:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Các test dùng dữ liệu giả trong thư mục tạm. Trường hợp 15 người kiểm tra trạng thái hoàn tất của chương trình; số liệu WESAD thực tế ghi ở phần dưới.

### Raw thật của lô Thái

Đã trích xuất thật ECG/EDA/Resp ngực, cửa sổ 60 giây, bước 5 giây, majority 90%:

| Subject | Tổng cửa sổ | Baseline | Stress | Amusement |
| --- | ---: | ---: | ---: | ---: |
| S2 | 395 | 219 | 113 | 63 |
| S7 | 411 | 228 | 118 | 65 |
| S13 | 417 | 226 | 124 | 67 |
| S15 | 418 | 226 | 128 | 64 |
| Tổng | 1.641 | 899 | 483 | 259 |

Thư mục: `data/features/chest_raw_stats_majority_v1/`. Bảng tổng có **1.641 hàng × 29 cột**, gồm 8 metadata và 21 thống kê raw. Thời gian ghi trong manifest khoảng 20 giây cho bốn người. Chưa đo đỉnh RAM; thời gian này chỉ áp dụng cho bộ thống kê raw.

SHA256 của `all_subjects.csv`:

```text
f0e6939a33254cfebb29e91ce55f5058c748b24dc4b2dc75151c727d031565cc
```

Đã chạy lại với `--resume`: cả S2/S7/S13/S15 báo `SKIP verified`, không nạp PKL để trích lại. Resume vẫn đọc file raw để xác minh checksum.

### Huấn luyện thật để kiểm tra hạ tầng

- Chạy Dummy, LR, KNN, SVM, RF, MLP; cả ba lớp và nhị phân; LOSO trên bốn subject của Thái: **48 outer folds đã hoàn tất**.
- Chạy nested tuning cả KNN/SVM/RF/MLP, hai task, outer test S2: **8 outer folds đã hoàn tất**, mỗi fold có 3 inner folds và 4 ứng viên. Đã chạy lại và xác nhận cả 8 checkpoint đều được skip.
- Cả hai run có trạng thái **`partial`** do dữ liệu mới gồm bốn người; chưa có điểm đánh giá đủ 15 người.
- Một số fold MLP đạt giới hạn 200 vòng lặp mà chưa hội tụ. Cảnh báo được lưu trong `details.json`; phần kiểm tra loss và max_iter giao Như.

Các thư mục kết quả trên máy Thái:

```text
outputs/experiments/thai_raw_stats_defaults/
outputs/experiments/thai_raw_stats_nested_quick/
```

## 3. Công việc tiếp theo

### Như nhận phần tiền xử lý và RF MLP

1. Kiểm tra hàm chia cửa sổ trong `preprocessing.py`, gồm homogeneous và majority.
2. Bổ sung đồng bộ ngực/cổ tay, lọc nhiễu và kiểm tra chất lượng tín hiệu theo proposal.
3. Phối hợp Mai xử lý ACC/EMG theo đoạn 5 giây rồi gộp trong 60 giây.
4. Hoàn thiện pipeline RF/MLP và kiểm tra hội tụ.
5. Giữ preprocessing học thống kê trong pipeline của fold, ghi rõ tên/phiên bản feature mới.

### Mai nhận feature baseline và bảng chung

1. Tiếp tục đặc trưng sinh lý: ECG/BVP, HRV, EDA tonic/phasic/SCR, RESP, TEMP, ACC/EMG; thêm ngực/cổ tay như proposal.
2. Viết hàm `process_subject(path, config) -> DataFrame` và đăng ký bằng `subject_processor: wesad.features:process_subject`.
3. Giữ `subject_id`, `window_id`, `start_s`, `end_s`, `label_original`, `quality_flag`; có thể thêm `label_binary`, `label_fraction`. Các cột còn lại phải là feature số.
4. Lưu bộ đặc trưng sinh lý vào output_dir riêng, kèm phiên bản và schema thống nhất.
5. Phụ trách Dummy/LR và kiểm tra metric; hợp nhất đủ 15 người sau khi xác minh cùng phiên bản, schema và không trùng khóa.

### Nam nhận EDA KNN SVM và kiểm định

1. Dùng bảng feature đã lưu để làm EDA, không nạp đồng thời 15 PKL.
2. Tạo pipeline KNN/SVM hoặc module factory riêng; giữ tên bước `model` để sử dụng lưới `model__...`.
3. Chốt tham số tìm kiếm trong config trước lượt đánh giá; `experiment.yaml` đang có bốn tổ hợp cho mỗi mô hình.
4. Khi có bảng chung đủ 15 người, chạy qua `run_experiment.py`; nhận `metrics_by_subject.csv` làm đầu vào Friedman/Wilcoxon/Holm.
5. Phối hợp Thái/Như làm so sánh cảm biến và hai cách chuẩn hóa; nhánh theo subject dùng thống kê test không nhãn phải báo riêng.

### Giao diện dùng chung

- Feature processor: `process_subject(path: Path, config: dict) -> DataFrame`.
- Model factory: `build_pipeline(*, name: str, config: dict) -> Pipeline chưa fit`.
- Đăng ký model factory bằng `model_factories` trong YAML, ví dụ `svm: wesad.models_neighbors_svm:build_pipeline`.
- Không đưa mã người/thời gian/nhãn vào X; không fit imputer/scaler/selection toàn bảng; không resample validation/test.
- Gửi cùng mã nguồn, config, schema và checksum khi bàn giao dữ liệu. Khác mã hoặc môi trường phải dùng output phiên bản mới.

## 4. Các lệnh đã chạy trên máy Thái

Chạy tại thư mục `DOAN/SGU26_Group09_Machine_Learning/wesad-stress-detection`.

```powershell
# Lô raw Thái
.\.venv\Scripts\python.exe scripts/extract_features.py `
  --config configs/features_smoke.yaml --local configs/local.yaml `
  --subjects S2 S7 S13 S15 --resume

# Sáu mô hình, hai task, đánh giá thử bốn người
.\.venv\Scripts\python.exe scripts/run_experiment.py `
  --config configs/experiment.yaml --allow-partial `
  --output outputs/experiments/thai_raw_stats_defaults --resume

# Kiểm tra nested cho bốn mô hình trên outer S2
.\.venv\Scripts\python.exe scripts/run_experiment.py `
  --config configs/experiment.yaml --models knn svm rf mlp --nested `
  --outer-subjects S2 --allow-partial `
  --output outputs/experiments/thai_raw_stats_nested_quick --resume
```

Máy khác tạo `.venv`, cài `requirements.lock.txt`, cài editable và tạo local.yaml như README. Không sao chép môi trường từ máy Thái. Raw và outputs không nằm trong Git; muốn chạy lại trên bảng hôm nay cần nhận gói feature từ Thái qua nơi chia sẻ dữ liệu của nhóm.

## 5. Phần còn lại

- Chưa có bộ feature sinh lý, cổ tay và kết hợp cuối cùng; dữ liệu hôm nay chỉ là thống kê raw để kiểm tra luồng.
- Chưa có kết quả WESAD đủ 15 người, tuning đầy đủ hay kết luận mô hình tốt nhất.
- Chọn feature theo tương quan/RF, các so sánh cảm biến và chuẩn hóa theo subject còn chờ chủ phần tích hợp. Runner hiện chỉ nhận `normalization: global_train` để không đánh đồng hai giả định.
- Chưa chạy kiểm định thống kê thực nghiệm chính, chưa có báo cáo/slide cuối.
- Chưa tạo Issue/PR hoặc gửi tin nhắn trực tiếp cho thành viên. Các bạn nhận mã và file bàn giao từ repository chung; gói feature cần chia sẻ riêng.

Kết quả hiện tại mới xác nhận luồng chạy trên bộ thống kê raw của bốn người.

## 6. Những file cần nhận khi tích hợp

`pyproject.toml`, `requirements.txt`, `requirements.lock.txt`, `.gitignore`, README; hai CLI trong scripts; `data.py`, `utils.py`, `extraction.py`, `evaluation.py`, `models.py` và phần thay đổi `preprocessing.py`; config dùng chung, file mẫu local, tests và bản bàn giao này.

`.venv`, config đường dẫn riêng và output nằm trong `.gitignore`. Checkpoint và manifest lưu trạng thái từng subject hoặc fold để resume.

Git root là `SGU26_Group09_Machine_Learning`. Mai phụ trách kiểm tra chéo phần Thái bằng các test và lệnh ở trên.

Ghi chú cập nhật: lần chỉnh diễn đạt sau bàn giao làm thay đổi mã băm source. Các output ngày 02/11 giữ mã băm của phiên bản đã chạy; lượt chạy bằng source mới dùng output_dir mới.
