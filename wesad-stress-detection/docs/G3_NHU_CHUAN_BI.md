# G3 phần Như — RF/MLP cơ bản, cập nhật 08/11/2026

> Cập nhật mới nhất: đã chạy đủ 15 lượt LOSO cho RF/MLP ở cả hai bài toán, 60 fold và 24.756 dự đoán. Xem [kết quả G3 của Như](G3_NHU_KET_QUA.md). Phần bên dưới giữ lịch sử chuẩn bị/quick; không dùng các điểm quick làm điểm main.

> Sau cập nhật Mai `813a11b`, gói RF/MLP đã được nhận và hợp nhất. Như đã rà toàn bộ 60 dòng metric và bốn nhóm summary trong bảng nhóm, đều khớp gốc; phần G3 cơ bản của Như hoàn tất, không chạy lại hoặc tự mở G4.

Nhóm đã chốt dùng **68 sinh lý** (`physiological_68_v1`). Đã nhận đúng gói chung, kiểm tra và chạy thử RF/MLP mặc định trên WESAD thật: 14 người train, S4 test. Chưa chạy đủ 15 lượt LOSO, chưa tuning hoặc kết luận hiệu năng chính thức.

## Đầu vào chung đã nhận

- ZIP: `g2_physiological68_group15_20261107.zip`, SHA-256 `68f10ae940e4c1b3d269cf5836803800f3460a92755a5494cd8468d479075c50`.
- Đã giải nén ngoài Git tại `data/features/g2_physiological68_group15_20261107/`.
- Bảng có **6.189 cửa sổ, 76 cột = 68 feature + 8 metadata**, đủ S2–S11 và S13–S17.
- SHA-256 CSV: `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba`, khớp khóa dữ liệu trong `configs/defaults.yaml`.
- Đã kiểm tra checksum gói, schema/thứ tự cột, đủ subject, khóa trùng và Inf. NaN/cờ chất lượng giữ nguyên; không điền trước trên toàn bảng.
- File all_subjects.csv gửi riêng trước đó có 120 cột và checksum `de34cf16aa21e5e15bc61f5c4ac22b859fc7237c048e82fd1b9e63ddef662b91` là **raw112**, không dùng cho lượt sinh lý này.

Số cửa sổ sinh lý khác raw112 (6.189 so với 6.196); không ép bằng nhau. Lần kiểm tra này dùng đúng bảng chung có checksum được chốt.

## Phần mã cơ bản

| File | Nội dung |
| --- | --- |
| `src/wesad/models_rf_mlp.py` | Factory riêng RF/MLP, gọi runner chung của Thái |
| `configs/g3_nhu_defaults.yaml` | Seed 42, hai bài toán, combined_physiological68; features=null vì CLI yêu cầu chỉ rõ file |
| `scripts/run_nhu_g3_quick.py` | Kiểm tra đúng schema 68 trước khi train, chạy một outer subject và ghi rõ phạm vi quick |
| `tests/test_models_rf_mlp.py` | Kiểm tra mặc định, train-only preprocessing, LOSO/resume và chặn raw112 |

CLI từ chối raw112, sai phiên bản hoặc sai tên/thứ tự feature. Việc nhóm chọn bộ 68 đã được ghi nhận; `features=null` chỉ buộc truyền đường dẫn rõ ràng, không còn mang nghĩa nhóm chưa chọn loại dữ liệu.

## Mặc định và xử lý thiếu

- RF: `RandomForestClassifier(random_state=42, n_jobs=1)`, các tham số estimator khác giữ mặc định của phiên bản khóa.
- MLP: `MLPClassifier(random_state=42)`, hidden_layer_sizes=(100,), max_iter=200, không tự tăng max_iter hoặc bật early stopping.
- Cả hai có median imputer với keep_empty_features=True và StandardScaler, chỉ fit trên train của từng fold. Cột hoàn toàn thiếu trong train được giữ theo fallback 0 của imputer; không học giá trị thay thế từ test.
- Không tuning, selection, balancing hoặc resampling. X chỉ gồm feature theo schema, không dùng mã người, thời gian, nhãn hoặc quality_flag.
- Tận dụng runner hiện có để lưu dự đoán, điểm, splits, thời gian fit+predict, warnings và confusion matrix dạng số. Không tạo một bộ LOSO riêng.

## Lượt thử đúng bảng 15 người

Train 14 người trừ S4; test 404 cửa sổ S4. Hai mô hình × hai bài toán = bốn lượt fit. Output ngoài Git: `outputs/g3_nhu_phys68_group15_quick_20261108/`.

| Bài toán | Mô hình | Macro-F1 trên S4 | Balanced Accuracy trên S4 |
| --- | --- | ---: | ---: |
| Ba lớp | RF | 0.6241 | 0.6667 |
| Ba lớp | MLP | 0.6212 | 0.6610 |
| Nhị phân | RF | 1.0000 | 1.0000 |
| Nhị phân | MLP | 1.0000 | 1.0000 |

Đây chỉ là **một subject test**, không phải điểm trung bình 15 người. Điểm nhị phân 1.0 trên S4 không đủ để kết luận mô hình hoàn hảo hoặc tổng quát hóa tốt. Cả bốn lượt không ghi cảnh báo trong lần chạy này. BVP thiếu vẫn là giới hạn của đầu vào; không chỉnh detector hoặc dữ liệu để làm tăng điểm.

Đã kiểm tra 1.616 dòng dự đoán: không trùng khóa theo task/model/subject/window, y_true đúng theo S4, S4 không xuất hiện trong train; tính lại Macro-F1/Balanced Accuracy và xác minh hash tệp từng fold. Status vẫn **partial**, chưa đủ 15 outer subjects. Bằng chứng nhỏ: [physiological68_group15_quick_check.json](../reports/g3_nhu/physiological68_group15_quick_check.json).

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/run_nhu_g3_quick.py --features data/features/g2_physiological68_group15_20261107/all_subjects.csv --test-subject S4 --output outputs/g3_nhu_phys68_group15_quick_lan2
```

Dùng output mới để giữ bằng chứng. Không đưa các dự đoán quick vào bảng G3 chính thức. Macro-F1 theo tập nhãn cố định, zero_division=0; Balanced Accuracy theo các lớp thực sự có trong test như runner chung.

## Kiểm tra và lịch sử

- Toàn checkout **90/90 test đạt**, có test chặn input raw112 và nhận đúng hợp đồng 68. Sửa fixture baseline để đọc hợp đồng trong Git, không phụ thuộc ZIP dữ liệu bị ignore; không thay code/mô hình baseline.
- Bản chuẩn bị trước đó kiểm tra bằng dữ liệu giả; [preparation_check.json](../reports/g3_nhu/preparation_check.json) giữ nguyên là bằng chứng lịch sử, không chứa điểm mô hình giả làm kết quả đồ án.
- Trước khi có ZIP chung, đã thử lô thật S4/S9/S11/S17 với 3 train / 1 test; bằng chứng riêng ở [physiological68_local4_quick_check.json](../reports/g3_nhu/physiological68_local4_quick_check.json). Không trộn lượt này với lượt 14 train / 1 test phía trên.

## Các đợt tiếp theo

Bước tiếp theo là chạy RF/MLP đủ 15 outer subjects theo giao thức G3 chung và giữ riêng output main với quick, rồi làm hình từ dự đoán thật. Cần dùng đúng khóa CSV trên và thống nhất giao diện bàn giao với Thái/Mai. Bản cập nhật này dừng ở lượt thử; chưa thêm hình hoặc thực hiện G4. Nếu cần đổi max_iter vì không hội tụ, phải báo riêng cấu hình thay đổi, không gọi đó là mặc định.

Người dùng tự commit và push. Dữ liệu CSV/ZIP, output huấn luyện và môi trường Python vẫn ngoài Git.
