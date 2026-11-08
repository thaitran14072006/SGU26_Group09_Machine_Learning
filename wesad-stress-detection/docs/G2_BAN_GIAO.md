# Bàn giao dữ liệu G2 — physiological_68_v1

## Bàn giao hiện tại — 07/11/2026

**Hoàn thành phạm vi extractor, schema, hợp nhất feature, quality report, checksum và kiểm tra đủ 15 người. G2 bản cơ bản đã được chốt theo các tiêu chí nghiệm thu trong phân công và yêu cầu của người dùng.** Xem [biên bản nghiệm thu G2](G2_NGHIEM_THU.md) về tập cửa sổ chung, chính sách NaN và hạn chế BVP; không xác nhận thay các lượt review/chữ ký cá nhân.

Đã tạo bộ chung bằng một lượt chạy đủ S2–S11/S13–S17 trên máy hiện tại: cùng nguồn `7720b01` của Như, cùng provider/hợp đồng, Python 3.12.14 và thư viện khóa. Lượt này tạo dữ liệu chung, không đối chiếu kết quả giữa các máy. Không dùng gói Nam Python 3.13.5 hoặc trộn các chữ ký môi trường cũ vào bộ mới. Các gói bàn giao trước vẫn giữ làm bằng chứng lịch sử.

| Nội dung | Kết quả |
| --- | --- |
| Subject | Đủ 15, không thiếu; mỗi subject một lần |
| Bảng chung | 6.189 dòng, 68 feature + 8 metadata |
| Nhóm cột | Ngực 37, cổ tay 31, kết hợp 68 |
| Nhãn baseline/stress/amusement | 3370 / 1848 / 971 |
| Trùng khóa / Inf | 0 / 0 |
| S2 | Chỉ 395 cửa sổ, không lấy lại từ các gói kiểm tra |
| Raw hash, manifest, chữ ký, checksum | Đã kiểm tra |
| Bảng tổng | Khớp nối CSV từng subject một lần; không nối thêm bảng tổng khác |
| ZIP | Đã kiểm tra CRC và SHA-256 từng tệp bàn giao |
| Test checkout | 76/76 đạt |

**Gói chính:** `data/features/g2_physiological68_group15_20261107/` và ZIP cùng tên. Đọc `all_subjects.csv` khi cần bảng chung; không nối đồng thời bảng này và CSV từng subject.

**Báo cáo nhỏ:** [group15_delivery_verification.json](../reports/g2_physiological68/group15_delivery_verification.json), [group15_delivery_quality.csv](../reports/g2_physiological68/group15_delivery_quality.csv), [group15_delivery_bvp_by_label.csv](../reports/g2_physiological68/group15_delivery_bvp_by_label.csv).

Gói có schema đầy đủ, hợp đồng gốc, manifest/config nguồn, quality report, thống kê BVP theo subject/nhãn, checksum và `common_windows.csv`. Các CSV feature và manifest được giữ nguyên byte từ lượt chạy chung. Công cụ `scripts/finalize_g2_physiological.py` bổ sung hợp đồng đầy đủ vào schema đầu ra; lưu schema/checksum/run status gốc tại `producer_evidence/`, không sửa phương pháp hoặc giả chữ ký code/môi trường. Bằng chứng khôi phục nguồn từ Git nằm trong `SOURCE_RECONSTRUCTION.json`.

### Cửa sổ chung và dữ liệu thiếu

Giữ cùng 6.189 khóa cửa sổ cho ba nhóm ngực/cổ tay/kết hợp. S14 loại 7 cửa sổ đạt nhãn do biên lọc; các trường hợp còn lại có số giữ/loại trong quality report. Không áp complete-case: chỉ 345 cửa sổ đủ cả 68 feature, số này là thống kê tác động, không phải tập cửa sổ được chọn thay cho bảng chung.

BVP còn 5.312 cửa sổ thiếu sáu feature nhịp/IBI. NaN có lý do vẫn được bàn giao nguyên trạng; không hạ ngưỡng, đổi detector, điền 0, fit imputer/scaler hoặc loại cửa sổ riêng cho từng nhóm cảm biến. Nhiệt độ ngực S2 thiếu có chủ đích và biên độ SCR không có sự kiện cũng có cờ tương ứng. Thống kê thiếu đã công bố; việc rà/cải tiến phương pháp BVP tiếp tục theo bản chung Như bàn giao, không là yêu cầu Mai phải lấp đầy mọi NaN.

### Tái tạo

Lượt feature dùng nguồn riêng tại `outputs/g2_physiological_source_7720b01/wesad-stress-detection`, code hash `abd964e88fa0e5922c8bfc2cb71e1f4ad46ebb1d381e3f6bf82571817e1b452b`. Cấu hình phương pháp là `configs/features_g2.yaml`, provider `wesad.preprocessing_physiological:prepare_subject`; không đổi schema/version đã chốt. Nguồn riêng tránh các thay đổi chưa commit trong checkout ảnh hưởng chữ ký lượt chạy.

Sau khi có gói nguồn đủ 15 người, đóng gói bằng lệnh dưới đây; chọn output mới để giữ bằng chứng cũ:

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/finalize_g2_physiological.py --source data/features/g2_physiological68_group15_source_20261107 --output data/features/g2_physiological68_group15_lan2 --raw-root data/raw --report reports/g2_physiological68/group15_delivery_lan2.json
```

TODO sau bản cơ bản: rà chất lượng BVP và phương pháp detector qua đầu mối Như, kiểm chứng tín hiệu bổ sung nếu nhóm yêu cầu; không là điều kiện phải lấp hết NaN của bản nghiệm thu này. Không tự làm mô hình G3 hoặc phần RAM. Raw/ZIP/CSV feature lớn giữ ngoài Git. Phần bàn giao trước đã push trong `779b658`; các cập nhật nghiệm thu mới chưa commit/push.

## Lịch sử trước khi nhận provider và tạo bộ chung

### Kiểm tra S2 và lô S5/S6/S10 đã gộp

Các kết quả dưới đây là lịch sử trước lượt tạo bộ chung đủ 15 người, không thay thế kết luận nghiệm thu hiện tại. Đã gộp thông tin cần lưu từ hai báo cáo trung gian về S2 và lô S5/S6/S10 vào đây.

**S2:** provider nối được với extractor; 1.204 cửa sổ ứng viên, loại 809 do nhãn, giữ 395 (219 baseline, 113 stress, 63 amusement), không loại thêm do biên lọc/nonfinite/thiếu mẫu. Đủ 68 feature + 8 metadata, không trùng khóa/Inf, có 3.046 ô NaN với cờ lý do. Bảy metadata đầu khớp raw112 S2. CSV SHA-256 `aaec0dea4f42114b6574c845b76319237fef8ac96273ad92c196608c9e48eba9` khớp checksum Như công bố; lúc đó chưa có gói CSV gốc của Như tại máy để kiểm tra độc lập toàn gói. Code hash lượt thử là `469b9ef1575aba4881425ee4982caa9f8f2f72030b7c1b351360b9beb5bcabae`, khác nguồn Như do checkout có thay đổi; Python cùng 3.12.14 nhưng khác build. Không sửa manifest để ép cùng chữ ký. Lượt kiểm tra lịch sử này đạt 72/72 test; lượt nghiệm thu cuối đạt 76/76.

Sáu feature nhịp/IBI BVP S2 thiếu tại 336/395 cửa sổ: baseline 160/219, stress 113/113, amusement 63/63. Tất cả 336 cửa sổ đều có tỷ lệ khoảng nhịp hợp lệ dưới 80%; 67 cửa sổ còn thiếu cả số khoảng hợp lệ và cặp kề, 37 còn thiếu cặp kề, 1 còn thiếu số khoảng hợp lệ. Có 231 cửa sổ chỉ không đạt tỷ lệ 80%. Khoảng được chấp nhận nằm trong 0.3–2 giây và lệch không quá 20% trung vị khoảng trong miền đó; cần ít nhất 20 khoảng và 10 cặp kề hợp lệ. 59 cửa sổ đạt toàn bộ điều kiện.

Rà tín hiệu BVP 389.056 mẫu hữu hạn cho thấy ngưỡng prominence toàn đoạn là 35.5054 (0.5 độ lệch chuẩn 71.0108). Thử prominence cục bộ sau khi cắt cửa sổ cho 72 cửa sổ đạt thay vì 59, cứu 19 nhưng mất 6; thử này thay cả biên phát hiện nên chỉ là chẩn đoán. Không đưa thử nghiệm vào feature chính thức.

| Ví dụ S2 | Bắt đầu (s) | Peak BVP / ECG trong 60s | Khoảng BVP hợp lệ |
| --- | ---: | ---: | ---: |
| Baseline đạt | 1095 | 72 / 71 | 85,9% |
| Baseline thiếu, biên độ thấp | 980 | 59 / 70 | 74,1% |
| Baseline thiếu, biên độ cao | 1395 | 96 / 84 | 27,4% |
| Stress thiếu | 2550 | 46 / 74 | 44,4% |
| Amusement thiếu | 4915 | 40 / 64 | 33,3% |

Đồ thị cho thấy có dấu hiệu bỏ sót chu kỳ nhỏ ở một số đoạn và nhận đỉnh phụ ở đoạn khác. ECG chỉ là tham khảo, chưa có peak chuẩn độc lập, không dùng thay BVP. Thử có kiểm soát trên cùng toàn đoạn/mask, chỉ đổi prominence cho năm ví dụ trên, không cứu được bốn ví dụ thiếu. Không đủ căn cứ quy hết lỗi cho bộ lọc hay chỉ hạ ngưỡng. Nhiệt độ ngực S2 thiếu theo `suspect_contact`; biên độ SCR thiếu khi không có sự kiện, không điền 0.

Bằng chứng chẩn đoán ngoài Git: `outputs/g2_s2_mai_review/` (integration, bvp_diagnosis, bvp_windows, tests) và `outputs/g2_s2_bvp_visual_review/` (diagnosis, window_comparison, năm ảnh raw/filtered/peak/khoảng nhịp/ECG). Đây không phải yêu cầu đối chiếu nhiều máy cho quy trình bàn giao tiếp theo.

**Lô S5/S6/S10 trước lượt chung:** dùng nguồn `7720b01` và hợp đồng Như, hash `abd964e88fa0e5922c8bfc2cb71e1f4ad46ebb1d381e3f6bf82571817e1b452b`. Đã kiểm tra manifest, raw hash, nhãn, schema, checksum và ZIP; không đổi ngưỡng BVP hoặc impute.

| Subject | Ứng viên | Loại do nhãn | Giữ | Baseline / Stress / Amusement | Cửa sổ thiếu nhịp BVP | Ô feature thiếu |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| S5 | 1240 | 826 | 414 | 230 / 119 / 65 | 392 | 2472 |
| S6 | 1403 | 991 | 412 | 226 / 121 / 65 | 412 | 2664 |
| S10 | 1088 | 662 | 426 | 226 / 135 / 65 | 326 | 2286 |
| Tổng | 3731 | 2479 | 1252 | 682 / 375 / 195 | 1130 | 7422 |

Không loại thêm do biên lọc/nonfinite/thiếu mẫu. Build Python standalone khác conda-forge của Như; giữ thông tin producer thật và không ghép các chữ ký cũ thành bộ chung. Bản nguồn xuất Git và kiểu xuống dòng được lưu trong `SOURCE_RECONSTRUCTION.json` kèm gói.

Gói lô cũ ngoài Git: `data/features/g2_physiological68_S5_S6_S10_20261107/` và ZIP cùng tên, SHA-256 `dc0de86176b6fa53b58a6e9f0db5acb4aa471a0645861eb08341040383b386d4`. Báo cáo nhỏ: [S5_S6_S10_verification.json](../reports/g2_physiological68/S5_S6_S10_verification.json). Gói Như ban đầu có SHA-256 `909a4b2e153e152474f2dd5797946968cb21a1437a6f051c4f517baa8fcef28f`, đủ 4 subject/1.658 cửa sổ; hợp đồng kèm gói khớp hợp đồng checkout. Các lô lịch sử không được nối thêm vào bộ 15 người hiện tại vì sẽ lặp subject.

### Lịch sử triển khai trước đó

Các trạng thái provider chưa có và số liệu adapter tham chiếu ở phần dưới là lịch sử, không mô tả gói bàn giao hiện tại ở trên.

Cập nhật bước 1 ngày 07/11/2026: **đã tách tiền xử lý khỏi mã feature của Mai**. `features.py` không còn Butterworth/SOS, resampling hoặc đổi scale biên độ. Config để `preprocessing_provider: null`; khi chưa nhận provider của Như, extractor dừng trước khi đọc PKL, không dùng nhánh lọc mặc định.

Các hàm tính feature, detector làm việc trên tín hiệu đã chuẩn bị, schema và test của Mai vẫn giữ. `Channel.values` phải có đúng đơn vị schema (ACC cổ tay đã là g), kèm mask và thông tin đỉnh/sự kiện tương ứng. Đầu vào cùng tín hiệu dùng cho PSD hoặc detection phải được provider chuẩn bị đúng băng tần; không lọc lần nữa trong feature kernel.

Adapter tham chiếu cũ đã chuyển ra `outputs/g2/reference_preprocessing.py` ngoài Git, chỉ để truy vết thử nghiệm trước đó; không phải mã của Như hay provider mặc định. Bản sao tạm trước khi tách đã được dọn; mã trước khi tách có thể truy vết bằng commit `13dafa0`. Các gói S2/S5/S6/S10 dưới đây là **lịch sử chạy thử adapter cũ**, không phải kết quả của giao diện hiện tại khi chưa tích hợp Như. Không ghi nhận Như đã review hoặc nhóm đã nghiệm thu G2. Sau khi tách, 11/11 test feature và toàn bộ 50/50 test đạt; log tại `outputs/g2/tests_after_adapter_separation.txt`. Chưa chạy lại S2 hoặc lô mới.

## Trạng thái thực tế

| Nội dung phần Mai | Trạng thái |
| --- | --- |
| Lớp tính 68 feature, tên/thứ tự/đơn vị theo schema | Đã triển khai và kiểm tra công thức |
| Metadata, binary labels, NaN có lý do; imputation không chạy trong extractor | Đã kiểm tra |
| ACC/EMG đủ 12 đoạn 5 giây rồi lấy mean; nhận đầu vào đúng đơn vị | Kernel đã triển khai; provider của Như chịu trách nhiệm đổi scale, kernel không chia ACC thêm lần nữa |
| Schema thực tế sinh cùng package, QA trước/sau loại và checksum | Đã có; resume giữ nguyên contract và độ chính xác CSV |
| S2 và lô S5/S6/S10 | Kết quả thử adapter cũ được giữ nguyên; chưa chạy lại pipeline chính thức sau khi tách |
| Hợp nhất gói thành viên, không trùng S2, kiểm tra đủ 15 người | Công cụ và test đã có; mới ghép thử bốn người, chưa có gói sinh lý của 11 người còn lại |
| Chốt dữ liệu feature G2 để huấn luyện | **Chưa đạt**: cần xử lý/review BVP, tích hợp Như và đủ 15 người |

Bộ 112 raw G1 giữ nguyên. Không trộn với bộ sinh lý 68; bộ thử cũ 175 feature không được dùng trong lượt này. Dictionary vẫn giữ tên `docs/G1_DANH_MUC_FEATURE.csv` để dùng một danh mục chuẩn, trạng thái cột đã cập nhật thành implemented.

## Các file mã và giao diện

- `src/wesad/features.py`: các kernel `interval_features`, `band_powers`, `block_features`, `window_features`; processor `process_subject(path, config) -> DataFrame` dùng CLI hiện có.
- `src/wesad/feature_packages.py`: kiểm tra package/schema/NaN/checksum/QA; hợp nhất strict, nhận S2 trùng chỉ khi raw hash và các dòng giống nhau.
- `scripts/check_g2_features.py`: kiểm tra gói, hợp nhất, và cổng yêu cầu đủ 15 người. Mặc định merge thiếu người bị từ chối; `--allow-partial-merge` ghi rõ selected_subjects_only.
- `configs/features_g2.yaml`: bộ physiological_68_v1, gắn hash toàn bộ contract; không sửa config raw của nhóm.
- `src/wesad/extraction.py`: bổ sung truyền contract thực tế vào manifest/schema, khôi phục khi resume; đọc CSV round_trip để không làm thay đổi số khi ghép/cache.
- `tests/test_features_g2.py`: 11 test lớp feature, trong đó có không lọc lại/đổi scale lại, thiếu provider dừng trước PKL, gọi provider cấu hình đúng một lần. Công cụ hợp nhất và test package vẫn là phần làm trước để cập nhật sau; kiểm tra toàn bộ sau tách ghi trong `outputs/g2/tests_after_adapter_separation.txt`.

Điểm nối tiền xử lý của Như: config `preprocessing_provider: module:function`. Hàm nhận `(data, config, contract)` và trả dictionary đủ 10 kênh với key `chest_ecg`, `chest_eda`, `chest_emg`, `chest_resp`, `chest_temp`, `chest_acc`, `wrist_bvp`, `wrist_eda`, `wrist_temp`, `wrist_acc`. Mỗi giá trị là `Channel` trong `features.py`, gồm tín hiệu đã đúng đơn vị, native fs, mask covered/usable, peaks toàn bản ghi, tonic/phasic và SCR events `(onset_index,peak_index,amplitude)` khi cần. Chỉ số theo tần số riêng từng kênh. Đơn vị adapter phải đúng schema; không chia ACC hai lần hoặc áp lọc lần nữa trong kernel.

Không còn `wesad.features:prepare_subject` hoặc `filter_segments` trong mã được giao của Mai. Provider của Như có thể gọi các hàm detector feature dùng chung trên tín hiệu đã lọc; riêng phần lọc/calibration/mask do provider cung cấp. Chưa đặt tên module Như khi module đó chưa có. Thay provider/mã/config cần output mới; không resume các gói thử cũ sang pipeline mới.

## Lịch sử chạy thử trước khi tách adapter — chỉ gói tham chiếu

| Subject | Cửa sổ | Baseline | Stress | Amusement | Số cửa sổ thiếu sáu feature nhịp BVP |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 | 395 | 219 | 113 | 63 | 336 |
| S5 | 414 | 230 | 119 | 65 | 392 |
| S6 | 412 | 226 | 121 | 65 | 412 |
| S10 | 426 | 226 | 135 | 65 | 326 |

Lô Mai S5/S6/S10 có **1.252 cửa sổ**, 682 Baseline / 375 Stress / 195 Amusement. Ghép kiểm tra cục bộ thêm S2 được 1.647 cửa sổ/4 người, 68 feature + 8 metadata. Scope vẫn selected_subjects_only, không phải gói đủ 15 người.

- `data/features/g2_physiological_68_v1_s2_reference/`: S2, schema/config/manifest/QA/checksum/log.
- `data/features/g2_physiological_68_v1_reference/`: lô S5/S6/S10, cùng mã/config/môi trường với S2.
- `data/features/g2_physiological_68_v1_local4_reference/`: hợp nhất thử 4 người; chưa dùng làm bảng mô hình chính.
- `outputs/g2/S2_reference_check.json`, `assigned_reference_check.json`, `local4_reference_check.json`: kiểm tra package passed trong phạm vi tương ứng.
- `outputs/g2/full_group_gate.json`: **failed có chủ đích** khi yêu cầu đủ 15 người nhưng mới có 4; chứng minh không chốt nhầm full G2.
- `outputs/g2/tests.txt`: 49/49 test đạt. Lượt đầu sandbox có một PermissionError ghi file tạm trên Windows; chạy lại ngoài sandbox đạt toàn bộ.
- Resume S5/S6/S10 đều SKIP verified và giữ contract trong schema.

Lô 3 người chạy tuần tự mất khoảng 44.154 giây ở lượt đo này. RSS của cây tiến trình lấy mẫu mỗi 0,2 giây cao nhất khoảng 0.99 GiB; đây là số quan sát lấy mẫu, không phải bảo đảm peak tuyệt đối. Log đo: `outputs/g2/assigned_run_resources.json`.

## Điểm cần review trước khi chốt G2

**BVP là điểm chưa đạt về khả năng sử dụng dữ liệu**: lô Mai có 1.130/1.252 cửa sổ (90,3%) thiếu sáu feature nhịp/PRV; S6 thiếu toàn bộ 412 cửa sổ. Các giá trị được ghi NaN và flag insufficient_or_unreliable_beats đúng hợp đồng, không thay bằng 0 hoặc nới ngưỡng để đẹp báo cáo. Đây là chất lượng/độ phù hợp của detector cần review, không đủ căn cứ kết luận cảm biến cổ tay không có giá trị.

Đã kiểm tra chẩn đoán phạm vi tính prominence trên từng cửa sổ bằng một script ngoài mã dự án: số cửa sổ BVP đủ điều kiện lần lượt chỉ tăng S2 59→72, S5 22→43, S6 0→4, S10 100→123. **Không áp dụng thay đổi đó** vào code/config/CSV đã bàn giao, vì không giải quyết vấn đề chính và làm đổi kết quả thiết kế. Kết quả chẩn đoán ở `outputs/g2/bvp_threshold_scope_diagnostic.json`. Cần Như/Mai xem waveform và quy tắc đỉnh/độ tin cậy; nếu đổi detector, tăng phiên bản và chạy lại gói dùng chung.

S2 chest TEMP có cảnh báo tiếp xúc từ readme, nên hai feature nhiệt ngực đều NaN kèm suspect_contact. Không-SCR hợp lệ có count/rate=0, amplitude=NaN kèm no_scr_events. Những NaN có lý do này là khác với lỗi cột hoặc Inf. Không fit imputer/scaler trên toàn bảng; chưa chạy huấn luyện/G3.

## Lệnh kiểm tra bước 1 và lịch sử chạy thử

Kiểm tra riêng phần feature hiện tại:

```powershell
.\.venv-g1\Scripts\python.exe -m unittest discover -s tests -p test_features_g2.py -v
```

Đợi provider Như rồi mới cấu hình `preprocessing_provider` và chạy lại S2. Không chạy raw với config null hoặc đặt lại adapter tham chiếu để coi là đã tích hợp.

**Lịch sử:** các lệnh sau đã chạy trước khi tách adapter; hiện không chạy lại nguyên trạng vì config đã bỏ provider mặc định. Dùng `.venv-g1` Python 3.12.14; local.yaml chỉ chứa data_root:

```powershell
.\.venv-g1\Scripts\python.exe scripts/extract_features.py --config configs/features_g2.yaml --local configs/local.yaml --subjects S2 --output-dir data/features/g2_physiological_68_v1_s2_reference
.\.venv-g1\Scripts\python.exe scripts/extract_features.py --config configs/features_g2.yaml --local configs/local.yaml --subjects S5 S6 S10 --resume
.\.venv-g1\Scripts\python.exe scripts/check_g2_features.py --features-dir data/features/g2_physiological_68_v1_reference --report outputs/g2/assigned_reference_check.json
.\.venv-g1\Scripts\python.exe scripts/check_g2_features.py --features-dir data/features/g2_physiological_68_v1_s2_reference data/features/g2_physiological_68_v1_reference --merge-output data/features/g2_physiological_68_v1_local4_reference --allow-partial-merge --report outputs/g2/local4_reference_check.json
```

Nếu output đã tồn tại, extractor dùng --resume sau kiểm tra chữ ký; merge yêu cầu output mới/rỗng. Đổi mã/config/môi trường cần output mới; không sửa manifest để ép chạy tiếp.

## Bàn giao để nhóm tiếp tục

1. Mai gửi lớp feature/schema và giao diện đầu vào cho Như; Như cung cấp provider lọc/calibration/mask. Gói tham chiếu cũ chỉ phục vụ chẩn đoán BVP, không xác nhận Như đã làm phần đó.
2. Khi thống nhất adapter/detector, khóa cùng phiên bản mã/config/môi trường và kiểm tra lại S2 trước khi cả bốn chạy lô chung.
3. Nhận các gói sinh lý còn thiếu S3/S4/S7/S8/S9/S11/S13/S14/S15/S16/S17 để hợp nhất đủ 15 người. Không lấy bộ raw112 hoặc dữ liệu khác chữ ký để lấp thiếu.
4. Dùng cổng --require-all-subjects hoặc merge mặc định để kiểm tra đủ 15 người, duy nhất khóa/cột/phiên bản/NaN có lý do và quality report trước khi ghi ready_for_training.

Mã hiện trên workspace chưa commit, snapshot ghi git_dirty=true và base commit a9eede5. Code SHA-256 của các gói tham chiếu hiện tại: `445d062c95442f800f63426a6850bdcb4f5cb1c3f478adb12696334ecc11cce0`. Hash contract gắn trong config; schema hiện ghi lớp feature đã triển khai, `ready_for_extraction=false` do provider chưa cấu hình, `ready_for_training=false` và chưa xác nhận tích hợp runtime Như. Không ghi phần G2 của cả nhóm hoàn tất.
