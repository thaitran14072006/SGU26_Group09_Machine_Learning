# G2 phần Như — bản cơ bản ngày 07/11/2026

Phạm vi theo phân công: lọc và đồng bộ theo thời gian, cửa sổ 60/5 với nhãn đa số 90%, kiểm tra ACC/EMG theo đoạn 5 giây; chạy lô S4, S9, S11, S17. Có chạy thêm S2 để kiểm tra. Chưa làm RF/MLP, tuning hoặc biểu đồ mô hình của các giai đoạn sau.

## Cập nhật tiếp theo sau phần Mai — 07/11/2026

Đã đọc main tới `f24cafc`: Mai tổng hợp đủ 15 người raw112 theo [báo cáo chung](G2_TONG_HOP_RAW112.md). Đây là kết quả Mai ghi nhận; lần cập nhật này không nhận lại gói 15 người hoặc chạy lại các lô raw để xác nhận độc lập.

Mai cũng tách lớp feature sinh lý khỏi tiền xử lý ở `08f0861`; cấu hình sinh lý đang chờ `preprocessing_provider`. Đã bổ sung `src/wesad/preprocessing_physiological.py`, hàm `prepare_subject(data, config, contract)` để trả về đủ 10 `Channel` theo giao diện Mai. Cấu hình `features_g2.yaml` trỏ rõ đến provider này; không đổi mặc định CLI `features_smoke.yaml` hoặc định nghĩa bộ raw112.

Provider dùng các phương pháp đã ghi trong hợp đồng sinh lý của Mai: ECG lọc 5–40 Hz, nhánh phát hiện nhịp 5–20 Hz; BVP 0.5–5 Hz; EMG 20–250 Hz; EDA và RESP theo thiết kế, EDA tonic low-pass 0.05 Hz và phasic là cleaned trừ tonic. Tận dụng helper phát hiện nhịp/SCR của Mai để không sao chép thuật toán. Bộ lọc thực hiện ở provider; lớp feature không lọc lần hai.

Các kênh giữ lưới thời gian/tần số gốc. Mask `covered` ghi mẫu đầu vào hữu hạn; `usable` loại vùng biên lọc, đoạn ngắn và phần lọc không hữu hạn. Với EDA, vùng biên tonic 60 giây được tính trên đoạn raw hữu hạn, không cộng lặp với vùng biên cleaned 10 giây. Wrist ACC chia 64 một lần trên bản sao theo hợp đồng đơn vị PKL của Mai; chest ACC giữ nguyên. Bộ raw112 và bảng chẩn đoán cũ vẫn giữ thang đo/công thức ban đầu.

### Lượt thử tích hợp S2

- Chạy thật qua `scripts/extract_features.py` với provider mới: **395 cửa sổ, 68 feature (37 ngực + 31 cổ tay) và 8 metadata**.
- 1.204 ứng viên, loại 809 do nhãn; không loại thêm do thiếu mẫu, nonfinite hoặc biên lọc. Nhãn 219/113/63; bảy metadata đầu khớp gói S2 raw112 cũ.
- Header/thứ tự cột đúng hợp đồng, không trùng khóa, không Inf; checksum và tổng số cửa sổ/lý do loại đúng.
- Có **3.046 giá trị NaN**, đều có cờ lý do thuộc cảm biến tương ứng. Hai feature nhiệt độ ngực S2 thiếu cả 395 cửa sổ vì `suspect_contact`; sáu feature nhịp/IBI của BVP thiếu ở **336/395 cửa sổ** vì chưa đủ nhịp đáng tin theo heuristic của Mai. Biên độ SCR thiếu khi không có SCR hợp lệ: ngực 40, cổ tay 80 cửa sổ. Không điền 0 hoặc hạ ngưỡng để ép đủ số.
- Đây là **đạt kiểm tra tích hợp/cấu trúc**, chưa xác nhận chất lượng sinh lý hoặc đủ điều kiện dùng bộ 68 cho mô hình. Tỷ lệ thiếu BVP cần Như/Mai rà detector và tín hiệu trước khi nghiệm thu nhánh này; thay thuật toán cần xem lại phiên bản/hợp đồng.
- Thêm 6 test provider, toàn repo **68/68 test đạt**, gồm kiểm tra đơn vị ACC, vùng biên, không nối qua gap và nối thực sự với extractor. Các số 43 test ở phần bản đầu bên dưới là kết quả lịch sử.

Gói S2 mới nằm ngoài Git tại `data/features/g2_physiological68_S2_provider_basic_20261107/`. Bằng chứng nhỏ: [S2_provider_basic_verification.json](../reports/g2_preprocessing_nhu/S2_provider_basic_verification.json). Giữ nguyên manifest và báo cáo các lượt cũ.

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/extract_features.py --config configs/features_g2.yaml --local configs/local.yaml --subjects S2 --output-dir data/features/g2_physiological68_S2_provider_lan2 --resume
```

Đợt này dừng ở provider cơ bản và S2. Bước tiếp theo là bàn giao Mai rà giao diện và tỷ lệ thiếu BVP; sau đó mới quyết định chạy lô sinh lý S4/S9/S11/S17 hoặc mở rộng 15 người. Chưa có nghiệm thu nhiều máy cho bộ 68, chưa chạy G3. Các phần bên dưới mô tả bản đầu và bộ raw112 đã bàn giao.

## Đã có

- `src/wesad/preprocessing_g2.py`: API `prepare_signals`, `iter_preprocessed_windows`, `acc_emg_block_statistics` để bàn giao tích hợp với extractor của Mai.
- `configs/preprocessing_g2_basic.yaml`: bộ lọc đã thiết kế trong G1, tham số và lô Như.
- `scripts/check_g2_preprocessing.py`: chạy tuần tự từng PKL, ghi QA, chẩn đoán ACC/EMG và thông tin nguồn chạy; báo failed nếu lỗi, giữ bằng chứng đã ghi.
- `tests/test_preprocessing_g2.py`: 8 bài kiểm tra mới; toàn bộ repo đạt 43/43 test trong Python 3.12.14.
- Lô 112 raw của Như đã trích xuất bằng extractor chung, không đổi công thức bộ nhóm đang dùng.

## Cách xử lý cơ bản

EDA ngực low-pass 5 Hz, EDA cổ tay low-pass 1 Hz; RESP ngực band-pass 0.1–0.35 Hz. Dùng Butterworth bậc thiết kế 4, SOS forward/backward và odd padding. Với band-pass, bậc bộ lọc kết quả cao hơn bậc thiết kế; forward/backward lọc hai lượt. Đây là xử lý offline, dùng cả mẫu phía sau, chưa phải bộ lọc thời gian thực.

Lọc từng đoạn hữu hạn liên tục trong mỗi subject. Không nối qua NaN/Inf; đoạn không đủ padding hoặc không còn phần trong bị đánh dấu không dùng. Bỏ 10 giây mỗi đầu đoạn EDA và 30 giây mỗi đầu đoạn RESP theo thiết kế G1. Thông số vùng biên là lựa chọn thiết kế cơ bản, chưa được tối ưu theo độ ổn định sinh lý.

Giữ tần số gốc 700/64/32/4 Hz; cắt theo cùng khoảng nửa mở [start, end), không nâng các kênh lên 700 Hz. Tính nhãn trên toàn bộ cửa sổ gốc, kể cả nhãn 0/4 trong mẫu số. Cửa sổ được giữ khi tất cả kênh yêu cầu đủ mẫu, hữu hạn và không chạm vùng lọc không dùng. QA đếm lý do đầu tiên khiến cửa sổ bị loại theo thứ tự kênh; các lý do là loại trừ nhau.

ACC giữ x/y/z và magnitude, giữ trọng lực và thang đo PKL; EMG giữ thang đo PKL. Mỗi cửa sổ chia đúng 12 đoạn 5 giây. Mỗi đoạn tính mean, std dân số, min, max, RMS, IQR, range và tích phân tuyệt đối `sum(abs(x))/fs`; lấy trung bình từng chỉ số qua 12 đoạn. Tổng 72 cột chẩn đoán: 8 kênh ACC + 1 EMG, mỗi kênh 8 chỉ số. Hậu tố `_block5_mean` phân biệt với thống kê trên cả cửa sổ. Đơn vị IAV là đơn vị ghi trong PKL nhân giây; chưa tự đổi sang đơn vị vật lý khác.

## Hai loại đầu ra

| Gói | Mục đích |
| --- | --- |
| `data/features/g2_nhu_raw112_20261107/` | Lô 112 raw theo định nghĩa nhóm hiện tại, 8 metadata + 112 feature; gửi cho Mai khi ghép dữ liệu |
| `outputs/g2_preprocessing_basic_20261107/` | QA có lọc và bảng ACC/EMG chẩn đoán, 8 metadata + 72 chỉ số; kiểm tra phần Như và bàn giao API |

Bảng chẩn đoán không phải bộ feature huấn luyện hoàn chỉnh và không dùng thay bảng 112. Bộ 112 hiện tại vẫn là raw, chưa áp bộ lọc hay cách tổng hợp 5 giây mới. Nhánh tích hợp sinh lý68 thử S2 được ghi riêng ở đầu tài liệu; chưa chạy đồng bộ các lô hoặc đổi bộ feature chung. Không ghép lẫn các định nghĩa vào cùng bảng.

## Kết quả thật đã chạy

| Subject | Ứng viên | Loại do nhãn | Giữ sau lọc | Baseline | Stress | Amusement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S4 | 1273 | 869 | 404 | 222 | 117 | 65 |
| S9 | 1033 | 622 | 411 | 227 | 119 | 65 |
| S11 | 1035 | 618 | 417 | 227 | 126 | 64 |
| S17 | 1173 | 747 | 426 | 226 | 135 | 65 |
| Tổng lô Như | 4514 | 2856 | 1658 | 902 | 497 | 259 |

S2 chạy thêm: 1204 ứng viên, loại 809 do nhãn, giữ 395 (219/113/63). Trên cả 5 subject đã thử, không loại thêm cửa sổ đạt nhãn do thiếu mẫu, NaN/Inf hoặc vùng biên lọc. Điều này không chứng minh tín hiệu đạt chất lượng sinh lý; các test tổng hợp đã kiểm tra các trường hợp bị loại.

Đã kiểm tra các cửa sổ và 7 metadata đầu khớp giữa gói raw và gói chẩn đoán cho lô 4 người; không trùng khóa, giá trị hữu hạn, checksum CSV raw đúng manifest, tổng số ứng viên bằng giữ cộng các lý do loại. Thời gian ghi nhận gồm đọc/tính/xuất/hash theo từng subject: audit khoảng 8.2–8.7 giây, raw khoảng 5.5–6.3 giây trong lần chạy này. Hai tiến trình đã chạy đồng thời trong một khoảng thời gian, nên đây không phải benchmark.

Bảng QA và bằng chứng nhỏ được giữ tại [quality_report.csv](../reports/g2_preprocessing_nhu/quality_report.csv) và [verification.json](../reports/g2_preprocessing_nhu/verification.json). Source đang có thay đổi chưa commit nên metadata ghi `git_dirty=true`; các tệp snapshot và hash được giữ đúng thực tế.

## Lệnh chạy lại

Chạy tại `wesad-stress-detection`. `configs/local.yaml` chỉ chứa đường dẫn raw riêng. Dùng output mới để giữ bằng chứng lần cũ; CLI audit cơ bản chưa hỗ trợ resume.

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv-g1\Scripts\python.exe -X utf8 scripts/check_g2_preprocessing.py --local configs/local.yaml --subjects S2 S4 S9 S11 S17 --output-dir outputs/g2_preprocessing_basic_lan2
.\.venv-g1\Scripts\python.exe -X utf8 scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S4 S9 S11 S17 --output-dir data/features/g2_nhu_raw112_lan2 --resume
```

## Bàn giao và phần để phát triển tiếp

Bản cơ bản phần Như đã có mã và lần chạy lô thực tế. Mai đã báo cáo đủ 15 người raw112 ở `f24cafc`. Nhánh sinh lý68 mới nối provider và thử S2, còn cần rà kết quả trước khi mở rộng; chưa kết luận G2 sinh lý cả nhóm hoàn tất. Không lấy bộ 112 raw hoặc 72 chỉ số chẩn đoán làm bằng chứng đã nghiệm thu trọn bộ feature sinh lý của proposal.

Các cải tiến để làm sau, không cố tình để lỗi trong chức năng đã có:

- TODO: kiểm tra phẳng, bão hòa, tiếp xúc kém và nhiễu chuyển động; hiện chỉ kiểm tra đủ mẫu, hữu hạn và vùng biên lọc.
- TODO: vẽ tín hiệu trước/sau lọc để rà bằng mắt; đánh giá độ nhạy cutoff/vùng biên trước khi quyết định phiên bản cuối.
- TODO: Mai/Như rà provider theo hợp đồng hiện có, nhất là tỷ lệ thiếu BVP của S2; xác nhận chất lượng trước lượt sinh lý tiếp theo.
- TODO: đưa resume, schema chẩn đoán và log thống nhất vào CLI chung khi Thái/Mai tích hợp. Bản này ưu tiên chạy tuần tự và đầu ra đơn giản.

Không fit scaler, imputer hoặc bộ chọn đặc trưng ở giai đoạn tiền xử lý này. Các bước đó thuộc train fold của thực nghiệm sau.
