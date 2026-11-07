# G2 phần Như — bản cơ bản ngày 07/11/2026

Phạm vi theo phân công: lọc và đồng bộ theo thời gian, cửa sổ 60/5 với nhãn đa số 90%, kiểm tra ACC/EMG theo đoạn 5 giây; chạy lô S4, S9, S11, S17. Có chạy thêm S2 để kiểm tra. Chưa làm RF/MLP, tuning hoặc biểu đồ mô hình của các giai đoạn sau.

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

Bảng chẩn đoán không phải bộ feature huấn luyện hoàn chỉnh và không dùng thay bảng 112. Bộ 112 hiện tại vẫn là raw, chưa áp bộ lọc hay cách tổng hợp 5 giây mới. Muốn tích hợp các bước này vào bộ feature chung, Như và Mai cần thống nhất phiên bản/schema mới và chạy đồng bộ các lô; không ghép lẫn hai định nghĩa vào cùng bảng.

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

Bản cơ bản phần Như đã có mã và lần chạy lô thực tế. Chưa kết luận G2 cả nhóm hoàn tất: Mai còn cần tích hợp theo phiên bản feature thống nhất và kiểm tra đủ 15 subject; các thành viên khác vẫn xác nhận lô của mình. Không lấy bộ 112 raw hoặc 72 chỉ số chẩn đoán làm bằng chứng đã triển khai trọn bộ feature sinh lý của proposal.

Các cải tiến để làm sau, không cố tình để lỗi trong chức năng đã có:

- TODO: kiểm tra phẳng, bão hòa, tiếp xúc kém và nhiễu chuyển động; hiện chỉ kiểm tra đủ mẫu, hữu hạn và vùng biên lọc.
- TODO: vẽ tín hiệu trước/sau lọc để rà bằng mắt; đánh giá độ nhạy cutoff/vùng biên trước khi quyết định phiên bản cuối.
- TODO: thống nhất đơn vị, bộ lọc ECG/BVP/EMG và ngưỡng chất lượng với Mai trước extractor sinh lý; hiện không tự thêm thuật toán chưa chốt.
- TODO: đưa resume, schema chẩn đoán và log thống nhất vào CLI chung khi Thái/Mai tích hợp. Bản này ưu tiên chạy tuần tự và đầu ra đơn giản.

Không fit scaler, imputer hoặc bộ chọn đặc trưng ở giai đoạn tiền xử lý này. Các bước đó thuộc train fold của thực nghiệm sau.
