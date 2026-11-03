# Giai đoạn 1 Mai Danh mục và hợp đồng feature

Ngày chuẩn bị: 03/11/2026. Phạm vi theo G1 trong PHAN_CONG_MAY_HOC.docx: danh mục feature, đơn vị, nhãn nhị phân, cách đánh dấu thiếu và schema đầu ra nháp. Tài liệu này không nghiệm thu extractor hoặc kết quả huấn luyện của G2/G3.

## File đầu ra đúng phân công

| File | Mục đích | Giai đoạn |
| --- | --- | --- |
| `configs/feature_schema.json` | Schema nháp, danh sách/thứ tự cột, kiểu dữ liệu, đơn vị và quy ước thiếu | G1 |
| `docs/G1_DANH_MUC_FEATURE.csv` | Danh mục feature đọc được bằng bảng tính, cùng tên và thứ tự với schema | G1 |
| `docs/G1_HOP_DONG_FEATURE.md` | Định nghĩa, cách phối hợp và điều kiện nghiệm thu | G1 |
| `docs/BAN_GIAO.md` | Bàn giao riêng G1 | G1 |
| `src/wesad/features.py` | `process_subject(path, config) -> DataFrame`; triển khai theo hợp đồng sau khi thống nhất tiền xử lý | G2 |
| `src/wesad/baselines.py` | Dummy và Logistic Regression | G3 |
| `src/wesad/metrics.py` | Metric theo người và tổng hợp | G3 |

Tên module baselines.py/metrics.py là điểm bàn giao đề xuất trong phân công. G1 chưa cần tạo module rỗng hoặc chuyển mã đang chạy từ models.py/evaluation.py. Bộ `physio_baseline_v1` đã thử trước đây là thử nghiệm riêng, không phải schema G1 này và chưa đủ các yêu cầu feature của bản phân công.

## Quy ước chung đã ghi nhận từ phân công

- Đối tượng: S2–S11, S13–S17. Seed 42.
- Cửa sổ 60 giây, bước 5 giây, trên thời gian gốc. Không nối các đoạn sau khi bỏ nhãn.
- Nhãn đa số ≥0,90 trên toàn bộ cửa sổ, kể cả nhãn không dùng trong mẫu số. Chỉ nhận nhãn 1/2/3; bỏ cửa sổ không đạt.
- `label_original`: 1 Baseline, 2 Stress, 3 Amusement. `label_binary`: 2→1, 1/3→0. Nhãn 0/4/khác không được ánh xạ thành non-stress.
- Metadata đứng trước, feature theo đúng thứ tự `feature_columns` trong schema. X chỉ gồm feature; không dùng mã người, thời gian, nhãn, độ thuần nhãn hoặc quality_flag.
- Khóa duy nhất `(subject_id, window_id)`; window_id = `<subject_id>_<start_index_700hz>`. start_s/end_s tính bằng giây từ đầu PKL đồng bộ; end_s − start_s = 60.
- Prefix feature: `chest_<sensor>_...`, `wrist_<sensor>_...`; nhóm combined là chest nối wrist theo thứ tự schema. Các so sánh dùng cùng khóa cửa sổ đủ điều kiện.

## Danh mục feature G1

Danh sách cụ thể và đơn vị nằm trong schema và CSV. Đây là danh mục dự kiến để Mai/Như đối chiếu và khóa ở đầu G2, không phải tuyên bố các cột đã được trích xuất.

| Nhóm | Danh mục |
| --- | --- |
| ECG ngực | HR mean/std, mean NN, SDNN, RMSSD, pNN50; năng lượng tín hiệu theo ba dải tần |
| BVP cổ tay | Pulse rate mean/std; mean IBI, SD IBI, RMSSD IBI, pNN50 IBI; năng lượng theo ba dải tần. Tên PRV tách biệt HRV ECG |
| EDA ngực/cổ tay | Mean/std tonic và phasic; SCR count, rate/min, mean/max amplitude tính từ onset đã xác định |
| RESP ngực | Nhịp thở mean/std |
| TEMP ngực/cổ tay | Mean và slope |
| ACC ngực/cổ tay | Mean, std, tích phân tuyệt đối cho x/y/z và magnitude; tính từng đoạn 5 giây, lấy trung bình qua 12 đoạn |
| EMG ngực | RMS, MAV, waveform length per second, tích phân tuyệt đối; tính từng đoạn 5 giây, lấy trung bình qua 12 đoạn |

Mean/std đều dùng ddof=0 cho thống kê tín hiệu/rate; SDNN/SD IBI dùng ddof=1. pNN50 là tỷ lệ 0–1. Tích phân tuyệt đối đoạn 5 s = sum(abs(x))/fs; đơn vị g·s với ACC hoặc mV·s với EMG. Magnitude ACC = sqrt(x²+y²+z²). Có loại trọng lực hay không phải được Như chốt, ghi vào processing config trước khi chạy.

Năng lượng dải tần được định nghĩa là tích phân PSD theo tần số, đơn vị bình phương đơn vị tín hiệu. Không gán tên LF/HF-HRV cho phổ của ECG/BVP thô. Phương pháp PSD, biên từng dải, cutoff bộ lọc và thuật toán phát hiện nhịp/SCR còn chờ thống nhất Như và đối chiếu proposal gốc. Các mục này được đánh dấu pending trong schema; không dùng cấu hình chưa khóa để chạy bảng chính.

Đơn vị đầu ra mục tiêu: ECG/EMG mV, EDA µS, TEMP °C, ACC g, BVP arbitrary unit; rate bpm hoặc breaths/min, khoảng nhịp ms. Phải xác minh đơn vị/scale đầu vào từ tài liệu raw trước khi xuất với các đơn vị đó; không mặc định dữ liệu đã đúng đơn vị. Quy tắc chuyển đổi ACC cổ tay và hệ số các kênh nằm trong danh sách cần xác minh, không tự áp dụng hệ số khi chưa có căn cứ.

## Thiếu dữ liệu và chất lượng

- CSV dùng ô trống để biểu diễn NaN ở feature không đủ dữ liệu; không dùng 0, -999 hoặc chuỗi trong cột feature để thay thiếu.
- Nhịp không đủ tin cậy: rate/HRV/PRV liên quan là NaN, gắn cờ kèm tên thiết bị/cảm biến. Ngưỡng độ tin cậy và số khoảng tối thiểu được khóa cùng detector với Như trước G2.
- SCR count = 0 chỉ khi EDA đạt chất lượng và detector chạy thành công nhưng không có SCR; lỗi detector hoặc tín hiệu không tin cậy phải là NaN.
- Inf không hợp lệ, phải điều tra; không tự đổi thành số hữu hạn. Không được thiếu metadata bắt buộc.
- quality_flag là chuỗi các mã cách nhau bằng dấu `;`, `ok` nếu các kiểm tra đã khóa đều đạt; ví dụ `wrist_bvp:insufficient_beats`, `chest_eda:nonfinite`, `chest_temp:suspect_contact`. Không dùng mã ok khi chưa chạy QA.
- Báo cáo chất lượng G2 ghi số cửa sổ trước/sau loại theo người và lý do; phải chốt tập cửa sổ chung ngực/cổ tay/kết hợp. G1 chưa tự chọn chính sách loại chỉ vì NaN ở một cảm biến.
- Imputer, scaler, selection và resampling chỉ học trên train của từng fold; không điền NaN bằng thống kê toàn bảng trong bước trích feature.

## Điểm phối hợp và điều kiện khóa schema

| Người | Mai gửi | Mai cần nhận trước khi khóa G2 |
| --- | --- | --- |
| Thái | Schema, tên/thứ tự cột, quy ước NaN/nhãn, giao diện processor | Cấu trúc PKL đã kiểm tra, loader/CLI, đường dẫn schema và hợp đồng kiểm tra bảng |
| Như | Danh mục feature, đơn vị mục tiêu, yêu cầu nhịp/SCR/ACC/EMG | Đồng bộ, scale, bộ lọc, detector, dải tần, xử lý biên và QA đã thống nhất |
| Nam | Nhãn và nhóm cột để EDA/mô hình dùng chung | Các bất thường nhãn/tín hiệu/giá trị thiếu đã khảo sát |

Mai làm danh mục/schema song song, không cần chờ Như hoàn tất code. Chỉ khóa phiên bản và nghiệm thu bảng S2 sau khi các điểm pending đã được thống nhất. Không coi một người đã đồng ý chỉ vì file đề xuất đã được tạo.

Nghiệm thu S2 trên bốn máy: cùng khóa cửa sổ, nhãn, start/end, cột/thứ tự, đơn vị và config; so sánh feature bằng dung sai theo từng nhóm được nhóm thống nhất. Không áp số cửa sổ của quy tắc homogeneous cho majority. Chưa có đối chiếu bốn máy ở thời điểm lập G1 này.
