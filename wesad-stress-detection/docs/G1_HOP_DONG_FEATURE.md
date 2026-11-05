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

Mean/std đều dùng ddof=0 cho thống kê tín hiệu/rate; SDNN/SD IBI dùng ddof=1. pNN50 là tỷ lệ 0–1. Tích phân tuyệt đối đoạn 5 s = sum(abs(x))/fs; đơn vị g·s với ACC hoặc mV·s với EMG. Magnitude ACC = sqrt(x²+y²+z²). Theo thiết kế Như `nhu_g1_design_v2`, ACC giữ trọng lực; không tự trừ mean hoặc chuyển scale.

Năng lượng dải tần được định nghĩa là tích phân PSD theo tần số, đơn vị bình phương đơn vị tín hiệu. Không gán tên LF/HF-HRV cho phổ của ECG/BVP thô. Các cutoff EDA/RESP đã lấy theo thiết kế G1 của Như bên dưới. Phương pháp PSD, biên từng dải và thuật toán phát hiện nhịp/SCR chưa được quy định đầy đủ, cần khóa trước khi triển khai G2. Các mục này được đánh dấu pending trong schema; không dùng cấu hình chưa khóa để chạy bảng chính.

Đơn vị đầu ra mục tiêu: ECG/EMG mV, EDA µS, TEMP °C, ACC g, BVP arbitrary unit; rate bpm hoặc breaths/min, khoảng nhịp ms. Phải xác minh đơn vị/scale đầu vào từ tài liệu raw trước khi xuất với các đơn vị đó; không mặc định dữ liệu đã đúng đơn vị. Quy tắc chuyển đổi ACC cổ tay và hệ số các kênh nằm trong danh sách cần xác minh, không tự áp dụng hệ số khi chưa có căn cứ.

## Tiền xử lý dùng theo thiết kế G1 của Như

Phần Mai chốt dùng [thiết kế G1 của Như](G1_NHU_THIET_KE_TIEN_XU_LY.md) làm đầu vào theo yêu cầu người dùng; không xây một nhánh tiền xử lý riêng. Đây là thiết kế để triển khai/kiểm tra G2, chưa tuyên bố bộ lọc đã được nghiệm thu.

- Đồng bộ theo PKL, giữ tần số gốc từng kênh; cắt khoảng `[start_s, end_s)` trên cùng lưới nhãn 700 Hz, không nâng mọi kênh lên 700 Hz.
- EDA ngực low-pass 5 Hz; RESP ngực band-pass 0,1–0,35 Hz; EDA cổ tay chọn low-pass 1 Hz theo thiết kế Như, kiểm tra tín hiệu ở G2.
- Theo đề xuất triển khai của Như: Butterworth bậc thiết kế 4, SOS, lọc hai chiều offline trên từng đoạn hữu hạn liên tục trước khi cắt cửa sổ. Không lọc xuyên NaN/Inf, không nối đoạn theo nhãn.
- Theo `configs/preprocessing_g1_design.yaml` revision `nhu_g1_design_v2`: bỏ biên mỗi đầu đoạn hữu hạn 10 giây với EDA ngực/cổ tay, 30 giây với RESP; odd padding; đoạn quá ngắn không sử dụng. Đây là thiết kế, chưa nghiệm thu tác động lọc trên tín hiệu thật.
- ACC giữ trọng lực, TEMP giữ tín hiệu gốc cho thống kê ban đầu; không tự chuyển scale. ACC/EMG tính feature trên 12 đoạn 5 giây rồi lấy trung bình.
- Mai phụ trách feature/schema từ tín hiệu Như bàn giao. Detector ECG/BVP/SCR, decomposition EDA, dải PSD, scale/đơn vị và ngưỡng QA sinh lý chưa được quy định đầy đủ vẫn ghi rõ còn mở; không tự coi là đã chốt.
- Bộ thử `physio_baseline_v1` lọc riêng trong từng cửa sổ không phải thiết kế thống nhất này; không lấy làm bộ cuối hoặc ghép với đầu ra theo Như.

## Thiếu dữ liệu và chất lượng

- CSV dùng ô trống để biểu diễn NaN ở feature không đủ dữ liệu; không dùng 0, -999 hoặc chuỗi trong cột feature để thay thiếu.
- Nhịp không đủ tin cậy: rate/HRV/PRV liên quan là NaN, gắn cờ kèm tên thiết bị/cảm biến. Ngưỡng độ tin cậy và số khoảng tối thiểu được khóa cùng detector với Như trước G2.
- SCR count = 0 chỉ khi EDA đạt chất lượng và detector chạy thành công nhưng không có SCR; lỗi detector hoặc tín hiệu không tin cậy phải là NaN.
- Inf không hợp lệ, phải điều tra; không tự đổi thành số hữu hạn. Không được thiếu metadata bắt buộc.
- quality_flag là chuỗi các mã cách nhau bằng dấu `;`, `ok` nếu các kiểm tra đã khóa đều đạt; ví dụ `wrist_bvp:insufficient_beats`, `chest_eda:nonfinite`, `chest_temp:suspect_contact`. Không dùng mã ok khi chưa chạy QA.
- Theo QA Như bàn giao: dùng chung khóa cửa sổ giữa các kênh yêu cầu; loại và ghi lý do nếu cửa sổ thiếu mẫu, không đủ độ dài, có NaN/Inf ở kênh yêu cầu hoặc chồng vùng biên lọc. Không chọn lại mẫu theo từng nhóm cảm biến. NaN feature do detector không tin cậy là trường hợp riêng, được ghi lý do và xử lý trong fold; không đồng nhất với thiếu mẫu tín hiệu đầu vào. Tín hiệu phẳng/tiếp xúc nghi ngờ được gắn cờ để xem xét, không tự coi là đã đạt QA sinh lý.
- Imputer, scaler, selection và resampling chỉ học trên train của từng fold; không điền NaN bằng thống kê toàn bảng trong bước trích feature.

## Điểm phối hợp và điều kiện khóa schema

| Người | Mai gửi | Mai cần nhận trước khi khóa G2 |
| --- | --- | --- |
| Thái | Schema, tên/thứ tự cột, quy ước NaN/nhãn, giao diện processor | Sử dụng loader/CLI, metadata, nhãn và giao diện processor Thái đã bàn giao |
| Như | Danh mục feature, đơn vị mục tiêu, yêu cầu nhịp/SCR/ACC/EMG | Dùng thiết kế G1 đã bàn giao; tích hợp feature theo thiết kế đó, ghi rõ các tham số chưa được quy định |
| Nam | Nhãn và nhóm cột để EDA/mô hình dùng chung | Các bất thường nhãn/tín hiệu/giá trị thiếu đã khảo sát |

Mai làm danh mục/schema song song, không cần chờ Như hoàn tất code. Chỉ khóa phiên bản và nghiệm thu bảng S2 sau khi các điểm pending đã được thống nhất. Không coi một người đã đồng ý chỉ vì file đề xuất đã được tạo.

Nghiệm thu S2 trên bốn máy: cùng khóa cửa sổ, nhãn, start/end, cột/thứ tự, đơn vị và config; với gói 21 thống kê raw G1, dùng Python 3.12.14 và thư viện khóa, so sánh feature theo rtol=1e-7, atol=1e-9 như Như bàn giao. Dung sai này không tự áp dụng cho 68 feature sinh lý. Không áp số cửa sổ của quy tắc homogeneous cho majority. Chưa có đối chiếu bốn máy ở thời điểm lập G1 này.

## Cập nhật từ EDA của Nam ngày 05/11/2026

Đã nhận kết quả tại `notebooks/reports/tables/eda_step5/eda_summary.json` và `eda_observations.md`: 15 người, 6.196 cửa sổ 60/5, 21 feature raw; 3.377 Baseline, 1.848 Stress, 971 Amusement; không có NaN/Inf feature hoặc khóa trùng trong bảng Nam khảo sát. Kiểm tra raw của Nam gồm ECG/EDA/ACC ở 15 người, không báo NaN/Inf hoặc lệch độ dài trong 45 cặp người–tín hiệu.

Phần Mai sử dụng các kết quả này làm đầu vào thiết kế: giữ nhãn nhị phân 2→1, 1/3→0; giữ QA sinh lý và cách ghi thiếu cho detector; cờ IQR chỉ phục vụ xem xét, chưa tự loại cửa sổ. Không suy rộng kiểm tra raw ba kênh sang chất lượng cổ tay/EMG/TEMP/RESP, đơn vị vật lý hoặc giá trị 68 feature dự kiến. Việc chọn feature vẫn học trong train của từng fold.

Danh mục, đơn vị mục tiêu, nhãn, cách ghi thiếu và schema nháp G1 đã được chuẩn bị, kiểm tra nhất quán. Cập nhật sau khi cài Python 3.12.14: kiểm tra một gói S2 trên máy này đã đạt (395 cửa sổ, 21 feature raw); đối chiếu bốn máy còn chưa hoàn tất; các tham số detector/PSD/scale chưa khóa vẫn được giữ trong `pending_decisions`.
