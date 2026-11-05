> Cập nhật theo quyết định nhóm: G1 hiện nghiệm thu bộ 112 thống kê raw, 63 ngực + 49 cổ tay, phiên bản all_sensors_raw_stats_v1. Gói 21 trước đây chỉ là lịch sử chạy mẫu.

# Thiết kế tiền xử lý G1

Người phụ trách: Phạm Quỳnh Như. Cập nhật ngày 05/11/2026 theo README Nam đưa vào main tại `e6415a4` và kết quả EDA bước 5 giây tại `babdf38`/`21a9ef0`.

Theo PHAN_CONG_MAY_HOC.docx, phần G1 của Như là chốt đồng bộ ngực/cổ tay, bộ lọc, cửa sổ 60/5, nhãn đa số 90% và test biên nhãn. Tài liệu này bàn giao thiết kế và kiểm tra mẫu; thuật toán lọc sinh lý và extractor 68 feature chưa được triển khai.

## 1. Cấu hình và quy tắc cửa sổ

- Dùng chung `configs/features_smoke.yaml`, không có config smoke riêng theo thành viên.
- Cửa sổ 60 giây = 42.000 mẫu nhãn ở 700 Hz; bước 5 giây = 3.500 mẫu. Hai cửa sổ liên tiếp chồng lấn 55 giây.
- Giữ trục thời gian gốc, các khoảng nửa mở `[0,60)`, `[5,65)`, `[10,70)`...; bỏ phần cuối thiếu độ dài.
- Nhãn đa số thuộc 1/2/3 và chiếm ít nhất 90% toàn cửa sổ. Nhãn 0/4/5/6/7 vẫn trong mẫu số, không trở thành lớp đích, không bị xóa để nối hai đoạn rời nhau.
- Đúng 37.800/42.000 mẫu được nhận; 37.799/42.000 bị loại. Không ép kết quả theo số cửa sổ của cấu hình cũ.
- Nhị phân: Stress 2 -> 1; Baseline 1 và Amusement 3 -> 0.
- Khóa `(subject_id, window_id)`, window_id = `<subject>_<start_index_700hz>`; thời gian tính từ đầu PKL đồng bộ.
- Seed chung 42 khi thuật toán có ngẫu nhiên; extractor thống kê raw hiện không lấy mẫu ngẫu nhiên.

CLI và hàm `iter_windows` mặc định đã thống nhất bước 5 giây, majority 90%. Quy tắc homogeneous chỉ còn là lựa chọn tường minh để đọc cấu hình cũ, không phải cấu hình chạy G1 hiện tại.

## 2. Thiết kế đồng bộ ngực và cổ tay

| Thiết bị/kênh | Tần số | Số mẫu trong 60 giây |
| --- | ---: | ---: |
| Ngực ECG, EDA, Resp, Temp, EMG, ACC | 700 Hz | 42.000 |
| Cổ tay BVP | 64 Hz | 3.840 |
| Cổ tay ACC | 32 Hz | 1.920 |
| Cổ tay EDA, TEMP | 4 Hz | 240 |

Đọc PKL WESAD đã đồng bộ, dùng chung mốc thời gian. Tạo danh sách cửa sổ/nhãn một lần trên lưới nhãn 700 Hz; mỗi kênh cắt cùng khoảng thời gian theo tần số gốc của nó. Không nâng toàn bộ kênh lên 700 Hz, không tự thêm offset hoặc cắt ngắn khi thiếu mẫu.

Đã kiểm tra cấu trúc S2: 10 kênh có shape đúng, không có NaN/Inf và cùng thời lượng 6.079 giây với nhãn. Đây là kiểm tra cấu trúc, không suy ra chất lượng sinh lý. Hàm iter_windows xác định nhãn trên lưới 700 Hz; processor multisensor hiện có cắt từng kênh theo tần số gốc và cùng khóa. Lượt mẫu G1 gọi processor này, không tự viết thêm một bộ xử lý riêng.

## 3. Lựa chọn bộ lọc và chất lượng

Hợp đồng có cấu trúc: `configs/preprocessing_g1_design.yaml`, `stage: G1`, `runtime_ready: false`. File này không truyền cho extractor raw.

| Kênh | Thiết kế | Bỏ biên mỗi đầu đoạn hữu hạn |
| --- | --- | ---: |
| EDA ngực | Low-pass 5 Hz | 10 giây |
| Resp ngực | Band-pass 0,1–0,35 Hz | 30 giây |
| EDA cổ tay | Low-pass 1 Hz | 10 giây |

Cutoff EDA ngực và Resp đối chiếu bài báo WESAD mục 4.1, trang PDF 4. Cutoff EDA cổ tay 1 Hz là lựa chọn G1 được Mai đưa vào schema ở `b5a0cb8`, không khẳng định là thông số bài báo; ở 4 Hz, cutoff phải nhỏ hơn Nyquist 2 Hz.

Chọn Butterworth bậc thiết kế 4, SOS, lọc hai chiều offline, odd padding trên các đoạn hữu hạn liên tục trong từng người trước khi cắt cửa sổ. Mai đã đưa revision `nhu_g1_design_v2`, gồm biên 10/30 giây, QA cửa sổ và dung sai raw G1, vào schema/hợp đồng tại `2f380a8`. Đây là thống nhất thiết kế trong repo; vùng biên chưa được chứng minh hết quá độ và cần kiểm tra tín hiệu khi triển khai.

Không lọc xuyên NaN/Inf, không thay bằng 0 hoặc nội suy âm thầm. Đoạn quá ngắn cho padding/bỏ biên được đánh dấu không dùng được. Loại và ghi lý do cho cửa sổ thiếu mẫu, có giá trị không hữu hạn hoặc giao vùng biên ở kênh yêu cầu; dùng chung tập khóa cho so sánh cảm biến. Tín hiệu phẳng/tiếp xúc nghi ngờ được gắn cờ để xem xét, không tự chứng nhận đạt chất lượng chỉ vì hữu hạn.

ACC giữ trọng lực; magnitude = sqrt(x²+y²+z²). ACC/TEMP giữ tín hiệu gốc cho thống kê ban đầu, chưa tự đổi scale sang g/mV/đơn vị khác. Ghi chú S2 cho biết nhiệt độ ngực có vấn đề tiếp xúc; chưa tự loại S2 hoặc sửa giá trị nhiệt độ.

ECG/BVP/EMG gắn với detector/feature, decomposition EDA/SCR, PSD, chuyển đơn vị và ngưỡng QA sinh lý cần Như/Mai thống nhất trước triển khai feature. Chưa áp dụng các thuật toán này vào lượt mẫu G1.

## 4. Bàn giao với Mai

Nhóm hiện chốt bộ 112 thống kê raw; schema sinh kèm gói 112 là hợp đồng nghiệm thu đang dùng. Schema nháp 68 feature của Mai là thiết kế sinh lý trước đó. Giữ revision thiết kế bộ lọc v2 làm tài liệu, chưa áp dụng bộ lọc lên gói raw 112 và chưa đổi công thức feature của nhóm.

ACC/EMG: chia cửa sổ 60 giây thành 12 đoạn 5 giây không chồng lấp, tính feature mỗi đoạn rồi lấy trung bình. Các feature khác theo cửa sổ 60 giây. Đây là hợp đồng, chưa viết helper hoặc extractor G2.

Giữ 8 metadata và tiền tố `chest_`/`wrist_` theo schema Mai. Metadata không vào X; imputer/scaler/selection/resampling chỉ học trong train của từng fold. Bảng mẫu hiện tại chỉ có 112 thống kê raw, dùng schema sinh kèm CSV, không dùng schema nháp 68 feature để kiểm tra bảng này.

## 5. Kiểm tra và điều kiện nghiệm thu

Đã chạy 35 test trên Python 3.12.14: 14 hạ tầng gốc, 7 kiểm tra gói G1, 6 kiểm tra cửa sổ/CLI và 8 kiểm tra đa cảm biến theo README. Test mới kiểm tra đúng 90%, thiếu một mẫu so với 90%, nhãn bị loại, lưới 60/5, phần dư cuối và giữ thời gian gốc. Kết quả test không thay cho nghiệm thu các máy.

Đã chạy mẫu S2 theo lệnh README và công cụ kiểm tra gói. Kết quả trong [biên bản S2](G1_NHU_KIEM_TRA_S2.md). Lệnh chạy, file bàn giao và cách đối chiếu ở [bàn giao G1](G1_NHU_BAN_GIAO.md).

Phần thiết kế và chạy mẫu trên máy Như đã có bằng chứng. Nam có bảng tóm tắt cùng số cửa sổ/lớp S2 và cùng cấu hình, nhưng chưa có toàn bộ gói để so khóa/giá trị từng dòng. Nghiệm thu bốn máy vẫn chờ gói cùng source/môi trường và xác nhận thực tế. Chỉ tiếp tục G2 khi người dùng yêu cầu.

## Nguồn đối chiếu

- PHAN_CONG_MAY_HOC.docx, mục 3 Giai đoạn 1, file người dùng cung cấp.
- README.md của Nam và `notebooks/reports/tables/eda_step5/` trên main.
- `configs/feature_schema.json`, `docs/G1_HOP_DONG_FEATURE.md` của Mai.
- Bài báo WESAD mục 4.1 và wesad_readme.pdf trong bộ dữ liệu gốc.
