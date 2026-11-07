# Nghiệm thu G2 tiền xử lý và đặc trưng

**Chốt G2 bản cơ bản ngày 07/11/2026**, theo mục 4 “Giai đoạn 2 Tiền xử lý và đặc trưng”, phần “Nghiệm thu” trong `PHAN_CONG_MAY_HOC.docx` và yêu cầu chốt của người dùng. Đây là kết luận kiểm tra kỹ thuật và chính sách bàn giao dữ liệu; không ghi nhận thay chữ ký/review riêng của bốn thành viên, không xác nhận đã hoàn thành trước hạn 05/11.

## Đối chiếu tiêu chí

| Tiêu chí trong phân công | Bằng chứng và kết luận |
| --- | --- |
| Đủ S2–S11 và S13–S17 | Đủ 15 subject, 6.189 dòng, không thiếu subject; mỗi người chỉ nhận một lần |
| Không trùng khóa | Không trùng `(subject_id, window_id)`; S2 có đúng 395 cửa sổ |
| Schema và phiên bản thống nhất | `physiological_68_v1`, 68 feature + 8 metadata; cùng mã nguồn, processing và Python 3.12.14 trong bộ chung |
| Ghi số cửa sổ trước/sau loại và lý do | 17.202 ứng viên; loại 11.006 do nhãn; 6.196 đạt nhãn; loại thêm 7 cửa sổ S14 do biên lọc; giữ 6.189. Không loại thêm do nonfinite/thiếu mẫu |
| Chốt tập cửa sổ chung ba nhóm cảm biến | Cùng 6.189 khóa cho ngực 37 feature, cổ tay 31 feature, kết hợp 68 feature; danh mục `common_windows.csv` trong gói |
| Công bố ảnh hưởng lọc chất lượng | Quality report từng subject, thống kê BVP từng subject/nhãn, lý do NaN; công bố 7 cửa sổ loại thêm và số complete-case 345, không dùng complete-case để cắt bảng |
| Số 6.000–7.000 chỉ là ước lượng | Không ép số cửa sổ; giữ số thật 6.189 |

Đã kiểm tra manifest, chữ ký nội bộ, raw hash của 15 người, checksum gói/schema/CSV, số dòng/nhãn và nhãn nhị phân. Bảng tổng khớp CSV từng subject nối một lần; không nối thêm bảng tổng từ gói khác. Không Inf. ZIP được đọc lại để kiểm tra CRC và hash từng tệp. Bộ kiểm thử checkout hiện tại: **76/76 đạt**.

## Phương pháp được giữ trong bản cơ bản

Giữ trục thời gian và tần số gốc, cửa sổ 60 giây/bước 5 giây, nhãn đa số 1/2/3 đạt ít nhất 90% trên toàn cửa sổ gốc, kể cả nhãn không dùng trong mẫu số. Provider thực hiện bộ lọc/calibration/mask một lần; lớp feature không lọc lại. ACC/EMG dùng đủ 12 đoạn 5 giây rồi lấy trung bình. Các nhóm ECG/BVP, EDA tonic/phasic/SCR, RESP và TEMP theo hợp đồng đầy đủ kèm gói, gồm đơn vị, cutoff và quy tắc thiếu.

Lượt tạo bộ chung dùng nguồn `7720b01` của Như, code hash `abd964e88fa0e5922c8bfc2cb71e1f4ad46ebb1d381e3f6bf82571817e1b452b`, Python 3.12.14 và thư viện khóa. Bằng chứng nguồn chạy đi kèm. Không thay detector/công thức, không sửa chữ ký và không trộn raw112 hoặc bảng chẩn đoán 72 cột.

## Chính sách chất lượng được chốt

1. Dùng đúng 6.189 khóa chung cho cả ba nhóm cảm biến. Cửa sổ vượt kiểm tra đầu vào/mask nhưng có feature không đủ tin cậy vẫn giữ, với NaN và cờ lý do. Không bỏ hàng riêng theo từng nhóm cảm biến.
2. Không yêu cầu BVP đầy 100%. Còn 5.312/6.189 cửa sổ thiếu sáu feature nhịp/IBI BVP; feature PSD vẫn có thể có giá trị. Mức thiếu phụ thuộc subject/nhãn, được công bố trong báo cáo, không diễn giải thành dữ liệu sinh lý đã được kiểm chứng độc lập.
3. Tổng 37.460 ô NaN có lý do, gồm các nhóm nhịp, SCR không có sự kiện, RESP không đủ chu kỳ và TEMP ngực S2 nghi tiếp xúc kém. Không điền 0 hoặc hạ ngưỡng để ép hết thiếu. Chỉ 345 cửa sổ đủ mọi feature; đó là số đếm tác động, không phải tập dùng chung thay thế.
4. Không fit imputer, scaler, bộ chọn feature hoặc resampling trên bảng đầy đủ. Xử lý NaN theo train fold thuộc giai đoạn thực nghiệm tiếp theo; không thực hiện G3 trong lượt nghiệm thu này.

**NaN có lý do là trường hợp được tài liệu phân công cho phép.** Hạn chế BVP đã công bố không tự trở thành yêu cầu phải sửa detector trước khi bàn giao bản cơ bản. Nếu sau này thay phương pháp, phải khóa phiên bản/hợp đồng mới và chạy lại dữ liệu chịu ảnh hưởng.

## Gói Nam mới và bộ dữ liệu chính thức

Đã kiểm tra `g2_physiological68_nam_py31214.rar`: Python 3.12.14, đủ S3/S8/S14/S16, 1.638 cửa sổ; checksum, manifest, raw hash, QA, schema và nhãn hợp lệ. Build Python/code hash của gói được lưu đúng producer, không tuyên bố giống toàn phần bộ chung. Báo cáo: [nam_py31214_verification.json](../reports/g2_physiological68/nam_py31214_verification.json).

Bộ chính thức vẫn là lượt chung đã chạy đủ 15 người trong một môi trường. Không nối gói Nam mới vào bảng này vì các subject đã có sẵn. Các gói bàn giao và báo cáo cũ giữ làm truy vết; vấn đề Python 3.13.5 của gói Nam trước không còn chặn nghiệm thu bộ chính thức.

- Gói: `data/features/g2_physiological68_group15_20261107/` và ZIP cùng tên, ngoài Git.
- Bảng đọc chung: `all_subjects.csv`.
- Hợp đồng/schema, manifest, config, quality report, checksum, danh mục cửa sổ chung và biên bản nghiệm thu được kèm trong gói.
- Kết quả nghiệm thu máy đọc được: [G2_acceptance.json](../reports/g2_physiological68/G2_acceptance.json).
- Báo cáo bàn giao: [G2_BAN_GIAO.md](G2_BAN_GIAO.md).

TODO sau bản cơ bản: rà/cải tiến detector BVP bằng kiểm tra tín hiệu và một phiên bản riêng nếu nhóm triển khai. Không tạo thêm yêu cầu RAM, không làm mô hình G3, không đưa raw/ZIP/bảng feature lớn lên Git. Chưa tự commit/push các cập nhật nghiệm thu này.
