# Bàn giao Mai Giai đoạn 1 ngày 03/11/2026

Phạm vi theo G1 của PHAN_CONG_MAY_HOC.docx, người phụ trách Nguyễn Thị Trúc Mai.

## Đầu ra

- `configs/feature_schema.json`: schema nháp có metadata, danh mục/thứ tự feature, kiểu dữ liệu, đơn vị, ba nhóm cảm biến và trạng thái các điểm cần chốt.
- `docs/G1_DANH_MUC_FEATURE.csv`: danh mục cùng tên/thứ tự với schema.
- `docs/G1_HOP_DONG_FEATURE.md`: định nghĩa feature, nhãn nhị phân, cách ghi thiếu, chất lượng và giao diện phối hợp.

Schema nháp gồm **8 metadata + 68 feature dự kiến** (37 ngực, 31 cổ tay). Nhóm kết hợp giữ thứ tự ngực rồi cổ tay. Đây là danh mục G1 cần đối chiếu trước khi triển khai, không phải 175 cột của bộ thử trước đó.

Đã chuẩn bị phần G1 của Mai. Trạng thái schema là **draft_pending_feature_details_and_s2_acceptance**: đã chốt dùng thiết kế tiền xử lý G1 của Như làm đầu vào theo yêu cầu người dùng; còn xác minh scale/đơn vị raw, dải tần/detector, vùng biên/QA và dung sai so sánh S2 chưa được quy định đầy đủ. Không đánh dấu nghiệm thu chung G1 hoặc S2 bốn máy khi chưa có bằng chứng.

## Đúng tên file theo phân công

Schema trong Git là `configs/feature_schema.json`; schema của từng gói feature G2 sẽ là `data/features/<feature_version>/feature_schema.json`. Hai file có mục đích khác nhau: bản G1 là hợp đồng dự kiến, bản G2 xác nhận bảng thực tế của một phiên bản đã khóa.

Module phụ trách theo phân công: `features.py` (G2), `baselines.py` và `metrics.py` (G3). G1 không tạo mô hình hoặc chạy lại 15 người. Các thử nghiệm sinh lý trước khi đọc phân công là bản thử riêng, không thay đầu ra G1 này.

## Kiểm tra đã thực hiện

Đọc được JSON/CSV; tên cột duy nhất, metadata tách khỏi feature; prefix và nhóm ngực/cổ tay/kết hợp thống nhất; mỗi feature có đơn vị/định nghĩa; thứ tự CSV trùng schema; có đầy đủ các nhóm yêu cầu và trạng thái pending.

## Bàn giao tiếp

Schema sử dụng metadata, khóa cửa sổ, nhãn và giao diện processor của hạ tầng Thái đã có. Tiền xử lý lấy theo thiết kế G1 Như đã bàn giao tại `G1_NHU_THIET_KE_TIEN_XU_LY.md`: đồng bộ tần số gốc, cửa sổ 60/5, EDA/RESP theo bộ lọc đề xuất của Như, ACC/EMG 12 đoạn 5 giây rồi gộp. Mai làm feature/schema theo đầu vào đó, không lập nhánh tiền xử lý riêng. Các thông số Như chưa quy định vẫn để mở trước khi khóa G2; bổ sung bất thường từ khảo sát Nam nếu ảnh hưởng feature/QA.

Chưa gửi tin nhắn, chưa tạo PR hay ghi nhận review thay thành viên.

## Thông số thiết kế G1 đã chọn theo Như

- Cửa sổ 60 giây, bước 5 giây, majority ≥90%; giữ tần số gốc từng kênh và cùng mốc PKL.
- EDA ngực low-pass 5 Hz; RESP ngực band-pass 0,1–0,35 Hz; EDA cổ tay low-pass 1 Hz.
- Butterworth bậc thiết kế 4, SOS, lọc hai chiều offline trên đoạn hữu hạn liên tục trước khi cắt cửa sổ.
- ACC/TEMP giữ tín hiệu gốc cho thống kê ban đầu; ACC/EMG chia 12 đoạn 5 giây rồi lấy trung bình feature.

Đây là chốt lựa chọn thiết kế G1 theo bản Như đã đưa vào repo, chưa phải nghiệm thu bộ lọc trên tín hiệu thật. Các thông số chưa có trong bản Như (detector, PSD, scale, vùng biên/QA) vẫn được ghi rõ còn mở để hoàn thiện trước lượt chính. Chưa chạy G2.
