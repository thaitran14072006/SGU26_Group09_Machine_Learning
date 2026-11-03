# Bàn giao Mai Giai đoạn 1 ngày 03/11/2026

Phạm vi theo G1 của PHAN_CONG_MAY_HOC.docx, người phụ trách Nguyễn Thị Trúc Mai.

## Đầu ra

- `configs/feature_schema.json`: schema nháp có metadata, danh mục/thứ tự feature, kiểu dữ liệu, đơn vị, ba nhóm cảm biến và trạng thái các điểm cần chốt.
- `docs/G1_DANH_MUC_FEATURE.csv`: danh mục cùng tên/thứ tự với schema.
- `docs/G1_HOP_DONG_FEATURE.md`: định nghĩa feature, nhãn nhị phân, cách ghi thiếu, chất lượng và giao diện phối hợp.

Schema nháp gồm **8 metadata + 68 feature dự kiến** (37 ngực, 31 cổ tay). Nhóm kết hợp giữ thứ tự ngực rồi cổ tay. Đây là danh mục G1 cần đối chiếu trước khi triển khai, không phải 175 cột của bộ thử trước đó.

Đã chuẩn bị phần G1 của Mai. Trạng thái schema là **draft_pending_preprocessing_agreement**: còn chờ thống nhất thiết kế tiền xử lý với Như, xác minh scale/đơn vị raw, dải tần/detector và dung sai so sánh S2. Không đánh dấu nghiệm thu chung G1 hoặc S2 bốn máy khi chưa có bằng chứng.

## Đúng tên file theo phân công

Schema trong Git là `configs/feature_schema.json`; schema của từng gói feature G2 sẽ là `data/features/<feature_version>/feature_schema.json`. Hai file có mục đích khác nhau: bản G1 là hợp đồng dự kiến, bản G2 xác nhận bảng thực tế của một phiên bản đã khóa.

Module phụ trách theo phân công: `features.py` (G2), `baselines.py` và `metrics.py` (G3). G1 không tạo mô hình hoặc chạy lại 15 người. Các thử nghiệm sinh lý trước khi đọc phân công là bản thử riêng, không thay đầu ra G1 này.

## Kiểm tra đã thực hiện

Đọc được JSON/CSV; tên cột duy nhất, metadata tách khỏi feature; prefix và nhóm ngực/cổ tay/kết hợp thống nhất; mỗi feature có đơn vị/định nghĩa; thứ tự CSV trùng schema; có đầy đủ các nhóm yêu cầu và trạng thái pending.

## Bàn giao tiếp

Đã đối chiếu với hạ tầng Thái: metadata, window_id, nhãn binary, NaN, feature_columns và giao diện processor khớp; **5/5 kiểm tra tương thích đạt**. Chi tiết và giới hạn trong [G1_DOI_CHIEU_HA_TANG.md](G1_DOI_CHIEU_HA_TANG.md); mã kiểm tra ở `tests/test_g1_contract.py`.

Không còn chờ Thái xây loader/CLI. Còn phải thống nhất tiền xử lý với Như và nhận bất thường khảo sát từ Nam. Hạ tầng hiện chỉ bắt buộc 6 metadata và chưa thực thi tất cả ràng buộc của schema G1; processor/kiểm tra G2 phải đảm bảo đủ 8 metadata, purity, lưới thời gian và thứ tự cột. Bộ raw 21 feature không được dùng thay bộ 68 feature dự kiến.

Chưa gửi tin nhắn, chưa tạo PR hay ghi nhận review thay thành viên.
