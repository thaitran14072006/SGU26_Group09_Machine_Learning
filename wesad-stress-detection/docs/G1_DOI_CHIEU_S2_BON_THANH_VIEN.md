# Đối chiếu bốn gói S2 — 06/11/2026

Nguồn: gói S2 trên máy Như và ba tệp người dùng cung cấp: Thai_G1_S2.zip, Mai_test.zip, Nam_test.zip. Gói Nam chứa một RAR bên trong, đã giải nén để đọc đủ năm tệp nghiệm thu. Không thay đổi nội dung gói nhận được.

## Kết quả dữ liệu

Cả bốn S2.csv giống nhau từng byte (SHA256 14bd24e6a471f58693e7822c03cc7b492eeba314dab290bc10a8281285d8384a).

- 395 cửa sổ, 8 cột metadata và 112 đặc trưng (63 ngực, 49 cổ tay).
- Nhãn: 219 Baseline, 113 Stress, 63 Amusement.
- Khóa cửa sổ, thời gian, nhãn, chất lượng và toàn bộ giá trị đặc trưng khớp; sai lệch tuyệt đối lớn nhất bằng 0.
- Cùng SHA256 dữ liệu S2 gốc, cấu hình 60 giây/bước 5 giây, majority >=90% và phiên bản all_sensors_raw_stats_v1.

## Ghi chú nguồn chạy và môi trường

| Người | Kiểm tra gói đơn | Khác biệt so với Như |
| --- | --- | --- |
| Như | Đạt | Gói tham chiếu Python 3.12.14 |
| Thái | Đạt | code_sha256 khác; git_dirty=true |
| Mai | Đạt | code_sha256 khác; không ghi nhận Git commit |
| Nam | Chưa đạt phiên bản môi trường | Python 3.13.5; numpy 2.5.3, pandas 3.0.6, imbalanced-learn không ghi nhận; code_sha256 khác; git_dirty=true |

Đối chiếu nghiêm ngặt giữa Như và từng người chưa đạt do thông tin nguồn chạy khác nhau. Không sửa manifest hoặc bỏ qua điều kiện để ép đạt.

## Kết luận nghiệm thu theo quyết định nhóm

Ngày 06/11/2026, Như xác nhận nhóm chấp nhận tiêu chí kết quả S2 giống nhau để chốt phần đối chiếu G1. Đã nhận đủ bốn gói và toàn bộ CSV giống hệt nhau, vượt mức chỉ khớp số dòng hoặc số đặc trưng. Theo tiêu chí này, đối chiếu S2 đạt và phần G1 được phân công của Như hoàn tất; không yêu cầu chạy lại.

Khác biệt mã nguồn và môi trường ở bảng trên được giữ làm thông tin tham khảo. Đây là quyết định nghiệm thu theo kết quả đầu ra, không phải kết luận các máy dùng cùng mã nguồn và môi trường. Công cụ kiểm tra nghiêm ngặt vẫn giữ nguyên hành vi và không sửa các manifest hoặc kết quả JSON trước đây để ép đạt.

Biên bản này xác nhận nội dung các gói đã nhận theo tên thành viên được cung cấp; không tự xác thực danh tính máy hoặc tính độc lập của các lần chạy. Kết luận chỉ áp dụng cho phần đối chiếu và phần G1 của Như, không thay xác nhận hoàn thành công việc riêng của các thành viên khác và không triển khai G2.

## Lưu bằng chứng

Báo cáo tổng hợp này được đưa vào Git. Các ZIP/RAR, CSV và kết quả JSON chi tiết vẫn lưu ngoài Git theo .gitignore, tại data/features/g1_received_20261106/ và outputs/g1/four_members_20261106/ trên máy Như; gói tham chiếu tại data/features/g1_s2_all_sensors_raw_stats_v1/.
