# Đối chiếu hợp đồng G1 với hạ tầng Thái

Ngày kiểm tra: 03/11/2026. Mốc Git tham chiếu: `e12c577d2e73e847b6db9374a3a970308c9fd110`. Phạm vi là tương thích giao diện G1; không nghiệm thu thuật toán sinh lý hoặc kết quả S2 trên bốn máy.

Đã đọc `data.py`, `extraction.py`, `preprocessing.py`, `evaluation.py`, CLI và cấu hình smoke; đối chiếu Git để phân biệt thay đổi cục bộ. Các hàm loader/validator, windowing, task_target và process_subject raw dùng trong kiểm tra không khác bản Git. Các thay đổi cục bộ trong run_extraction về resume/quality manifest là thử nghiệm trước, không tính là phần Thái đã bàn giao.

## Kết quả

| Điểm đối chiếu | Hạ tầng hiện có | Kết luận cho G1 |
| --- | --- | --- |
| Đối tượng | SUBJECTS = S2–S11, S13–S17 | Khớp 15 người |
| Metadata | Có 8 cột của schema; METADATA còn nhận label_purity để tương thích | Khớp; không cần thêm label_purity |
| Khóa cửa sổ | `<subject>_<start_index>`; thời gian = index/700 | Khớp quy ước start_index_700hz |
| Cửa sổ/nhãn | iter_windows hỗ trợ majority ≥90%, mẫu số gồm nhãn bị loại; smoke dùng 60/5 | Khớp khi dùng config majority; config features_chest.yaml cũ vẫn là homogeneous 60/60 |
| Nhãn nhị phân | task_target ánh xạ stress 2→1, các lớp 1/3→0 sau validate nhãn | Khớp; nhãn ngoài 1/2/3 phải bị loại/từ chối trước task_target |
| Chọn X | load_feature_table lấy feature_columns từ schema, runner lấy X theo danh sách đó | Khớp; 8 metadata không vào X, thứ tự X theo schema |
| Giá trị thiếu | NaN feature được giữ; Inf bị từ chối; imputer trong pipeline train | Khớp quy ước CSV ô trống; lý do thiếu cần processor G2 cung cấp |
| Kiểm tra bảng | Từ chối khóa trùng, metadata bắt buộc thiếu, subject/nhãn sai, thời gian không hữu hạn hoặc end≤start | Dùng lại được |
| Processor mở rộng | subject_processor nhận `module:function`, hàm `(path, config) -> DataFrame` | Có sẵn điểm nối cho features.py; không phải viết lại loader/CLI |
| Schema output | run_extraction suy ra feature_columns từ bảng thật rồi tạo schema gói output | Schema configs là hợp đồng G1; schema output G2 phải đối chiếu lại với hợp đồng đã khóa |

## Giới hạn cần ghi rõ

1. **21 raw feature không phải 68 feature G1.** Processor raw có tên ecg_/eda_/resp_, chưa có prefix thiết bị và chưa có HRV/SCR/cổ tay. Kiểm tra xác nhận không thể dùng schema 68 cột để đọc bảng raw 21 cột. Giữ nguyên bộ raw; G2 dùng processor và feature_version/output_dir mới sau khi chốt với Như.
2. Validator chung chỉ bắt buộc 6 metadata; label_binary và label_fraction là tùy chọn ở hạ tầng, trong khi hợp đồng Mai yêu cầu đủ 8. Có kiểm tra binary nếu cột hiện diện; chưa kiểm tra label_fraction ≥0,90, grid 5 s, mẫu window_id hoặc toàn bộ dtype theo schema. run_extraction kiểm tra thời lượng, nhưng load_feature_table riêng không bắt buộc 60 s. Processor/kiểm tra nghiệm thu G2 phải thực hiện các ràng buộc chặt hơn này.
3. Loader hiện chỉ đọc feature_columns, không thực thi các trường status, ready_for_training, unit hoặc nullable của schema G1. Cờ draft là thông tin hợp đồng, chưa phải chốt chặn trong code. Không truyền schema nháp vào lượt huấn luyện chính chỉ vì loader đọc được.
4. Loader chọn X đúng thứ tự schema nhưng không bắt CSV vật lý có đúng thứ tự hoặc không có cột dư; cần đối chiếu toàn bộ output_column_order khi nghiệm thu G2. Prefix thiết bị là quy ước của processor mới, không phải quy tắc có sẵn trong validator.
5. G1 vẫn chờ thiết kế tiền xử lý, scale/đơn vị, detector/dải tần, chính sách QA và dung sai S2. Đây không phải thiếu hạ tầng đọc bảng từ Thái.

Không sửa mã Thái trong lượt đối chiếu này. Các yêu cầu bổ sung trên được ghi vào hợp đồng để tích hợp/kiểm tra ở G2, chưa coi là đã thực thi.

## Kiểm tra thực thi

**5/5 kiểm tra đạt** trong `tests/test_g1_contract.py`:

- Danh sách người, metadata và nhóm cột khớp hạ tầng.
- Bảng giả đủ 68 feature đọc được, X đúng thứ tự schema dù CSV đảo cột; NaN được giữ.
- Nhãn binary khớp runner; binary không khớp bị từ chối.
- Nhãn không dùng, khóa trùng và Inf bị từ chối.
- Processor raw tạo metadata/window_id/time đúng hợp đồng trên dữ liệu giả 65 giây; bảng raw 21 feature không được nhận như bảng 68 feature.

Lệnh chạy từ gốc dự án (đặt TEMP/TMP vào thư mục được phép ghi nếu cần):

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p test_g1_contract.py -v
```

Kiểm tra dùng dữ liệu giả và file tạm, không chạy lại raw 15 người hoặc huấn luyện. Chưa thực hiện review của thành viên hay gửi tin nhắn thay người dùng.
