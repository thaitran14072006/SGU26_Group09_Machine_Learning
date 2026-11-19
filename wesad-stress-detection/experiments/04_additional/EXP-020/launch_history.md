# nhật ký khởi động exp020 — 18/11/2026

- R01 khởi động thành công bằng runner timestamp ban đầu.
- Hai lệnh khởi động song song tiếp theo báo `FileExistsError: [WinError 183] Cannot create a file when that file already exists` ở `outputs/exp020_mlp_nested_20261118_222536_918478`.
- Hai lệnh này dừng trước huấn luyện, không tính thành lượt train.
- Sửa `scripts/exp020_train.py`: thêm UUID vào timestamp tên thư mục output. Không thay đổi lưới hoặc pipeline ML.
- Khởi động R02/R03 mới với thư mục khác nhau; mỗi lượt train đủ 30 outer folds và qua kiểm tra dự đoán, nhãn, điểm, inner splits, loss và checksum.
- Snapshot launcher trước sửa nằm trong `results/source_snapshot.zip`; sau sửa nằm trong các snapshot của `results/repeatability/R02` và `R03`.

Đây là lỗi khởi động có thật trong lần thực thi này, không phải lỗi hội tụ hoặc kết quả mô hình kém. Notebook đọc kết quả được thực thi bằng Python với bộ thu stdout và ảnh; ba lượt train được chạy qua CLI.
