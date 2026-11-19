# EXP-006

## so sánh train và test bổ sung ngày 18/11/2026

`results/train_test/train_vs_test.png` đặt điểm train và outer test trên cùng
trục cho 15 người, cả hai bài toán và hai độ đo. Điểm được tính từ cùng mô hình
refit, giữ cấu hình đã chốt. Dự đoán test khớp kết quả cũ. Các ô cũ trong notebook
được giữ nguyên, phần mới nằm cuối notebook.
Điểm train tính trên cửa sổ gốc, không trên các bản sao ROS. Đây không phải loss
theo epoch hoặc learning curve train–validation. Xem `analysis.md` và
`verification.json` trong thư mục đó. Không dùng test để chọn tham số mới.

Chạy phần đo điểm mới bằng `scripts/nhu_train_test_diagnostics.py`; vẽ/kiểm tra
từ dữ liệu đã đo bằng `scripts/nhu_train_test_plots.py --run outputs/<lượt đo>`.
