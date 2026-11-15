# tổng hợp kết quả challenge — 15/11/2026

Như tổng hợp từ kết quả hiện có, không chạy lại mô hình của các bạn khác.
Bảng số đo ở results/group_model_comparison.csv, mỗi dòng có đường dẫn nguồn.
Điểm càng thấp càng tốt. Độ lệch chuẩn trong bảng so sánh đã quy về mẫu ddof=1;
nhật ký gốc giữ nguyên cách tính và ngày của người tạo.

## kiểm tra và giới hạn

- Dữ liệu train/test hiện tại khớp nội dung CSV trong ZIP người dùng gửi.
- Danh sách development/holdout và cv_fold khớp KFold seed 7 theo đúng thứ tự Id.
- RF của Như đã kiểm tra 30 fold, 1.459 Id và dự đoán từ pipeline nạp lại.
- Đây không phải chứng nhận đã train lại cả bốn mô hình trên cùng môi trường.
- Hash byte train ở nhật ký Thái khác file hiện tại; chưa có snapshot dữ liệu
  của lượt đó để xác nhận khác do xuống dòng hay khác nội dung. Chưa chốt xếp hạng.
- Các phiên bản sklearn và độ lớn không gian tìm kiếm giữa các bạn còn khác nhau.
- Không lấy holdout để chọn trọng số ensemble. Mai dùng OOF development nếu kết hợp.
- Chưa nộp các submission lên Kaggle và chưa có điểm leaderboard.
