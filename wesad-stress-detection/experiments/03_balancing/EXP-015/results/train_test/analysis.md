# EXP-015: so sánh train và test — 18/11/2026

Đây là lượt refit bổ sung để đo cả train và test từ cùng mô hình. Giữ nguyên cấu hình đã chốt, không tuning lại hoặc chọn theo điểm test.
Dự đoán test ở đủ 30 folds khớp toàn bộ kết quả đã lưu. Không coi điểm train mới đo là điểm đã được ghi trong lượt chạy gốc.

- binary: macro f1 train 1.0000, test 0.7826; khoảng cách 0.2174.
- multiclass: macro f1 train 1.0000, test 0.6056; khoảng cách 0.3944.

Mô hình khớp rất sát train nhưng điểm trên người chưa gặp thấp hơn ở nhiều folds. Khoảng cách này cho thấy khả năng tổng quát hóa còn hạn chế, có thể liên quan đến overfitting và khác biệt giữa các người; biểu đồ chưa tách được hai nguyên nhân.

Trục ngang là người được giữ lại làm test, không phải epoch. Hai đường trên mỗi trục là điểm train và outer test. Không gọi đây là learning curve train–validation hay loss theo epoch. Điểm train tính trên các cửa sổ gốc của 14 người, kể cả ở exp015; không nhân thêm trọng số do các mẫu ROS bị lặp.

Mean và độ lệch chuẩn mẫu theo 15 người. Các tập train giữa outer folds có chồng lặp nên không xem các điểm train là 15 quan sát độc lập để kiểm định.

Không dùng đồ thị để tự sửa tham số theo outer test. Nếu cần thử cải tiến, chọn bằng training/inner validation theo hướng nghiên cứu đã định.