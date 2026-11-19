# kiểm tra chạy lại exp015 — 18/11/2026

Có ba lượt train độc lập: lượt gốc ngày 16/11 và hai lượt mới ngày 18/11.
Mỗi lượt có 30 outer folds, tổng 90 lần fit MLP. Không dùng resume để tính thành lượt mới.
Cùng dữ liệu, môi trường, best_params từ exp010 và seed 42.

Đối chiếu 12.378 dự đoán và 30 bộ điểm mỗi lượt. Xem runs.csv và scores.csv.
Đây là kiểm tra khả năng chạy lại với cùng seed; chưa kiểm tra biến thiên giữa các seed.
Kết quả khớp không có nghĩa mô hình đã tổng quát hóa tốt hơn hoặc có ý nghĩa thống kê.
Mã EXP chung của cả ba lượt là EXP-015.
