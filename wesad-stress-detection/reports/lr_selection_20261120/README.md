# Kết quả giai đoạn chọn đặc trưng LR

Kiểm tra lúc 2026-11-20T23:46:46.075872+07:00.

So sánh ANOVA (FS-01), mutual information (FS-02) và loại tương quan cao (FS-03). Thời gian chạy và cấu hình được lưu trong hồ sơ từng run.

| Run | Task | Macro-F1 ± SD | Balanced Accuracy ± SD | Δ F1 so LR đối chứng |
|---|---|---|---|---|
| FS-01 | multiclass | 0.6127 ± 0.1762 | 0.6589 ± 0.1429 | +0.0233 |
| FS-01 | binary | 0.8836 ± 0.1617 | 0.8939 ± 0.1319 | +0.0734 |
| FS-02 | multiclass | 0.6075 ± 0.1679 | 0.6577 ± 0.1429 | +0.0181 |
| FS-02 | binary | 0.8340 ± 0.1898 | 0.8562 ± 0.1521 | +0.0238 |
| FS-03 | multiclass | 0.5850 ± 0.1504 | 0.6281 ± 0.1373 | -0.0044 |
| FS-03 | binary | 0.8135 ± 0.1932 | 0.8331 ± 0.1752 | +0.0032 |

Đủ 3 run × 30 fold; mỗi run có 12.378 dự đoán OOF. Kiểm tra độc lập đạt. Gate selection khóa 30 công thức theo task/outer subject bằng inner Macro-F1. Không dùng bảng điểm outer ở trên để chọn phương pháp cho bước sau.

Kết quả tăng/giảm phụ thuộc phương pháp và người. Không kết luận khác biệt có ý nghĩa thống kê từ chênh lệch trung bình này. Điểm inner là tiêu chí lựa chọn và có thể lạc quan; benchmark đã được sử dụng nhiều lần, chưa có external holdout.

Mỗi bài toán được đánh giá riêng trên cùng 15 subject. Các chênh lệch trong bảng được tính so với LR đối chứng trên các cặp subject tương ứng.

Nguồn: comparison.csv, verification.json, và các thư mục run ghi trong cột source.
