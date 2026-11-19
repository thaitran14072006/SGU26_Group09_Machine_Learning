# So sánh train và test

Mở `notebooks/04_evaluation/Thai_train_test_comparison.ipynb` và chạy các cell theo thứ tự.

- Bốn cấu hình: Dummy, LR mặc định, LR cân bằng lớp và LR chọn đặc trưng.
- 15 lượt LOSO cho mỗi cấu hình và bài toán; tổng cộng 120 fold.
- Giữ nguyên tham số và k đã chọn trong thí nghiệm gốc, fit lại trên train để lấy điểm train.
- Mỗi dự đoán test đã được đối chiếu chính xác với kết quả gốc.
- `train_test_metrics.csv`: điểm train/test và gap theo fold.
- `train_test_summary.csv`: mean và SD theo 15 fold.
- `verification.json`: bằng chứng đối chiếu, chia người và cảnh báo.
- `analysis_config.json`: nguồn dữ liệu, cấu hình nguồn, môi trường và checksum.
- `figures/`: 8 đồ thị theo cấu hình/bài toán và 2 hình tổng hợp.

Đường đồ thị nối mã người test, không biểu diễn epoch. Điểm train là resubstitution, không phải điểm validation độc lập.
