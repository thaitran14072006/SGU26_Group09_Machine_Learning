# Phần thực hiện 

## Mô hình cá nhân

[Notebook Gradient Boosting](notebooks/gradient_boosting.ipynb) trình bày EDA, tiền xử lý, các lượt thử đặc trưng và siêu tham số, phân tích sai số và xuất dự đoán. [Báo cáo cá nhân](../../reports/individual/Mai_report.md) tóm tắt kết quả.

Thư mục `outputs/` chứa minh chứng thí nghiệm, biểu đồ, mô hình đã lưu và dự đoán OOF. File nộp cá nhân là [submission_Mai.csv](../../results/submissions/submission_Mai.csv).

## Tổng hợp dự đoán của nhóm

Mai phụ trách tổng hợp các file dự đoán, thử trọng số bằng OOF và xuất submission ensemble. Mô hình đầu vào gồm Gradient Boosting của Mai, SVR của Nam, Random Forest của Như và ElasticNet của Thái.

- [Notebook ensemble](notebooks/ensemble.ipynb) giải thích cách kiểm tra file dự đoán, chọn trọng số và đọc kết quả.
- [Script Python](src/run_ensemble.py) ghép dữ liệu theo `Id`, chấm OOF và holdout, rồi xuất [bảng so sánh](../../ensemble/oof_comparison.csv) và [submission ensemble](../../results/submissions/submission_ensemble.csv).

Chạy lại bằng **Run All** trong notebook hoặc chạy `python -B members/Mai/src/run_ensemble.py` từ thư mục Challenge1.
