# Kết hợp dự đoán Challenge1

Thử kết hợp Gradient Boosting của Mai, SVR của Nam, Random Forest của Như và ElasticNet của Thái. Bốn bộ OOF được ghép theo `Id` và đối chiếu nhãn trên 1.168 căn development. RMSE log của từng bộ khi tính lại theo 5 fold chung khớp với kết quả nguồn.

Chênh lệch hash train trong ghi chú tổng hợp được giải thích bởi kiểu xuống dòng. SHA-256 của `train.csv` sau khi chuyển CRLF về LF khớp chính xác với hash trong `Thai_EXP-010`.

## Chọn trọng số

Thử trung bình đều và các bộ trọng số không âm có tổng bằng 1 với bước 0,10. Các dự đoán được kết hợp trên log giá. Cấu hình được chọn theo RMSE log trung bình của 5 fold OOF.

| Phương án | RMSE log CV |
|---|---:|
| SVR, mô hình đơn có CV thấp nhất | 0.12090 |
| Trung bình đều bốn mô hình | 0.12062 |
| 30% Gradient Boosting, 50% SVR, 20% ElasticNet | **0.11780** |

Random Forest có trọng số 0 trong cấu hình được chọn. Kết quả OOF dùng để chọn trọng số nên có thể lạc quan. Sau khi chốt trọng số, đánh giá trên 292 căn holdout cho **RMSE log 0.11072** và **MAE 13.707** trên giá gốc. Dự đoán holdout của SVR được tái tạo từ cấu hình V3 đã công bố và điểm khớp báo cáo của Nam.

## Tệp kết quả

- [Bảng thử trọng số](oof_comparison.csv)
- [Submission ensemble](../results/submissions/submission_ensemble.csv) có 1.459 dự đoán theo thứ tự file mẫu. Chưa gửi lên Kaggle.

Mai phụ trách tổng hợp và thử kết hợp. [Notebook ensemble](../members/Mai/notebooks/ensemble.ipynb) trình bày các bước và kết quả. [Script Python](../members/Mai/src/run_ensemble.py) chạy quy trình và xuất các tệp kết quả.

Chạy lại từ thư mục Challenge1 bằng `python -B members/Mai/src/run_ensemble.py`.
