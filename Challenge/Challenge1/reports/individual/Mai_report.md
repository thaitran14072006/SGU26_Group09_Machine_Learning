# Báo cáo cá nhân Gradient Boosting

## Bài toán và dữ liệu

Mô hình dự đoán `SalePrice` cho 1.459 căn trong tập test. Danh sách chia chung gồm 1.168 căn development và 292 căn holdout. Năm fold trên development được dùng xuyên suốt để so sánh các cấu hình bằng RMSE trên log giá.

Trên development, giá bán có đuôi phải. Các cột `PoolQC`, `MiscFeature` và `Alley` thiếu nhiều nhất. Một số giá trị NA biểu thị căn nhà không có tiện ích, chẳng hạn hồ bơi. Chất lượng tổng thể và diện tích sinh hoạt đều có tương quan dương với giá nên được chọn để thử đặc trưng tương tác.

## Phương pháp và kết quả

Pipeline điền biến số thiếu bằng trung vị, xử lý các trường hợp NA mang nghĩa không có hạng mục theo mô tả dữ liệu và mã hóa one-hot. Phần tiền xử lý được fit trong từng fold. Mô hình học `log(SalePrice)`.

| Thực nghiệm | Ngày quy ước | CV RMSE log trung bình ± ĐLC |
|---|---|---:|
| Gradient Boosting cơ sở | 01/11/2026 | 0.13655 ± 0.01619 |
| Thêm `OverallQual × GrLivArea` | 06/11/2026 | 0.13067 ± 0.01421 |
| Giữ đặc trưng tương tác và tinh chỉnh | 13/11/2026 | **0.12863 ± 0.01493** |

Ngày trong bảng là mốc quy ước cho nhật ký. Kết quả lấy từ notebook đã chạy và được đối chiếu bằng lần chạy lại ngày 15/11/2026. Độ lệch chuẩn dùng cùng cách tính với notebook. Thông số, điểm từng fold, nguồn và thời điểm đối chiếu nằm trong các file `result.json`.

Cấu hình được chọn dùng 200 cây, `learning_rate=0.1`, `max_depth=3` và `random_state=7`. RMSE log train là 0.05189, thấp hơn validation 0.07674. Trên holdout, cấu hình đạt **RMSE log 0.11589** và **MAE 15.236**. Nhóm giá trên 224.600 có MAE 26.386. Bốn trong năm căn sai lệch lớn nhất bị dự đoán thấp hơn giá thực tế.

## Kết quả dự đoán

Mô hình được fit lại trên toàn bộ train rồi xuất 1.459 dự đoán theo thứ tự file mẫu. Thư mục cá nhân lưu mô hình, biểu đồ, sai số holdout và 1.168 dự đoán OOF. Khi dùng mô hình đã lưu, cần tạo `QualityArea` và xử lý `MasVnrType` như hàm `make_features` trong notebook.

- [Notebook](../../members/Mai/notebooks/gradient_boosting.ipynb)
- [Submission cá nhân](../../results/submissions/submission_Mai.csv)
- [OOF cá nhân](../../members/Mai/outputs/predictions/oof_Mai.csv)
