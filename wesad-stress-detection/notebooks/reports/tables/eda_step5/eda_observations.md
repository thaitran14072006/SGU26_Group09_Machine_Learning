## Nhận xét EDA — cửa sổ 60 giây, bước 5 giây



Bảng hiện có **6,196 cửa sổ**, **15 người** và **21 đặc trưng** thống kê raw ECG/EDA/Resp ngực.

Cấu hình mục tiêu là majority ≥90%. Có thể có tối đa 10% mẫu khác nhãn trong cửa sổ hợp lệ.

Đã đối chiếu cấu hình trích xuất từ config.json; đây là kiểm tra metadata, chưa tính lại feature/nhãn từ PKL.



### Phân bố lớp

- Baseline: 3,377 cửa sổ (54.50%).

- Stress: 1,848 cửa sổ (29.83%).

- Amusement: 971 cửa sổ (15.67%).

Số cửa sổ theo người dao động 395–426.

Tỷ lệ số cửa sổ lớp lớn nhất/nhỏ nhất là 3.48.

Các subject thiếu ít nhất một lớp: không có trong bảng hiện tại.



### Chất lượng và điểm cần kiểm tra

Có 0 giá trị NaN, 0 giá trị Inf và 0 dòng thuộc nhóm khóa trùng.

Cột hằng/thiếu toàn bộ: [].

label_fraction quan sát từ 0.9015 đến 1.0000.

quality_flag phản ánh phạm vi kiểm tra của extractor; tính hữu hạn không chứng minh chất lượng sinh lý.

- ecg_max: 1400 giá trị bị đánh dấu IQR (22.60% trên giá trị hữu hạn).

- eda_std: 972 giá trị bị đánh dấu IQR (15.69% trên giá trị hữu hạn).

- eda_iqr: 917 giá trị bị đánh dấu IQR (14.80% trên giá trị hữu hạn).

Đánh dấu IQR phục vụ xem xét; chưa tự loại cửa sổ hoặc kết luận đó là lỗi.



### Tương quan

- ecg_std và ecg_rms: r = 1.000000.

- eda_mean và eda_rms: r = 0.999998.

- resp_std và resp_rms: r = 0.999998.

- eda_max và eda_rms: r = 0.998406.

- eda_mean và eda_max: r = 0.998334.

Nếu chọn đặc trưng cho mô hình, học lựa chọn trong train của từng fold; không chọn trên toàn bảng rồi báo điểm test như đánh giá độc lập.



### Tín hiệu thô

Đã kiểm tra 15 người và 45 cặp người–tín hiệu ECG/EDA/ACC trong lần chạy này.

Tổng NaN: 0; tổng Inf: 0; số cặp sai độ dài so với nhãn: 0.



### Hướng mô hình hóa

Chỉ dùng 21 feature làm X; giữ subject_id cho outer LOSO, không đưa metadata hoặc nhãn vào X.

Stress là lớp 1 trong bài toán nhị phân; Baseline và Amusement là Non-stress (0).

Imputer/scaler/selection/resampling học trong train của từng fold. Tuning chỉ dùng subject thuộc outer train.

Cửa sổ liên tiếp chồng lấn 55 giây, không coi chúng là quan sát độc lập. Không chia ngẫu nhiên các dòng train/test.

EDA không chứng minh hiệu năng hay khả năng khái quát của KNN/SVM. Báo cáo giữa kỳ tiếp tục bổ sung baseline và mô hình mặc định; tuning và kiểm định theo kế hoạch nhóm.