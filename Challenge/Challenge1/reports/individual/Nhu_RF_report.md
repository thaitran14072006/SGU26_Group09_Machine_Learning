# random forest house prices — như

Ngày thực hiện: 15/11/2026.

Dự đoán SalePrice từ 79 thuộc tính gốc; bỏ Id khỏi X. Train có 1.460 nhà,
development 1.168 và holdout 292 nhà theo split seed 7 của nhóm.
Huấn luyện bằng log tự nhiên của giá, chuyển lại bằng exp khi tính MAE và xuất dự đoán.

## giả thuyết và phương pháp

Kiểm tra xử lý NA theo ý nghĩa, đặc trưng diện tích/tuổi nhà/tiện ích và giới hạn
độ sâu hoặc kích thước lá có giảm sai số so với RF cơ sở không.
Mỗi fold tự fit imputer và one-hot. RF không cần chuẩn hóa. Không loại ngoại lệ,
không fit preprocessing trên holdout hoặc test. So sánh sáu phương án đã định trước,
chọn bằng 5-fold CV; độ lệch chuẩn mẫu dùng ddof=1.

| thí nghiệm | đặc trưng | cv rmse log ± sd | cv mae giá |
|---|---|---:|---:|
| Nhu_EXP-001 | baseline | 0.14956 ± 0.01952 | 18,159 |
| Nhu_EXP-002 | semantic | 0.15014 ± 0.01900 | 18,130 |
| Nhu_EXP-003 | engineered | 0.14518 ± 0.02113 | 17,550 |
| Nhu_EXP-004 | engineered | 0.14535 ± 0.02032 | 17,521 |
| Nhu_EXP-005 | engineered | 0.14537 ± 0.02315 | 17,638 |
| Nhu_EXP-006 | engineered | 0.14533 ± 0.02308 | 17,611 |

## kết quả

Chọn Nhu_EXP-003 bằng CV: 0.14518 ± 0.02113.
Chênh lệch với baseline: -0.00438; âm là giảm sai số.
Holdout được đánh giá sau khi lưu selected_config.json: RMSE log 0.13307,
MAE giá 16,825. Điểm train development 0.05403
cần đọc cùng CV/holdout để nhận diện khoảng cách train–validation.
Không đổi cấu hình sau khi xem holdout. Chưa có điểm Kaggle.

## sai số và đóng góp đặc trưng

Nhóm giá có MAE lớn nhất trên holdout: >250k.
Ba feature đứng đầu importance CV: TotalArea, OverallQual, CentralAir.
Importance đo mức giảm impurity, có thể thiên về biến nhiều mức và chia sẻ đóng góp
giữa biến tương quan; không chứng minh quan hệ nhân quả. Exp003 so với exp002 là
kiểm tra thêm cả nhóm feature, chưa tách ablation từng feature.

## sản phẩm và chạy lại

Notebook: members/Nhu/notebooks/15_11_2026_rf.ipynb.
Kết quả: members/Nhu/outputs/experiments/20261115_171330_237145.
CSV nộp: results/submissions/submission_Nhu.csv; model đã fit toàn bộ train nằm trong run/models.
Mỗi lần chạy notebook tạo thư mục mới. Chỉ nạp joblib do nhóm tự tạo.
Các cải tiến có thể làm sau: permutation importance trong CV, ablation từng nhóm,
khảo sát thêm max_features và số cây bằng development CV, không tối ưu theo holdout.
