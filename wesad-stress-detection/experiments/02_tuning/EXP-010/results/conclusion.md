## nhận xét kết quả — 15/11/2026

- binary: macro f1 tuning 0.7819 ± 0.1869; baseline 0.7986 ± 0.1846; giảm 0.0167. 5 người tăng, 7 người giảm, 3 người bằng nhau.
- multiclass: macro f1 tuning 0.6187 ± 0.1338; baseline 0.6462 ± 0.1144; giảm 0.0275. 5 người tăng, 10 người giảm, 0 người bằng nhau.

có 0/30 outer folds ghi cảnh báo trong quá trình search/refit. 0/30 mô hình refit chạm max_iter=200.
số epoch refit từ 15 đến 67.

train loss thuộc mô hình tốt nhất refit trên outer train, không phải validation loss. cảnh báo được runner gom cho cả search và refit, chưa gán riêng từng ứng viên.

tuning không đảm bảo cải thiện trên người mới. kết luận chỉ áp dụng cho bốn cấu hình đã khai báo; các cấu hình thay nhiều tham số cùng lúc nên chưa tách được ảnh hưởng riêng của alpha hay kiến trúc.

so sánh dùng cùng dữ liệu và subject, sklearn 1.9.1. baseline dùng pandas 3.1.1, lượt tuning dùng pandas 3.1.0; chưa kiểm tra độ nhạy với khác biệt này. đây là so sánh mô tả, chưa kiểm định ý nghĩa thống kê.

exp015 sẽ khảo sát cân bằng lớp train-only với tham số đã chọn. exp020 sẽ khảo sát kiến trúc và regularization khác lưới hiện tại. không sửa grid theo outer test để làm điểm đẹp hơn.