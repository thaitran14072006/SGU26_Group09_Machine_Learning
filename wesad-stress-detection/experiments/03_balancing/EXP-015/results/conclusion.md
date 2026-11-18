## nhận xét — 16/11/2026

- binary: macro f1 cân bằng 0.7826 ± 0.1971; exp010 0.7819 ± 0.1869; tăng 0.0007. 7 người tăng, 5 người giảm, 3 người bằng nhau.
- multiclass: macro f1 cân bằng 0.6056 ± 0.1424; exp010 0.6187 ± 0.1338; giảm 0.0130. 5 người tăng, 9 người giảm, 1 người bằng nhau.

giữ đúng best_params của exp010 cho cả 30 outer folds; không tìm tham số lại.
30/30 kiểm tra sampler đạt; số mẫu test trước và sau bằng nhau.
có 0/30 lượt fit ghi cảnh báo; 0/30 lượt chạm max_iter=200.

cân bằng lớp không đảm bảo tăng macro f1 trên người mới. tham số ở đây được chọn cho pipeline chưa cân bằng; chưa kết luận đó là tham số tốt nhất cho pipeline có ros.

loss được tính trên train sau oversampling, không phải loss test. phân phối và số mẫu train đã đổi nên không diễn giải chênh lệch loss tuyệt đối với exp010 như bằng chứng cải thiện.

audit dựng lại preprocessing và sampler với đúng train và seed, không fit thêm mlp. runner chưa xuất sample_indices_ từ chính pipeline đã fit; đó là phần có thể bổ sung sau.

so sánh theo cặp cùng dữ liệu, subject, tham số và môi trường. mean/sd mẫu theo 15 người. chưa kiểm định thống kê, chưa khảo sát seed khác hoặc cách resampling khác. nếu khảo sát thêm cần chọn bằng inner cv, không tối ưu theo outer test.

exp020 sẽ nghiên cứu kiến trúc và regularization với lưới khác exp010. giữ nguyên các kết quả hiện tại.