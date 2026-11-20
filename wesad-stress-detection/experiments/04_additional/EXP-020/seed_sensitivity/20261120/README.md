# exp020 — bổ sung ngày 20/11/2026

như · EXP chung: EXP-020

Chạy mới seed 43 và 44, giữ seed 42 làm kết quả chính. Mỗi lượt mới đủ 30 outer folds và 360 inner fits; không đổi lưới hay dữ liệu. Không chọn seed tốt nhất.

multiclass: Macro-F1 của seed 42/43/44 lần lượt là 0.6211 / 0.6145 / 0.6203. Khoảng biến động trung bình theo seed là 0.0065.

binary: Macro-F1 của seed 42/43/44 lần lượt là 0.8164 / 0.7927 / 0.8446. Khoảng biến động trung bình theo seed là 0.0520.

Xem summary_by_seed.csv, scores_by_subject_seed.csv, prediction_agreement.csv và hình trong figures. Notebook chính và notebook làm việc có phần bổ sung kèm output đã thực thi.
