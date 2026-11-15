# BÁO CÁO CÁ NHÂN – NAM – SVR

## 1. Bài toán và giả thuyết
- Dự đoán SalePrice từ 79 đặc trưng, dùng nhãn log(SalePrice).
- Giả thuyết SVR RBF có thể học quan hệ phi tuyến; cần chuẩn hóa các biến số.

## 2. Dữ liệu / chia tập
- 1.460 dòng train, 1.459 dòng test theo đề bài.
- Development 80%, holdout 20%, random_state=7; 5-fold CV trên development.
- Ghi rõ Id split đã được nhóm duyệt.

## 3. EDA
- Chèn hình `eda_log_price.png`, `eda_missingness.png`.
- Nêu các quan sát thực tế sau khi chạy.

## 4. Bảng phiên bản
| Version | Giả thuyết/thay đổi | CV RMSE log (mean ± std) | Thời gian | Kết luận |
|---|---|---|---|---|
| V0 | SVR baseline | _chạy để điền_ | | |
| V1 | log feature transforms | _chạy để điền_ | | |
| V2 | feature selection/engineering | _chạy để điền_ | | |
| V3 | GridSearchCV | _chạy để điền_ | | |
| V4 | error analysis/OOF | _chạy để điền_ | | |
| V5 | chọn tốt nhất theo CV | _chạy để điền_ | | |

## 5. Phân tích sai số
- Chèn `v4_residuals.png` và tổng hợp sai số theo 4 nhóm giá.
- Nêu nhóm nhà nào dự đoán kém, lý do khả dĩ, giới hạn.

## 6. Holdout (chỉ đánh giá sau khi chọn cấu hình)
- RMSE log: _điền từ v5_holdout_report.json_
- MAE giá thực: _điền từ v5_holdout_report.json_

## 7. Sản phẩm và cách chạy lại
- Notebook EDA, V0–V5, experiment log, submission_Nam_SVR.csv.
- Commit/PR link: _điền_

## 8. Kết luận và hướng nghiên cứu tiếp
- Nêu điều cải thiện đáng kể nhất và các thử nghiệm không hiệu quả.
- Chỉ báo cáo số liệu thực.
