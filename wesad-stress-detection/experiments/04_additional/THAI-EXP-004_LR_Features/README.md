# LR chọn đặc trưng

Mã riêng: THAI-EXP-004. EXP chung: EXP-016.

Mở `notebooks/experiments/Thai_EXP-004_working.ipynb`, chọn kernel `.venv` và chạy các cell từ trên xuống.
Notebook lưu cả output bảng và hình. Các fold đã hoàn tất được kiểm tra checksum trước khi dùng lại.

- Đối chứng: THAI-EXP-002, LR đủ 68 đặc trưng.
- Chọn k = 16, 32, 48 hoặc 68 bằng inner GroupKFold 3 phần; đánh giá bằng outer LOSO 15 người.
- Imputer, scaler và selector nằm trong pipeline được fit riêng theo train của từng fold.
- `r01/`: kiểm tra S2; `r02/`: kết quả đủ 15 người cho binary và multiclass.
- `experiments_logs.xlsx`: nhật ký riêng, gồm 4 dòng R01/R02 và ánh xạ EXP chung.
- Bằng chứng inner CV, đặc trưng đã chọn và dự đoán: `outputs/notebook_research/THAI-EXP-004/`.

Số liệu trong báo cáo là trung bình và độ lệch chuẩn giữa các người. Confusion matrix gộp các cửa sổ ngoài mẫu.
