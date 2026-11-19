> **NGHIÊN CỨU BỔ SUNG NGOÀI DANH MỤC 20 HƯỚNG.** Không phải EXP-005 (RF baseline), không tính vào tiến độ EXP-001–020. Kết quả đã chạy được giữ để tham khảo; không tự tiếp tục chuỗi THAI-EXP-006–010.

# LR regularization

Mã riêng: THAI-EXP-005. Thí nghiệm mở rộng, chưa gán EXP chung.

Mở `notebooks/experiments/Thai_EXP-005_working.ipynb`, chọn kernel `.venv` và chạy từ trên xuống.

- Đối chứng: THAI-EXP-002, LR mặc định C=1 với đủ 68 đặc trưng.
- Grid: C = 0.001, 0.01, 0.1 hoặc 1.0; chọn bằng inner GroupKFold 3 phần, Macro-F1.
- Outer LOSO 15 người; binary và multiclass. Giữ nguyên preprocessing và các tham số LR còn lại.
- `r01/`: kiểm tra S2; `r02/`: kết quả 30 fold; `experiments_logs.xlsx`: nhật ký riêng.
- Bằng chứng lựa chọn C theo từng fold trong `outputs/notebook_research/THAI-EXP-005/`.
- Đọc kết quả như nghiên cứu mở rộng sau khi đã xem test; không sửa grid bằng điểm outer test.
