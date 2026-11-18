# exp015 — cân bằng lớp mlp

Như chạy mới ngày 16/11/2026, thuộc đồ án WESAD.
Notebook được chỉnh cách trình bày và kiểm tra lại các ô đọc kết quả ngày 18/11/2026; số đo vẫn thuộc lượt train ngày 16/11/2026.
Đối chứng là exp010; giữ đúng best_params theo task/subject, chỉ thêm
RandomOverSampler(seed=42) trên outer train. Không tuning lại, không resample test.

- Notebook làm việc: `notebooks/experiments/Nhu_EXP-015_MLP_balancing.ipynb` ở gốc project.
- Notebook trong thư mục này là bản tổng hợp có output và 4 hình.
- `results/`: dự đoán 12.378 dòng, 30 checkpoint, tham số, sampler audit, loss và so sánh.
- `results/conclusion.md`: nhận xét từ số đo thực tế.

Mở notebook, mặc định RUN_TRAIN=False chỉ đọc kết quả. Nếu không có output gốc,
notebook đọc bản đóng gói tại đây. Dữ liệu physiological68 vẫn giữ ngoài Git,
theo đúng đường dẫn và checksum trong config. Cần có gói exp010 để đối chiếu.

Từ thư mục `wesad-stress-detection`:

```powershell
# Lượt mới, tạo thư mục output riêng
& '.\.venv-g1\Scripts\python.exe' -u scripts/exp015_train.py

# Kiểm tra bản tổng hợp, không fit MLP
& '.\.venv-g1\Scripts\python.exe' scripts/exp015_train.py --verify-only experiments/03_balancing/EXP-015/results
```

Nếu bị ngắt, dùng `--resume-run outputs/<tên lượt đó>` với data, source,
config và môi trường không đổi. Không dùng resume trên bản tổng hợp: output_dir
trong config vẫn trỏ về output gốc.

Audit dựng lại preprocessing/ROS với cùng train và seed; chưa xuất sample_indices_
từ chính MLP đã fit. Việc này được ghi rõ, không nhận là đã quan sát trực tiếp
indices bên trong fitted pipeline của runner. Audit không fit thêm MLP.

Mean/sd tính theo 15 người, sd mẫu ddof=1. Chưa kiểm định thống kê hoặc khảo sát
seed khác. Tham số chọn cho pipeline chưa cân bằng chưa chắc tốt nhất khi thêm ROS.
Giữ nguyên điểm giảm, không chọn cấu hình theo outer test để ép tăng điểm.
