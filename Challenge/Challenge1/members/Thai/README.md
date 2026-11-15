# Thái — House Prices

15 thí nghiệm Ridge và ElasticNet, từ `Thai_EXP-001` đến `Thai_EXP-015`.
Notebook đặt tên theo ngày 01–13/11/2026. Ngày 12 và 13 có hai bản,
phân biệt bằng `12_01`, `12_02`, `13_01`, `13_02`.

Chia development/holdout 80/20, seed 7. Chọn cấu hình bằng 5-fold CV trên
development. Chỉ đánh giá holdout sau khi chọn cấu hình. Mỗi fold tự fit
điền thiếu, one-hot và chuẩn hóa. Mục tiêu là `log(SalePrice)`.

- `notebooks/`: 15 notebook đã chạy, có kết quả và nhận xét.
- `src/thai_experiments.py`: cấu hình, pipeline, chạy CV và cập nhật nhật ký.
- `outputs/experiments/`: cấu hình, kết quả JSON, điểm từng fold và OOF.
- `outputs/figures/`: biểu đồ EDA và sai số.
- `outputs/models/best_pipeline.joblib`: pipeline đã fit toàn bộ train.
- `outputs/predictions/submission_Thai.csv`: dự đoán 1.459 nhà trong test.
- `../../reports/individual/nhat_ky_thuc_nghiem.xlsx`: nhật ký Excel có liên kết minh chứng.
- `../../reports/individual/Thai_report.md`: phương pháp và kết quả cuối.

## Chạy lại

Môi trường lần chạy đầu: Python 3.13.9, scikit-learn 1.7.2. Cần numpy,
pandas, matplotlib, joblib và nbformat. Mở notebook rồi chọn Run All.
Mỗi lần chạy ghi thêm một dòng vào CSV, giữ các lần chạy trước.
Excel được cập nhật bằng Node.js và `@oai/artifact-tool` (có trong runtime
ChatGPT Work). Nếu thiếu phần này, CSV vẫn được lưu và có thông báo lỗi Excel.

Từ thư mục Challenge1:

```powershell
python members/Thai/src/run_all.py
node members/Thai/src/export_log.mjs
```

`run_all.py` tạo và chạy notebook còn thiếu, giữ notebook đã có.
Muốn chạy lại một thí nghiệm đã có, mở notebook tương ứng.

Ngày thực hiện theo mốc người dùng quy định; thời điểm chạy máy lưu riêng
trong JSON và CSV. Kết quả là số đo đã chạy, chưa có điểm Kaggle.
Mã nguồn trước khi bổ sung xuất Excel được giữ ở
`outputs/experiments/source_at_initial_run.py` để đối chiếu SHA256 trong JSON.
