# exp020 — kiến trúc và regularization của mlp

như · 18/11/2026 · EXP chung: EXP-020

Câu hỏi: mạng nhỏ hơn và regularization mạnh hơn có cải thiện kết quả so với
EXP-010 không? Lưới mới: hidden sizes `[16]` / `[32,16]`, alpha `0.01` / `0.1`.
Giữ learning rate 0.001, max_iter 200, không early stopping, không ROS.

Đã train ba lượt độc lập cùng seed 42, mỗi lượt đủ 15 người cho cả binary và
multiclass: tổng 1.080 inner fits và 90 outer refits. R01 là lượt chính;
R02/R03 kiểm tra tái lập, không chọn lượt có điểm tốt nhất. Dự đoán của cả ba
lượt giống nhau. Đây chưa phải kiểm tra độ nhạy với seed khác.

| task | macro f1, mean ± sd | balanced accuracy, mean ± sd |
|---|---:|---:|
| binary | 0.8164 ± 0.1754 | 0.8356 ± 0.1643 |
| multiclass | 0.6211 ± 0.1150 | 0.6609 ± 0.1038 |

Mean/sd mẫu theo 15 subject. Chọn tham số bằng 3 inner GroupKFold, imputation
và scaling học bên trong train fold. Không dùng outer test chọn tham số.
Macro-F1 tăng 0.0345 ở binary và 0.0024 ở multiclass so với EXP-010; không
tăng ở mọi người. Điểm train vẫn 1.0 nên khoảng cách train/test còn lớn.
Không có cảnh báo được ghi lại trong 30 outer searches của R01.

Notebook làm việc: `notebooks/experiments/Nhu_EXP-020_MLP_architecture.ipynb`.
Bản chốt trong thư mục này có đủ 10 mục, output thật và ảnh. Mặc định
`RUN_TRAIN=False`, chỉ đọc kết quả; không tự train khi mở notebook.

Từ thư mục `wesad-stress-detection`, với môi trường train đã khóa:

```powershell
& '.\.venv-g1\Scripts\python.exe' -X utf8 -u scripts/exp020_train.py
& '.\.venv-g1\Scripts\python.exe' -X utf8 scripts/exp020_train.py --verify-only experiments/04_additional/EXP-020/results
```

Lệnh train tạo output mới, không ghi đè bản chốt. `--resume-run outputs/<lượt>`
chỉ dùng tiếp lượt bị ngắt khi source/config/môi trường không đổi; không dùng
trên bản đóng gói vì output_dir vẫn lưu đường dẫn gốc.

`scripts/exp020_report.py --runs <R01> <R02> <R03>` kiểm tra ba lượt, đo thêm
train/test bằng 30 refits cố định tham số, vẽ và đóng gói. Chỉ chạy khi EXP-020
chưa có; không tự ghi đè lịch sử. Phụ thuộc plotting là matplotlib. Lần này
môi trường train thiếu matplotlib nên dùng thư viện đã có trên ổ D, thêm sau
site-packages train để giữ nguyên phiên bản ML. Xem `results/report_environment.json`.
Lệnh thực tế dùng để bổ sung thư viện có sẵn và chạy report:

```powershell
& '.\.venv-g1\Scripts\python.exe' -X utf8 -c "import sys,runpy; sys.path.append(r'D:/venvs/wesad312/Lib/site-packages'); sys.argv=['scripts/exp020_report.py','--runs',r'<R01>',r'<R02>',r'<R03>']; runpy.run_path(sys.argv[0],run_name='__main__')"
```

`scripts/exp020_notebook.py` tạo và thực thi các ô đọc kết quả bằng Python
hiện tại, thu stdout/ảnh thật. Script này cũng cần matplotlib. Chạy với môi
trường train và phần plotting như trên để tái tạo output notebook.

Các minh chứng chính: `results/verification.json`, `repeatability.csv`,
`repeatability_verification.json`, checkpoint từng fold, dự đoán,
`convergence.csv`, `inner_search_all.csv`, `comparison_summary.csv`,
`train_test/verification.json` và source snapshots.
Biểu đồ train/test theo subject là lượt refit bổ sung; cả 30 dự đoán test
khớp lượt nested gốc. Loss theo epoch là train objective có regularization,
không phải validation/test loss. Không so loss tuyệt đối giữa alpha khác nhau.

Còn có thể mở rộng grid bằng inner CV, thử seed khác, tách ảnh hưởng alpha
và kiến trúc, thêm validation theo nhóm và kiểm định. Đây là so sánh thăm dò
trên benchmark đã xem ở các EXP trước, chưa có final holdout độc lập.
Nguồn EXP-006 có phiên bản pandas khác; EXP-010 cùng môi trường train hiện tại.
Xem `launch_history.md` về lỗi trùng tên output và cách sửa thực tế.

Mã commit trong provenance là HEAD lúc train; thay đổi EXP-020 vẫn chưa commit/push.

## bổ sung ngày 20/11/2026

Đã chạy thêm seed 43 và 44, mỗi seed đủ 30 outer folds và 360 inner fits. Seed 42 vẫn là kết quả chính, không chọn seed có điểm cao nhất. Xem [kết quả theo seed](seed_sensitivity/20261120/README.md) và phần bổ sung trong hai notebook.

Ảnh bổ sung: [so sánh seed](figures/seed_macro_f1_by_seed.png), [điểm từng người](figures/seed_subject_scores_by_seed.png).

## bổ sung ngày 22/11/2026

Đã chạy thêm seed 45 và 46, mỗi seed đủ 30 outer folds và 360 inner fits. Seed 42 vẫn là kết quả chính, không chọn seed có điểm cao nhất. Xem [kết quả theo seed](seed_sensitivity/20261122/README.md) và phần bổ sung trong hai notebook.
