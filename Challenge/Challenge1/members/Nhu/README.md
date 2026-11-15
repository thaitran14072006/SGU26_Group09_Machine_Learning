# random forest — như

Ngày chạy đầu: 15/11/2026. Dữ liệu House Prices, log(SalePrice), bỏ Id.
Development/holdout 80/20 và 5 fold CV theo đúng các file trong `data/splits`, seed 7.

- `notebooks/15_11_2026_rf.ipynb`: notebook đã chạy, có output, hình và nhận xét.
- `src/rf_workflow.py`: pipeline, sáu phương án CV, phân tích sai số và dự đoán.
- `outputs/experiments/`: mỗi lần chạy tạo thư mục mới; xem `outputs/latest_run.txt`.
- `../../reports/individual/Nhu_RF_report.md`: báo cáo cá nhân.
- `../../results/submissions/submission_Nhu.csv`: 1.459 giá dự đoán, chưa nộp Kaggle.

Mở notebook và chọn Python đã cài `requirements.txt`, chạy từ trên xuống.
Notebook tìm Challenge1 từ thư mục gốc Machine-learning hoặc từ một thư mục con.
Nhật ký chung và bảng so sánh nhóm được cập nhật bởi `src/collect_group_logs.py`;
các dòng nhập từ thành viên khác có ghi nguồn, không phải lượt Như train lại.

Sáu phương án: RF cơ sở; xử lý NA theo nghĩa; thêm diện tích/tuổi nhà/tiện ích;
depth 12; leaf 3; depth 12 kết hợp leaf 3. Giữ 100 cây và seed 7.
Chọn bằng CV trước khi đánh giá holdout. Importance lấy từ train folds;
không chọn feature theo holdout. Pipeline cuối fit toàn bộ train và lưu joblib.

Đã kiểm tra điểm từng fold, 1.459 Id và dự đoán từ model nạp lại.
Đây là kiểm tra inference, chưa phải một lần train độc lập trên máy khác.
Có thể nâng cấp bằng ablation từng nhóm, permutation importance trong CV và
khảo sát max_features. Không đổi tham số theo điểm holdout.

`src/run_notebook.py` là tiện ích tạo và thực thi notebook mới bằng IPython;
không ghi đè notebook đã có kết quả. Muốn chạy lại, dùng notebook để giữ lịch sử
thư mục output riêng. Không đổi ngày cũ thành ngày chạy lại.
