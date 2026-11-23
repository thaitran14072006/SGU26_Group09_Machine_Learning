# WESAD Streamlit — 15 subjects, Binary & Multiclass

## Tích hợp

Chép **thư mục `demo_streamlit/`** vào gốc repo `wesad-stress-detection/` (cùng cấp `src/`, `scripts/`, `data/`). File ZIP không kèm dữ liệu WESAD hoặc model thật. Không cần chỉnh notebook EXP-003/017.

```powershell
cd C:\Users\thanh\source\repos\ML\Machine-learning\wesad-stress-detection
conda activate wesad31214
python -m pip install -r demo_streamlit/requirements.txt
streamlit run demo_streamlit/app.py
```

Ứng dụng hiển thị 15 người S2–S11 và S13–S17, bảng đầy đủ 68 features, thống kê, biểu đồ và tải dữ liệu CSV. Dữ liệu mặc định:

`data/features/g2_physiological68_group15_20261107/all_subjects.csv`

## Kết quả LOSO

Nhập đường dẫn của `predictions.csv` thực tế tại sidebar. Mặc định tìm kết quả EXP-017 ở `experiments/04_improvements/EXP-017_KNN_Feature_Selection/predictions.csv`. Nếu thực nghiệm xuất `run_outputs/predictions.csv`, hãy sửa đường dẫn trên sidebar. Cần có các cột `subject_id`, `task`, `y_true`, `y_pred`; tùy chọn `variant`, `window_id`. Đánh giá trên 15 subject bằng dự đoán LOSO khác với dự đoán lại trên dữ liệu huấn luyện.

## Hai model demo (Binary + Multiclass)

Tạo hai mô hình KNN + imputer + StandardScaler + SelectKBest(30) từ dữ liệu thật:

```powershell
python demo_streamlit/train_models.py
```

Kết quả:
- `demo_streamlit/models/knn_top30_binary.joblib`
- `demo_streamlit/models/knn_top30_multiclass.joblib`

Mô hình mặc định là KNN k=5, uniform, Euclidean (theo mẫu EXP-017). Đây là pipeline **tái huấn luyện cho triển khai**, không phải 15 model LOSO. Nếu kết quả 20 EXP sau này chỉ ra mô hình khác tốt hơn, cần cập nhật `train_models.py` và kiểm tra lại feature ordering.

**Lưu ý:** Hai model được train trên toàn bộ 15 người. Chúng dùng để minh họa tính năng dự đoán hoặc suy luận trên dữ liệu mới; không được dùng điểm dự đoán lại của 15 người làm kết quả đánh giá. Phần "Đánh giá LOSO" dùng predictions xuất từ mô hình train trên 14 người tương ứng. Không tải joblib từ nguồn không tin cậy vì `joblib.load` có thể thực thi mã.

## Lỗi thường gặp

- `Không tìm thấy ... all_subjects.csv`: đảm bảo dataset ở đúng thư mục hoặc nhập đường dẫn trong sidebar.
- `Nhận diện ... thay vì 68 features`: kiểm tra các cột ngoài nhãn, id; `train_models.py` cố ý dừng nếu không đúng 68.
- `Chưa có predictions.csv`: chạy EXP-017 Main Run hoặc nhập đường dẫn có thật ở sidebar.
- `Chưa có đủ hai mô hình`: chạy `python demo_streamlit/train_models.py`.
- Dữ liệu bị chậm: chọn một người hoặc vài cửa sổ; biểu đồ chỉ vẽ 500 dòng đầu.

## Commit khuyến nghị

Chỉ commit `demo_streamlit/app.py`, `common.py`, `train_models.py`, `requirements.txt`, `.gitignore`, `README.md`. Không commit dữ liệu WESAD gốc và không cần commit model joblib nếu repo chính sách không cho phép.

## Bản sửa tương thích CSV G2 (23/11/2026)

CSV 68 đặc trưng có thể kèm 4 cột metadata `start_s`, `end_s`, `label_original`, `label_fraction`. Các cột này **không** đi vào mô hình. Cột `label_original` là nhãn multiclass (1=Baseline, 2=Stress, 3=Amusement), và binary được suy ra bằng `label_original == 2`. Không chỉnh sửa CSV nguồn.

Trong sidebar chọn **Tất cả 15 người** để xem toàn bộ dữ liệu; chọn S5 sẽ chỉ hiển thị S5. Nếu huấn luyện trước bị lỗi, chạy lại `python demo_streamlit/train_models.py` để sinh hai file joblib.

## Bản cập nhật tổng hợp dự đoán (23/11/2026)

- Phần Tổng quan giữ hai biểu đồ **nhãn thật** Binary và Multiclass cạnh nhau.
- Tab Dự đoán **không còn chọn xem hoặc hiển thị từng cửa sổ riêng**; thay bằng hai bảng thống kê Binary và Multiclass đặt cạnh nhau.
- Mỗi bảng có cột **Trạng thái, Số cửa sổ, Tỷ lệ (%)**. Tỷ lệ tính trên số cửa sổ vừa dự đoán, không phải xác suất hay độ chính xác.
- Ô **Số cửa sổ cần dự đoán** tự giới hạn theo dữ liệu hiện đang lọc: chọn toàn bộ 15 người thì tối đa số dòng của toàn bộ CSV (ví dụ 6.189); chọn một người thì tối đa số dòng của người đó. Mặc định 100 cửa sổ, có thể nhập số tối đa để dự đoán hết.
- Kết quả chỉ hiển thị sau khi bấm nút **Chạy dự đoán cả hai**. Chuyển người, số cửa sổ, dữ liệu hoặc tệp mô hình sẽ không hiển thị kết quả cũ.
- Nút tải `demo_predictions.csv` vẫn xuất chi tiết từng hàng để đối soát, dù không hiện bảng từng cửa sổ trong UI.
- Hai pipeline triển khai đã train trên toàn bộ 15 người chỉ dùng cho minh họa; **không dùng dự đoán lại trên dữ liệu train để báo cáo hiệu suất**. Hãy dùng kết quả LOSO ở tab riêng.

**Cập nhật:** Sao chép đè thư mục `demo_streamlit` vào gốc `wesad-stress-detection`. Nếu dữ liệu hoặc file LOSO nằm ở đường dẫn khác, chỉnh đường dẫn trong sidebar.
