BÀN GIAO MÔ HÌNH WESAD NHÓM 09

group_submission/ chứa knn/, svm/, rf/, mlp/, dummy/ và summary/. Các bảng tổng hợp nằm trong summary/, mở Tong_hop_model.txt để xem chung. Mỗi gói model có binary.joblib, multiclass.joblib, metadata_binary.json, metadata_multiclass.json, feature_schema.json. Hai model LR giữ tại artifacts/models/lr_candidate_20261122/ và dùng metadata.json chung.

Mở PowerShell tại thư mục project. Dùng Python 3.12.14. Nếu chưa có .venv thì tạo trước, sau đó cài requirements.lock.txt và mã nguồn:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e .

Ví dụ nạp RF, giữ đúng thứ tự feature:
import json
from pathlib import Path
import joblib
import pandas as pd
folder = Path('artifacts/models/group_submission/rf')
cols = json.loads((folder / 'feature_schema.json').read_text('utf-8'))['feature_columns']
X = pd.read_csv('app/examples/features.csv')[cols].astype(float)
model = joblib.load(folder / 'binary.joblib')
print(model.predict(X))

Binary dùng 0 Non-stress, 1 Stress. Multiclass dùng 1 Baseline, 2 Stress, 3 Amusement. X chỉ chứa feature, không chứa nhãn hay mã người. LR cần mã nguồn wesad.lr_workflow vì pipeline dùng lớp WeightedLR.

LR/RF/MLP/Dummy xuất bằng Python 3.12.14, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1. KNN/SVM của Nam ghi Python 3.13.5, scikit-learn 1.9.1, NumPy 2.5.3, pandas 2.3.3, SciPy 1.18.1. Phiên bản đầy đủ nằm trong metadata từng file. Cần kiểm tra lại khi đổi môi trường.

Lập danh mục, đối chiếu số và nạp thử từng model:
.\.venv\Scripts\python.exe scripts/build_model_summary.py
.\.venv\Scripts\python.exe scripts/build_model_summary.py --check
.\.venv\Scripts\python.exe scripts/check_submission.py --include-dummy

summary/model_summary.csv ghi mỗi model một dòng, có điểm và đường dẫn LOSO gốc. summary/model_summary_nhu.csv giữ bảng riêng của Như. summary/Tong_hop_model.txt tóm tắt số model, điểm và lần kiểm tra nạp gần nhất. score_mismatches.csv ghi các chỗ lệch để hỏi lại người xuất model. File chỉ có tiêu đề nghĩa là chưa thấy số lệch trong lần đối chiếu. submission_check.json lưu kết quả kiểm tra trong phiên Python mới.

Dummy là baseline để so sánh, không phải model đề xuất. Nhóm giữ most_frequent theo EXP-001. Khi cần nộp Dummy, chạy từng bài toán:
.\.venv\Scripts\python.exe scripts/export_dummy_model.py --task binary
.\.venv\Scripts\python.exe scripts/export_dummy_model.py --task multiclass
Script dùng xb, không ghi đè model có sẵn. Sau khi lưu, nạp lại và so predict trên 10 dòng đầu rồi cập nhật danh mục.

Điểm LOSO đánh giá cấu hình hoặc quy trình lựa chọn qua các fold theo người. Model bàn giao fit trên toàn bộ 15 người theo metadata. Chạy CSV mẫu chỉ xác nhận model nạp và dự đoán được, không tạo thêm điểm test độc lập. LR dùng điểm quy trình tuần tự, không lấy run C có điểm cao nhất. RF/MLP giữ nguồn tham chiếu và seed trong metadata.
