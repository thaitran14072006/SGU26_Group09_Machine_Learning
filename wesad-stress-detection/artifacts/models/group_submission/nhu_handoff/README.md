# Bàn giao RF và MLP

Như, 23/11/2026. Bốn pipeline trong `../rf` và `../mlp` đã fit toàn bộ 6.189 cửa sổ của 15 subject.
RF dùng EXP005 mặc định (phần RF do Như thực hiện), không lấy EXP019 của Mai.
MLP dùng pipeline EXP015 seed43: median imputer → StandardScaler → RandomOverSampler → MLP.
Chốt tham số MLP bằng GroupKFold 3 nhóm trên toàn bộ dữ liệu dành cho refit, dùng bốn ứng viên gốc EXP010.
Các subject không giao nhau giữa train/validation. Không dùng một outer fold bất kỳ làm model cuối.

`../model_summary_nhu.csv` chứa điểm LOSO tham chiếu. Điểm này không phải điểm test độc lập của pipeline refit.
Việc chọn EXP/seed đã xem kết quả test có thể tạo thiên lệch chọn lựa. Không gắn điểm train hoặc CV chọn tham số thành điểm LOSO mới.

Mỗi thư mục model có schema 68 cột theo đúng thứ tự, metadata từng bài toán, config, CSV mẫu và dự đoán mong đợi.
Giữ Python 3.12.14 và phiên bản thư viện trong requirements.txt. Chỉ nạp joblib từ gói tin cậy.

Nạp và dự đoán (chạy từ wesad-stress-detection):
```python
import json, joblib, pandas as pd
from pathlib import Path
bundle = Path("artifacts/models/group_submission/mlp")  # hoặc rf
schema = json.loads((bundle / "feature_schema.json").read_text(encoding="utf-8"))
X = pd.read_csv(bundle / "example_features.csv")[schema["feature_columns"]]
model = joblib.load(bundle / "binary.joblib")  # hoặc multiclass.joblib
print(model.predict(X))
```

Xuất lại sang thư mục MỚI:
```powershell
python scripts/export_nhu_models.py --output-root artifacts/models/group_submission_new
```
Kiểm tra trong tiến trình mới:
```powershell
python scripts/verify_nhu_models.py
```

Mai ghép model_summary_nhu.csv vào danh mục chung khi nhận đủ thành viên. Không ghi đè model_summary.csv của Mai và không thay bundle LR đang dùng trong demo.
