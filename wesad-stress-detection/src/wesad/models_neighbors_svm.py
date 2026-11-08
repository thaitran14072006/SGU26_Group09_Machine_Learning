"""Pipeline KNN/SVM cho G3 (mặc định) và G4 (nested CV).

Đặt file trong src/wesad/ và thêm vào configs/experiment.yaml:

    model_factories:
      knn: wesad.models_neighbors_svm:build_pipeline
      svm: wesad.models_neighbors_svm:build_pipeline

Runner hiện có gọi factory bằng name=name, config=config. Giữ nguyên tên
bước ``model`` để search_spaces với model__n_neighbors, model__C, ... hoạt động.
Module chỉ tạo pipeline; LOSO, tuning và lưu kết quả do evaluation.py thực hiện.
Không chạy trực tiếp file này để huấn luyện.
"""

from collections.abc import Mapping

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_pipeline(name, config):
    """Tạo pipeline chưa fit, tương thích wesad.models.build_pipeline.

    Cấu hình dùng chung:
      seed: seed cho RandomOverSampler (mặc định 42).
      balance: KNN dùng oversampling; SVM dùng class_weight='balanced'.
      variance_threshold: None hoặc ngưỡng loại feature ít biến thiên.
      model_params: tham số estimator theo tên mô hình, không có prefix model__.

    model_params được áp dụng sau balance, giống factory chung của repo;
    vì vậy class_weight trong model_params.svm có thể ghi đè giá trị balanced.
    G3 giữ balance=false, không selection và không ghi đè tham số estimator.
    G4 dùng search_spaces trong YAML để tuning toàn pipeline trong inner CV.
    Mọi bước học dữ liệu chỉ được fit khi runner gọi fit trên train của fold.
    """
    if name not in {"knn", "svm"}:
        raise ValueError(f"Unsupported model {name!r}; choose 'knn' or 'svm'.")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping read from the experiment YAML.")

    if name == "knn":
        model = KNeighborsClassifier()
    else:
        # SVC mặc định dùng kernel='rbf'; không tự tăng C hoặc bật probability.
        model = SVC()

    balanced = config.get("balance", False)
    if balanced and name == "svm":
        model.set_params(class_weight="balanced")
    model.set_params(**config.get("model_params", {}).get(name, {}))

    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    threshold = config.get("variance_threshold")
    if threshold is not None:
        steps.append(("selection", VarianceThreshold(threshold=threshold)))
    if balanced and name == "knn":
        steps.append(("sampler", RandomOverSampler(random_state=int(config.get("seed", 42)))))
    steps.append(("model", model))
    return Pipeline(steps)
