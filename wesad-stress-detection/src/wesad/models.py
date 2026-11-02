"""Tạo pipeline theo cấu hình; hỗ trợ factory riêng cho từng mô hình."""

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wesad.utils import import_callable

MODEL_NAMES = ("dummy", "lr", "knn", "svm", "rf", "mlp")


def build_pipeline(name, config):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name}; choose {MODEL_NAMES}.")
    factory = config.get("model_factories", {}).get(name)
    if factory:
        return import_callable(factory)(name=name, config=config)
    seed = int(config.get("seed", 42))
    balanced = config.get("balance", False)
    estimators = {
        "dummy": lambda: DummyClassifier(strategy="most_frequent"),
        "lr": lambda: LogisticRegression(),
        "knn": lambda: KNeighborsClassifier(),
        "svm": lambda: SVC(),
        "rf": lambda: RandomForestClassifier(random_state=seed, n_jobs=1),
        "mlp": lambda: MLPClassifier(random_state=seed, early_stopping=False),
    }
    model = estimators[name]()
    if balanced and name in {"lr", "svm", "rf"}:
        model.set_params(class_weight="balanced")
    model.set_params(**config.get("model_params", {}).get(name, {}))
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name not in {"dummy", "rf"}:
        steps.append(("scaler", StandardScaler()))
    if config.get("variance_threshold") is not None:
        steps.append(("selection", VarianceThreshold(config["variance_threshold"])))
    if balanced and name in {"knn", "mlp"}:
        steps.append(("sampler", RandomOverSampler(random_state=seed)))
    steps.append(("model", model))
    return Pipeline(steps)


def search_grid(name, config):
    grid = config.get("search_spaces", {}).get(name)
    if not grid:
        raise ValueError(f"No search space supplied for {name}.")
    return grid
