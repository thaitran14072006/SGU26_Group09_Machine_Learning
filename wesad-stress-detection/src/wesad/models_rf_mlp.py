"""Nhu G3: default RF/MLP with train-fold preprocessing; no G4 tuning."""

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def build_default_pipeline(name, config):
    """Shared runner factory. Keep estimator defaults except seed and RF jobs.

    Imputer/scaler fit inside each training fold. Retain empty training columns
    using SimpleImputer's documented fallback rather than changing feature order.
    """
    if name not in {"rf", "mlp"}:
        raise ValueError("Nhu's G3 factory supports rf and mlp only.")
    if (config.get("nested", False) or config.get("balance", False)
            or config.get("variance_threshold") is not None
            or config.get("model_params", {}).get(name)):
        raise ValueError("G3 default run cannot tune, balance, select or override model parameters.")
    if config.get("seed", 42) != 42:
        raise ValueError("G3 defaults use seed=42.")
    model = (RandomForestClassifier(random_state=42, n_jobs=1) if name == "rf"
             else MLPClassifier(random_state=42))
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
