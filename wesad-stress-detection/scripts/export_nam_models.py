"""Export Nam's 4 deployable WESAD pipelines: KNN EXP-017, SVM EXP-018.
Run from any working directory: python scripts/export_nam_models.py --dry-run
Then: python scripts/export_nam_models.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Avoid OpenBLAS memory/thread errors on constrained Windows machines.
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wesad.data import SUBJECTS, load_feature_table  # noqa: E402
from run_exp017 import make_pipeline as knn_pipeline  # noqa: E402
from run_exp018 import make_pipeline as svm_pipeline  # noqa: E402

DATA_REL = "data/features/g2_physiological68_group15_20261107/all_subjects.csv"
SCHEMA_REL = "data/features/g2_physiological68_group15_20261107/feature_schema.json"
EXPECTED_DATA_SHA = "2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba"
EXPECTED_SCHEMA_SHA = "8730ca9189547bd54e0ec2cec5f574a5e2a989b64f6ea6b0a8dd4a186669f998"
OUT_REL = "artifacts/models/group_submission"

SOURCES = {
    "knn": {
        "experiment_id": "EXP-017", "variant": "knn_top30",
        "summary": "experiments/04_additional/EXP-017_KNN/summary.csv",
        "runner": "scripts/run_exp017.py", "maker": knn_pipeline,
    },
    "svm": {
        "experiment_id": "EXP-018", "variant": "svm_top30",
        "summary": "experiments/04_additional/EXP-018_SVM/summary.csv",
        "runner": "scripts/run_exp018.py", "maker": svm_pipeline,
    },
}
TASK_LABELS = {"binary": [0, 1], "multiclass": [1, 2, 3]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versions() -> dict:
    names = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "scipy", "joblib", "PyYAML"]
    d = {}
    for name in names:
        try:
            d[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            d[name] = None
    return d


def read_scores() -> dict:
    all_scores = {}
    for algorithm, spec in SOURCES.items():
        path = ROOT / spec["summary"]
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu summary LOSO dùng để đối chiếu: {path}")
        df = pd.read_csv(path)
        expected_cols = {"task", "variant", "n_subjects", "n_test", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "balanced_accuracy_std"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Sai schema summary.csv của {algorithm}: {sorted(expected_cols-set(df.columns))}")
        for task in TASK_LABELS:
            records = df.loc[(df.task == task) & (df.variant == spec["variant"])]
            if len(records) != 1:
                raise ValueError(f"Thiếu hoặc trùng dòng {algorithm}/{task}/{spec['variant']}")
            row = records.iloc[0]
            if int(row.n_subjects) != 15 or int(row.n_test) != 6189:
                raise ValueError(f"LOSO chưa đủ 15 người/6189 cửa sổ: {algorithm}/{task}")
            all_scores[(algorithm, task)] = {k: float(row[k]) for k in (
                "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "balanced_accuracy_std")}
    return all_scores


def check_inputs() -> tuple[pd.DataFrame, list[str], str, str, dict]:
    data_path, schema_path = ROOT / DATA_REL, ROOT / SCHEMA_REL
    for path in [data_path, schema_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu dữ liệu cục bộ: {path}; file lớn không có trong ZIP/GitHub. Dùng bản G2 đã khóa ở máy.")
    dsha, ssha = sha256_file(data_path), sha256_file(schema_path)
    if dsha != EXPECTED_DATA_SHA:
        raise ValueError(f"Dữ liệu không khớp SHA256 đã khóa: {dsha} (mong đợi {EXPECTED_DATA_SHA})")
    if ssha != EXPECTED_SCHEMA_SHA:
        raise ValueError(f"Schema không khớp SHA256 của EXP-017: {ssha}")
    frame, columns = load_feature_table(data_path, schema_path, expected_subjects=SUBJECTS)
    if len(columns) != 68 or len(frame) != 6189:
        raise ValueError(f"Yêu cầu 68 features và 6189 windows; đang có {len(columns)}, {len(frame)}")
    if frame.subject_id.astype(str).nunique() != 15:
        raise ValueError("Phải có đủ 15 subject")
    vals = frame[columns].astype(float).to_numpy()
    if np.isinf(vals).any():
        raise ValueError("Các đặc trưng chứa vô cực")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema["feature_columns"] != list(columns):
        raise ValueError("Sai thứ tự cột schema")
    return frame, columns, dsha, ssha, schema


def validate_pipeline(pipe, algorithm: str, X: pd.DataFrame, y: pd.Series) -> None:
    steps = list(pipe.named_steps)
    if steps != ["imputer", "scaler", "selection", "model"]:
        raise ValueError(f"Sai steps {algorithm}: {steps}")
    if pipe.named_steps["selection"].k != 30:
        raise ValueError("Phải chọn đúng top-30")
    if algorithm == "knn":
        model = pipe.named_steps["model"]
        if (model.n_neighbors, model.weights, model.metric, model.p) != (5, "uniform", "minkowski", 2):
            raise ValueError("Sai tham số EXP-017")
    else:
        model = pipe.named_steps["model"]
        if (model.kernel, model.C, model.gamma, model.class_weight) != ("rbf", 1.0, "scale", None):
            raise ValueError("Sai tham số EXP-018")
    if pipe.named_steps["selection"].get_support(indices=True).size != 30:
        raise ValueError("Selector chưa fit đúng 30 features")
    if int(pipe.n_features_in_) != 68:
        raise ValueError("Pipeline phải nhận 68 features")
    predictions = pipe.predict(X.head(20))
    if len(predictions) != min(len(X),20):
        raise ValueError("Sai số dự đoán")
    if not set(predictions).issubset(set(y.unique())):
        raise ValueError("Dự đoán ngoài nhãn")


def export(force: bool, dry_run: bool) -> None:
    frame, cols, dsha, ssha, original_schema = check_inputs()
    scores = read_scores()
    out = ROOT / OUT_REL
    if not force and any((out / algo / f"{task}.joblib").exists() for algo in SOURCES for task in TASK_LABELS):
        raise FileExistsError("Đã tồn tại model trong thư mục bàn giao. Dùng --force nếu đã kiểm tra và muốn thay thế.")

    print(f"[OK] G2: {len(frame)} cửa sổ; {frame.subject_id.nunique()} người; {len(cols)} đặc trưng")
    for algorithm, task in scores:
        print(f"[LOSO] {algorithm}/{task}: Macro-F1={scores[(algorithm,task)]['macro_f1_mean']:.4f}; BA={scores[(algorithm,task)]['balanced_accuracy_mean']:.4f}")
    if dry_run:
        print("[DRY RUN] Không huấn luyện hoặc tạo mô hình.")
        return

    X = frame[cols].astype(float)
    labels_original = frame.label_original.astype(int)
    if labels_original.isna().any() or set(labels_original) != {1, 2, 3}:
        raise ValueError("Nhãn gốc phải là 1,2,3")
    now = datetime.now(timezone.utc).isoformat()
    staged_parent = Path(tempfile.mkdtemp(prefix="nam_export_", dir=str(ROOT)))
    try:
        rows = []
        for algorithm, spec in SOURCES.items():
            folder = staged_parent / algorithm
            folder.mkdir(parents=True, exist_ok=True)
            feature_schema = {
                "feature_columns": list(cols), "n_input_features": 68,
                "source_schema_sha256": ssha, "schema_version": original_schema.get("schema_version", "physiological_68_v1"),
                "selected_features": 30,
                "note": "Input always has 68 columns in this order. SelectKBest is INSIDE the fitted pipeline."
            }
            (folder/"feature_schema.json").write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")
            for task, labels in TASK_LABELS.items():
                y = (labels_original == 2).astype(int) if task == "binary" else labels_original
                pipe = spec["maker"](30)  # EXACT experiment pipeline imported from repo runner.
                print(f"[FIT] {algorithm.upper()} / {task}: 6189 windows, 15 subjects...", flush=True)
                with threadpool_limits(limits=1):
                    pipe.fit(X, y)
                    validate_pipeline(pipe, algorithm, X, y)
                path = folder / f"{task}.joblib"
                joblib.dump(pipe, path, compress=3)
                reloaded = joblib.load(path)
                np.testing.assert_array_equal(pipe.predict(X.head(20)), reloaded.predict(X.head(20)))
                if set(reloaded.classes_.tolist()) != set(labels):
                    raise ValueError(f"Wrong class labels {algorithm}/{task}: {reloaded.classes_}")
                selected_idx = reloaded.named_steps["selection"].get_support(indices=True)
                selected_names = [cols[i] for i in selected_idx]
                score = scores[(algorithm, task)]
                meta = {
                    "owner": "Nam", "model": algorithm, "task": task,
                    "experiment_id": spec["experiment_id"], "variant": spec["variant"],
                    "pipeline_source": spec["runner"], "pipeline_source_sha256": sha256_file(ROOT/spec["runner"]),
                    "created_utc": now, "data_path": DATA_REL, "data_sha256": dsha,
                    "schema_path": SCHEMA_REL, "schema_sha256": ssha,
                    "n_windows_fit": len(frame), "n_subjects_fit": 15,
                    "subjects_fit": list(SUBJECTS),
                    "input_feature_count": len(cols), "selected_feature_count": len(selected_names),
                    "selected_feature_names": selected_names,
                    "steps": list(reloaded.named_steps), "model_params": reloaded.named_steps["model"].get_params(),
                    "label_mapping": ({"0":"Non-Stress", "1":"Stress"} if task=="binary" else {"1":"Baseline", "2":"Stress", "3":"Amusement"}),
                    "labels": labels, "fit_scope": "full 15 subjects: deployment fit, NOT independent evaluation",
                    "loso_source": spec["summary"], "loso_mean_by_subject": score,
                    "python": platform.python_version(), "package_versions": versions(),
                    "joblib_sha256": sha256_file(path),
                    "reloaded_prediction_check": "passed on first 20 training rows (serialization smoke test only)"
                }
                (folder / f"metadata_{task}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),encoding="utf-8")
                rows.append({
                    "model": algorithm, "task": task, "artifact_path": f"{OUT_REL}/{algorithm}/{task}.joblib",
                    "experiment_id": spec["experiment_id"], "variant": spec["variant"],
                    "macro_f1_mean_loso": score["macro_f1_mean"],
                    "macro_f1_std_loso": score["macro_f1_std"],
                    "balanced_accuracy_mean_loso": score["balanced_accuracy_mean"],
                    "balanced_accuracy_std_loso": score["balanced_accuracy_std"],
                    "loso_source": spec["summary"], "n_loso_subjects": 15,
                    "training_windows": len(X), "training_subjects": 15, "data_sha256": dsha,
                    "model_sha256": meta["joblib_sha256"]
                })
        pd.DataFrame(rows).to_csv(staged_parent/"nam_model_summary.csv",index=False,encoding="utf-8-sig")
        # Commit only after ALL four models pass validation.
        out.mkdir(parents=True, exist_ok=True)
        for algorithm in SOURCES:
            dest = out / algorithm
            if dest.exists():
                if not force:
                    raise FileExistsError(f"Thư mục đã tồn tại: {dest}; không ghi đè nếu thiếu --force")
                shutil.rmtree(dest)
            shutil.move(str(staged_parent/algorithm), str(dest))
        shutil.move(str(staged_parent/"nam_model_summary.csv"), str(out/"nam_model_summary.csv"))
        print(f"[SUCCESS] 4 models + schema + metadata + summary at {out}")
        print("[NEXT] python scripts/verify_nam_models.py")
    finally:
        shutil.rmtree(staged_parent, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate G2 data + verified LOSO sources without fitting")
    parser.add_argument("--force", action="store_true", help="Replace existing handoff files after confirmation")
    args = parser.parse_args()
    export(force=args.force, dry_run=args.dry_run)
