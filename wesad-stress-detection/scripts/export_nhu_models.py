"""Xuất RF/MLP cho bàn giao; giữ riêng điểm LOSO và kiểm tra file đã fit."""
import argparse
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from time import perf_counter
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from threadpoolctl import threadpool_limits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_digest(path):
    """Text checksums remain valid with Git LF/CRLF checkout conversion."""
    raw = Path(path).read_bytes()
    if Path(path).suffix in {".json", ".yaml", ".csv", ".md", ".txt"}:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (args.output_root or root / "artifacts/models/group_submission").resolve()
    # Never overwrite a previously exported pipeline.
    for name in ("rf", "mlp"):
        if (out / name).exists():
            raise FileExistsError(f"Đã có {out / name}; chọn output-root mới để giữ bản cũ.")
    sys.path.insert(0, str(root / "src"))
    from wesad.data import load_feature_table
    from wesad.models import build_pipeline

    references = {
        "rf": ("EXP-005", "experiments/01_baselines/EXP-005_RF/config.yaml",
               "experiments/01_baselines/EXP-005_RF/results.csv"),
        "mlp": ("EXP-015", "experiments/03_balancing/EXP-015/seed_runs/20261122/seed_43/config.yaml",
                "experiments/03_balancing/EXP-015/seed_runs/20261122/seed_43/metrics_by_subject.csv"),
    }
    tuning_path = "experiments/02_tuning/EXP-010/seed_runs/20261122/seed_43/config.yaml"
    versions = {n: importlib.metadata.version(n) for n in
                ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "joblib", "PyYAML", "threadpoolctl")}
    evidence = out / "nhu_handoff"
    evidence.mkdir(parents=True, exist_ok=False)
    versions["python"] = platform.python_version()
    save_json(evidence / "environment.json", versions)
    (evidence / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items() if k != "python") + "\n", encoding="utf-8")
    records, journal = [], []
    for name, (exp, cfg_path, metrics_path) in references.items():
        cfg = yaml.safe_load((root / cfg_path).read_text(encoding="utf-8"))
        data_path, schema_path = root / cfg["features"], root / cfg["schema"]
        assert digest(data_path) == cfg["feature_sha256"], "Dữ liệu khác bộ đã chốt"
        frame, columns = load_feature_table(data_path, schema_path, cfg["expected_subjects"])
        assert len(columns) == 68 and len(frame) == 6189 and frame.subject_id.nunique() == 15
        X = frame[columns].astype(float)
        folder = out / name
        folder.mkdir()
        shutil.copy2(schema_path, folder / "feature_schema.json")
        shutil.copy2(root / cfg_path, folder / "reference_config.yaml")
        source_metrics = pd.read_csv(root / metrics_path)
        shutil.copy2(root / metrics_path, folder / "reference_loso_metrics.csv")
        # The fixture contains only features; labels/subject IDs never enter X.
        sample_indices = frame.groupby("label_original", sort=True).head(3).index
        sample = X.loc[sample_indices].reset_index(drop=True)
        sample.to_csv(folder / "example_features.csv", index=False)
        for task in ("binary", "multiclass"):
            print(f"Bắt đầu {name}/{task}", flush=True)
            started = perf_counter()
            y = (frame.label_original == 2).astype(int) if task == "binary" else frame.label_original.astype(int)
            labels = [0, 1] if task == "binary" else [1, 2, 3]
            pipe = build_pipeline(name, cfg)
            chosen, cv_score, cv_splits = {}, None, []
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                if name == "mlp":
                    tune_cfg = yaml.safe_load((root / tuning_path).read_text(encoding="utf-8"))
                    grid = tune_cfg["search_spaces"]["mlp"]
                    splits = list(GroupKFold(n_splits=3).split(X, y, frame.subject_id))
                    for train, valid in splits:
                        a, b = set(frame.subject_id.iloc[train]), set(frame.subject_id.iloc[valid])
                        assert not a & b
                        cv_splits.append({"train_subjects": sorted(a), "validation_subjects": sorted(b)})
                    search = GridSearchCV(pipe, grid, cv=splits, n_jobs=1, refit=True,
                        scoring=make_scorer(f1_score, labels=labels, average="macro", zero_division=0),
                        error_score="raise", return_train_score=True)
                    search.fit(X, y)
                    fitted, chosen, cv_score = search.best_estimator_, search.best_params_, float(search.best_score_)
                    table = pd.DataFrame(search.cv_results_)
                    table.drop(columns=[c for c in table if c.startswith("param_")]).to_csv(folder / f"selection_{task}.csv", index=False)
                else:
                    fitted = pipe.fit(X, y)
            assert list(fitted.feature_names_in_) == columns
            assert set(fitted.classes_) == set(labels)
            model_path = folder / f"{task}.joblib"
            with model_path.open("xb") as stream:
                joblib.dump(fitted, stream, compress=3)
            loaded = joblib.load(model_path)
            np.testing.assert_array_equal(fitted.predict(X), loaded.predict(X))
            predictions = loaded.predict(sample)
            probabilities = loaded.predict_proba(sample)
            assert probabilities.shape == (len(sample), len(labels))
            assert np.isfinite(probabilities).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-8)
            pd.DataFrame({"prediction": predictions}).to_csv(folder / f"expected_{task}.csv", index=False)
            rows = source_metrics[(source_metrics.task == task) & (source_metrics.model == name)]
            assert len(rows) == rows.subject_id.nunique() == 15
            reference_scores = {"macro_f1": float(rows.macro_f1.mean()), "macro_f1_std": float(rows.macro_f1.std(ddof=1)),
                                "balanced_accuracy": float(rows.balanced_accuracy.mean()),
                                "balanced_accuracy_std": float(rows.balanced_accuracy.std(ddof=1))}
            messages = [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]
            model = fitted.named_steps["model"]
            metadata = {
                "author": "như", "date": "23/11/2026", "model": name, "task": task, "experiment_id": exp,
                "reference_config": cfg_path, "reference_loso_source": metrics_path,
                "reference_loso_scores": reference_scores,
                "metric_scope": "Điểm tham chiếu từ LOSO 15 người của EXP, không phải điểm test của file refit toàn bộ dữ liệu.",
                "selection_caveat": "EXP/seed được chọn sau khi xem kết quả; chưa phải đánh giá độc lập của lựa chọn cuối.",
                "features": cfg["features"], "feature_sha256": digest(data_path), "feature_version": cfg["feature_version"],
                "feature_schema": "feature_schema.json", "feature_columns": columns,
                "n_windows_fit": len(frame), "subjects_fit": cfg["expected_subjects"],
                "fit_scope": "all_15_subjects; no held-out test for this final refit",
                "balance": "RandomOverSampler inside train pipeline" if name == "mlp" else "none",
                "pipeline_steps": list(fitted.named_steps), "chosen_params": chosen,
                "estimator_params": model.get_params(deep=False), "training_selection_macro_f1": cv_score,
                "selection_protocol": "GroupKFold(3), original 4 candidates from EXP010; ROS fit inside each training fold; refit full 15 subjects" if name == "mlp" else "fixed EXP005 defaults, no tuning",
                "selection_splits": cv_splits, "environment": versions, "labels": labels,
                "label_names": {"0":"Non-stress","1":"Stress"} if task == "binary" else {"1":"Baseline","2":"Stress","3":"Amusement"},
                "warnings": messages, "elapsed_seconds": round(perf_counter()-started, 4),
                "model_sha256": digest(model_path), "reload_matches_all_6189_predictions": True,
            }
            if name == "mlp":
                metadata["loss_curve"] = [float(v) for v in model.loss_curve_]
                metadata["n_iter"] = int(model.n_iter_)
                metadata["loss"] = float(model.loss_)
            save_json(folder / f"metadata_{task}.json", metadata)
            save_json(folder / f"config_{task}.json", {"model":name,"task":task,"seed":cfg["seed"],"balance":cfg["balance"],"chosen_params":chosen,"reference_config":cfg_path})
            records.append({"model":name,"task":task,"model_path":f"artifacts/models/group_submission/{name}/{task}.joblib",
                            "experiment_id":exp,**reference_scores,"loso_source":metrics_path,"n_subjects_loso":15,
                            "n_windows_fit":len(frame),"fit_scope":"all_15_subjects","score_scope":"reference_LOSO_not_final_refit_test"})
            journal.append({"date":"23/11/2026","author":"như","model":name,"task":task,
                            "action":"fit, lưu và nạp lại pipeline", "warnings":messages, "seconds":metadata["elapsed_seconds"]})
            print(f"Đã lưu {name}/{task}: {model_path.stat().st_size} bytes; {len(messages)} cảnh báo", flush=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "summary/model_summary_nhu.csv", index=False)
    save_json(evidence / "journal.json", journal)
    sources = ["src/wesad/models.py","src/wesad/data.py","src/wesad/utils.py"]
    with zipfile.ZipFile(evidence / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(root / source, source)
        archive.write(Path(__file__), "scripts/export_nhu_models.py")
    (evidence / "README.md").write_text('''# Bàn giao RF và MLP

Như, 23/11/2026. Bốn pipeline trong `../rf` và `../mlp` đã fit toàn bộ 6.189 cửa sổ của 15 subject.
RF dùng EXP005 mặc định (phần RF do Như thực hiện), không lấy EXP019 của Mai.
MLP dùng pipeline EXP015 seed43: median imputer → StandardScaler → RandomOverSampler → MLP.
Chốt tham số MLP bằng GroupKFold 3 nhóm trên toàn bộ dữ liệu dành cho refit, dùng bốn ứng viên gốc EXP010.
Các subject không giao nhau giữa train/validation. Không dùng một outer fold bất kỳ làm model cuối.

`../summary/model_summary_nhu.csv` chứa điểm LOSO tham chiếu. Điểm này không phải điểm test độc lập của pipeline refit.
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
''',encoding="utf-8")
    save_json(evidence / "checksums.json", {p.relative_to(out).as_posix():artifact_digest(p) for p in out.rglob("*") if p.is_file() and p != evidence / "checksums.json"})
    print("Xuất xong 4 pipeline. Cần chạy verify_nhu_models.py trong tiến trình Python mới.", flush=True)


if __name__ == "__main__":
    main()
