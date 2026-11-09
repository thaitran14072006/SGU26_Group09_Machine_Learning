"""Đánh giá LOSO, tuning theo nhóm và lưu kết quả từng fold."""

from pathlib import Path
import hashlib
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, make_scorer)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from wesad.data import SUBJECTS, load_feature_table
from wesad.models import build_pipeline, search_grid
from wesad.utils import (code_hash, environment, file_hash, fingerprint, provenance,
                         log_event, read_json, resolve_path, write_csv, write_json, PROJECT_ROOT)


def task_target(frame, task):
    if task == "multiclass":
        return frame.label_original.astype(int), [1, 2, 3]
    if task == "binary":
        return (frame.label_original == 2).astype(int), [0, 1]
    raise ValueError("Task must be multiclass or binary.")


def grouped_inner_splits(X, y, groups, n_splits):
    if len(set(groups)) < n_splits or n_splits < 2:
        raise ValueError("Not enough outer-train subjects for grouped inner CV.")
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    description = []
    for train, validation in splits:
        a, b = set(groups.iloc[train]), set(groups.iloc[validation])
        if a & b:
            raise ValueError("Subject leakage in inner CV.")
        description.append({"train_subjects": sorted(a), "validation_subjects": sorted(b)})
    return splits, description


def run_experiment(config, resume=False, outer_subjects=None, allow_partial=False):
    table_path = resolve_path(config["features"])
    schema = resolve_path(config["schema"]) if config.get("schema") else None
    frame, columns = load_feature_table(table_path, schema)
    expected = set(config.get("expected_subjects", SUBJECTS))
    available = set(frame.subject_id)
    if not expected or expected - set(SUBJECTS):
        raise ValueError("Invalid expected_subjects.")
    if not allow_partial and available != expected:
        raise ValueError("Main runs require all expected subjects. Use --allow-partial only for checks.")
    if available - expected or len(available) < 2:
        raise ValueError("Need at least two subjects within the expected subject list.")
    selected = set(outer_subjects) if outer_subjects is not None else available
    if not selected or selected - available:
        raise ValueError("Requested outer subjects are not present in the table.")
    tasks, models = config.get("tasks", ["multiclass", "binary"]), config.get("models", ["dummy", "lr"])
    if not tasks or len(tasks) != len(set(tasks)) or not models or len(models) != len(set(models)):
        raise ValueError("Choose nonempty unique tasks and models.")
    if config.get("normalization", "global_train") != "global_train":
        raise ValueError("Core runner supports global_train only. Subject-test normalization needs a separate declared experiment.")
    X, groups = frame[columns].astype(float), frame.subject_id
    effective = {k: v for k, v in config.items() if k not in {"features", "schema", "output_dir"}}
    source_paths = config.get("source_paths")
    if source_paths:
        sources = [resolve_path(p) for p in source_paths]
        if any(not p.is_relative_to(PROJECT_ROOT) or p.suffix != '.py' for p in sources):
            raise ValueError("Experiment source paths must be Python files inside the project.")
        normalize = config.get('source_hash_normalization') == 'LF'
        experiment_code = fingerprint({p.relative_to(PROJECT_ROOT).as_posix():
            hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if normalize else file_hash(p)
            for p in sources})
    else:
        experiment_code = code_hash()
    identity = {"settings": effective, "data_sha256": file_hash(table_path), "feature_columns": columns,
                "code_sha256": experiment_code, "environment": environment()}
    signature = fingerprint(identity)
    out = resolve_path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "config.json"
    if manifest_path.exists() and read_json(manifest_path)["signature"] != signature:
        raise ValueError("Experiment folder has different data/settings/code/environment; use a new output_dir.")
    # Preserve the run's source identity; generic provenance may hash unrelated
    # G2 tools when a declared experiment source set is in use.
    write_json(manifest_path, {**provenance(), "signature": signature, **identity})
    write_json(out / "environment.json", environment())
    write_json(out / "run_status.json", {"status": "running", "requested_outer_subjects": sorted(selected)})
    try:
        for task in tasks:
            y, labels = task_target(frame, task)
            for name in models:
                pipeline = build_pipeline(name, config)
                for train, test in LeaveOneGroupOut().split(X, y, groups):
                    subject = str(groups.iloc[test].iloc[0])
                    if subject not in selected:
                        continue
                    folder = out / "folds" / task / name / subject
                    completion = folder / "complete.json"
                    if resume and completion.exists():
                        try:
                            done = read_json(completion)
                            if (done["signature"] == signature
                                    and {"predictions.csv", "metrics.json", "details.json"}.issubset(done["files"]) and all(
                                    (folder / file).is_file() and file_hash(folder / file) == digest
                                    for file, digest in done["files"].items())):
                                log_event(out, f"{task}/{name}/{subject}: SKIP verified")
                                continue
                        except (OSError, ValueError, KeyError):
                            pass
                    train_groups, test_groups = set(groups.iloc[train]), set(groups.iloc[test])
                    if train_groups & test_groups or len(test_groups) != 1:
                        raise ValueError("Subject leakage in outer CV.")
                    X_train, y_train = X.iloc[train], y.iloc[train]
                    started = perf_counter()
                    params, inner = {}, []
                    inner_results = None
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                        warnings.simplefilter("always")
                        estimator = clone(pipeline)
                        if config.get("nested", False) and name not in {"dummy", "lr"}:
                            splits, inner = grouped_inner_splits(X_train, y_train, groups.iloc[train],
                                                                int(config.get("inner_splits", 3)))
                            estimator = GridSearchCV(estimator, search_grid(name, config), cv=splits,
                                scoring=make_scorer(f1_score, average="macro", labels=labels, zero_division=0),
                                n_jobs=1, refit=True, error_score="raise", return_train_score=False)
                        estimator.fit(X_train, y_train)
                        predicted = estimator.predict(X.iloc[test])
                        if hasattr(estimator, "best_params_"):
                            params = estimator.best_params_
                            inner_results = pd.DataFrame(estimator.cv_results_)
                    prediction = frame.iloc[test][["subject_id", "window_id"]].copy()
                    prediction["y_true"], prediction["y_pred"] = y.iloc[test].to_numpy(), predicted
                    prediction["model"], prediction["task"] = name, task
                    prediction["sensor_set"] = config.get("sensor_set", "unspecified")
                    prediction["normalization"] = "global_train"
                    metrics = {"subject_id": subject, "task": task, "model": name, "n_test": len(test),
                        "macro_f1": float(f1_score(y.iloc[test], predicted, average="macro", labels=labels, zero_division=0)),
                        "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test], predicted)),
                        "seconds": round(perf_counter() - started, 4)}
                    details = {"outer_train_subjects": sorted(train_groups), "outer_test_subjects": [subject],
                        "inner_splits": inner, "best_params": params,
                        "warnings": sorted({str(w.message) for w in caught}),
                        "classification_report": classification_report(y.iloc[test], predicted, labels=labels,
                            output_dict=True, zero_division=0),
                        "labels": labels, "confusion_matrix": confusion_matrix(y.iloc[test], predicted, labels=labels).tolist()}
                    if config.get("record_mlp_loss", False) and name == "mlp":
                        fitted = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
                        model = fitted["model"]
                        details["mlp_optimization"] = {
                            "scope": "best_candidate_refit_on_outer_train_only",
                            "n_iter": int(model.n_iter_), "max_iter": int(model.max_iter),
                            "loss": float(model.loss_),
                            "loss_curve": [float(value) for value in model.loss_curve_],
                            "reached_max_iter": bool(model.n_iter_ >= model.max_iter),
                            "interpretation": "Training objective, not validation loss or proof of generalization. See captured convergence warnings too.",
                        }
                    write_csv(folder / "predictions.csv", prediction)
                    write_json(folder / "metrics.json", metrics)
                    write_json(folder / "details.json", details)
                    files = ["predictions.csv", "metrics.json", "details.json"]
                    if inner_results is not None:
                        write_csv(folder / "inner_cv.csv", inner_results)
                        files.append("inner_cv.csv")
                    write_json(completion, {"signature": signature, "files": {file: file_hash(folder / file) for file in files}})
                    log_event(out, f"{task}/{name}/{subject}: F1={metrics['macro_f1']:.4f}")
        return summarize(out, signature, frame, expected, tasks, models)
    except Exception as exc:
        log_event(out, f"FAILED: {exc}")
        write_json(out / "run_status.json", {"status": "failed", "error": str(exc)})
        raise


def summarize(out, signature, frame, expected, tasks, models):
    records, predictions, split_records, best_params, summaries = [], [], {}, {}, []
    complete = set(expected) == set(SUBJECTS)
    for task in tasks:
        labels = [1, 2, 3] if task == "multiclass" else [0, 1]
        for name in models:
            per_model, model_predictions, present = [], [], set()
            for subject in sorted(set(frame.subject_id)):
                folder = out / "folds" / task / name / subject
                if not (folder / "complete.json").exists():
                    continue
                done = read_json(folder / "complete.json")
                if done.get("signature") != signature or not all(
                        (folder / f).exists() and file_hash(folder / f) == digest for f, digest in done["files"].items()):
                    continue
                metric, detail = read_json(folder / "metrics.json"), read_json(folder / "details.json")
                prediction = pd.read_csv(folder / "predictions.csv", dtype={"subject_id": str, "window_id": str})
                reference = set(frame.loc[frame.subject_id == subject, "window_id"].astype(str))
                if prediction.duplicated(["subject_id", "window_id"]).any() or set(prediction.subject_id) != {subject} or set(prediction.window_id) != reference:
                    raise ValueError("Incomplete or duplicate out-of-fold predictions.")
                present.add(subject)
                records.append(metric)
                per_model.append(metric)
                predictions.append(prediction)
                model_predictions.append(prediction)
                key = f"{task}/{name}/{subject}"
                split_records[key] = {k: detail[k] for k in ("outer_train_subjects", "outer_test_subjects", "inner_splits")}
                best_params[key] = detail["best_params"]
            is_complete = present == expected and set(frame.subject_id) == expected
            complete = complete and is_complete
            if per_model:
                joined = pd.concat(model_predictions, ignore_index=True)
                score = {"task": task, "model": name, "complete_15_subjects": is_complete and len(expected) == 15,
                         "n_subjects": len(present), "missing_subjects": sorted(expected - present)}
                for metric in ("macro_f1", "balanced_accuracy"):
                    values = [r[metric] for r in per_model]
                    score[metric + "_mean"] = float(np.mean(values))
                    score[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                score["labels"] = labels
                score["confusion_matrix"] = confusion_matrix(joined.y_true, joined.y_pred, labels=labels).tolist()
                summaries.append(score)
    if not records:
        raise ValueError("No completed outer folds.")
    write_csv(out / "metrics_by_subject.csv", pd.DataFrame(records))
    write_csv(out / "predictions.csv", pd.concat(predictions, ignore_index=True))
    write_json(out / "splits.json", split_records)
    write_json(out / "best_params.json", best_params)
    result = {"status": "complete" if complete else "partial", "subjects_expected": sorted(expected),
              "standard_deviation": "sample ddof=1; null for a single subject", "results": summaries}
    write_json(out / "summary.json", result)
    write_json(out / "run_status.json", {"status": result["status"], "n_completed_folds": len(records)})
    return result
