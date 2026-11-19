"""Measure fixed cumulative RF configurations on train and held-out subjects."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from wesad.data import load_feature_table
from wesad.evaluation import task_target
from wesad.models import build_pipeline
from wesad.utils import (
    environment, file_hash, fingerprint, read_config, read_json,
    resolve_path, write_csv, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick-subject")
    args = parser.parse_args()
    config = read_config(args.config)
    if len({item["name"] for item in config["stages"]}) != len(config["stages"]):
        raise ValueError("Stage names must be unique")
    table, schema = resolve_path(config["features"]), resolve_path(config["schema"])
    if file_hash(table) != config["feature_sha256"]:
        raise ValueError("Feature checksum differs")
    frame, columns = load_feature_table(table, schema, config["expected_subjects"])
    subjects = config["expected_subjects"]
    if args.quick_subject and args.quick_subject not in subjects:
        raise ValueError("Unknown test subject")
    selected = [args.quick_subject] if args.quick_subject else subjects
    output = resolve_path(config["output_dir"])
    if args.quick_subject:
        output = output.with_name(output.name + "_quick_" + args.quick_subject)
    settings = {
        "config": config, "quick_subject": args.quick_subject,
        "environment": environment(), "schema_sha256": file_hash(schema),
        "source_sha256": {name: file_hash(ROOT / name) for name in (
            "scripts/run_rf_train_test.py", "src/wesad/models.py",
            "src/wesad/data.py", "src/wesad/evaluation.py", "src/wesad/utils.py",
        )},
    }
    signature = fingerprint(settings)
    manifest_file = output / "run_info.json"
    if manifest_file.exists():
        if read_json(manifest_file)["signature"] != signature:
            raise ValueError("Run settings changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output has files without a run manifest")
        write_json(manifest_file, {
            **settings, "signature": signature,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "quick" if args.quick_subject else "main",
        })
    X = frame[columns].astype(float)
    records = []
    expected_keys = []
    for task in config["tasks"]:
        y, labels = task_target(frame, task)
        for subject in selected:
            train = frame.subject_id.ne(subject)
            test = ~train
            if frame.loc[train, "subject_id"].nunique() != 14:
                raise ValueError("Expected 14 training subjects")
            for stage in config["stages"]:
                key = f"{task}_{subject}_{stage['name']}"
                expected_keys.append(key)
                checkpoint = output / "folds" / (key + ".json")
                if checkpoint.exists():
                    record = read_json(checkpoint)
                    if record["signature"] != signature or record["key"] != key:
                        raise ValueError("Checkpoint settings differ")
                else:
                    estimator = build_pipeline("rf", {
                        "seed": config["seed"], "balance": False,
                        "model_params": {"rf": stage["params"]},
                    })
                    start = perf_counter()
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                        warnings.simplefilter("always")
                        estimator.fit(X.loc[train], y.loc[train])
                        scores = {}
                        for split, mask in (("train", train), ("test", test)):
                            prediction = estimator.predict(X.loc[mask])
                            scores[split + "_macro_f1"] = float(f1_score(
                                y.loc[mask], prediction, labels=labels,
                                average="macro", zero_division=0,
                            ))
                            scores[split + "_balanced_accuracy"] = float(
                                balanced_accuracy_score(y.loc[mask], prediction)
                            )
                    record = {
                        "key": key, "signature": signature, "task": task,
                        "subject_id": subject, "stage": stage["name"],
                        "params": estimator["model"].get_params(),
                        "n_train": int(train.sum()), "n_test": int(test.sum()),
                        "train_subjects": sorted(frame.loc[train, "subject_id"].unique()),
                        "seconds": perf_counter() - start,
                        "warnings": [str(item.message) for item in caught], **scores,
                    }
                    write_json(checkpoint, record)
                records.append(record)
                print(key, f"train={record['train_macro_f1']:.4f}",
                      f"test={record['test_macro_f1']:.4f}", flush=True)
    results = pd.DataFrame(records).drop(columns=["signature", "params", "train_subjects", "warnings"])
    if results.key.duplicated().any() or set(results.key) != set(expected_keys):
        raise ValueError("Fold inventory differs")
    write_csv(output / "metrics.csv", results)
    write_json(output / "verification.json", {
        "status": "passed", "signature": signature,
        "scope": "quick" if args.quick_subject else "main",
        "n_folds": len(results), "n_test_subjects": len(selected),
    })
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
