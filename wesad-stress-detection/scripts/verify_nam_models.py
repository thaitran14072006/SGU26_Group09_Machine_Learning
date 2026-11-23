"""Verify Nam's model handoff independently of exporter code (requires local G2 CSV)."""
import hashlib
import json
import os
import sys
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/models/group_submission"
DATA = ROOT / "data/features/g2_physiological68_group15_20261107/all_subjects.csv"
EXPECTED_SUBJECTS = [f"S{i}" for i in (*range(2,12), *range(13,18))]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing local data: {DATA}")
    df = pd.read_csv(DATA)
    if len(df) != 6189 or set(df.subject_id.astype(str)) != set(EXPECTED_SUBJECTS):
        raise ValueError("Incorrect G2 data")
    summary_path = OUT / "summary/nam_model_summary.csv"
    summary = pd.read_csv(summary_path)
    if len(summary) != 4 or set(zip(summary.model,summary.task)) != {(a,t) for a in ('knn','svm') for t in ('binary','multiclass')}:
        raise ValueError("Summary missing required four rows")
    for algorithm in ("knn", "svm"):
        folder = OUT/algorithm
        schema = json.loads((folder/"feature_schema.json").read_text("utf-8"))
        columns = schema["feature_columns"]
        if len(columns)!=68 or len(set(columns))!=68 or any(c not in df for c in columns):
            raise ValueError(f"Bad schema: {algorithm}")
        X = df.loc[:19, columns].astype(float).replace([np.inf,-np.inf],np.nan)
        for task, expected in (("binary",{0,1}),("multiclass",{1,2,3})):
            path = folder/f"{task}.joblib"
            metadata = json.loads((folder/f"metadata_{task}.json").read_text("utf-8"))
            if sha256_file(path) != metadata["joblib_sha256"]:
                raise ValueError(f"Modified model artifact: {path}")
            if metadata["data_sha256"] != sha256_file(DATA):
                raise ValueError(f"Data hash mismatch: {path}")
            if metadata["labels"] != sorted(expected) or metadata["n_windows_fit"] != 6189:
                raise ValueError(f"Metadata mismatch: {path}")
            model = joblib.load(path)  # Only load known/trusted bundles.
            if list(model.named_steps) != ["imputer","scaler","selection","model"]:
                raise ValueError(f"Wrong pipeline steps: {path}")
            if model.named_steps["selection"].get_support(indices=True).size != 30:
                raise ValueError(f"Wrong feature selection: {path}")
            if set(model.classes_.tolist()) != expected:
                raise ValueError(f"Wrong labels: {path}")
            p = model.predict(X)
            if len(p) != 20 or not set(p).issubset(expected):
                raise ValueError(f"Bad prediction: {path}")
            print(f"PASS: {algorithm}/{task}; first predictions={p[:5].tolist()}")
    print("ALL FOUR MODELS VERIFIED. This is a reload/smoke test, not an unbiased evaluation.")

if __name__=="__main__":
    main()
