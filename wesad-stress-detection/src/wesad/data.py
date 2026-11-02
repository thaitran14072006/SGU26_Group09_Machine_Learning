"""Đọc dữ liệu WESAD và kiểm tra cấu trúc bảng đặc trưng."""

from pathlib import Path
import pickle
import re
import csv

import numpy as np
import pandas as pd

from wesad.utils import read_json

SUBJECTS = tuple(f"S{i}" for i in (*range(2, 12), *range(13, 18)))
METADATA = {"subject_id", "window_id", "start_s", "end_s", "label_original",
            "label_binary", "quality_flag", "label_fraction", "label_purity"}
REQUIRED = ("subject_id", "window_id", "start_s", "end_s", "label_original", "quality_flag")


def subject_paths(raw_root, subjects=None):
    root = Path(raw_root)
    if subjects is None:
        subjects = sorted((p.name for p in root.glob("S*")
                           if p.is_dir() and re.fullmatch(r"S\d+", p.name)), key=lambda s: int(s[1:]))
    if not subjects or len(subjects) != len(set(subjects)) or set(subjects) - set(SUBJECTS):
        raise ValueError("Choose unique WESAD subjects S2-S11, S13-S17.")
    paths = [root / subject / f"{subject}.pkl" for subject in subjects]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_subject(path):
    # Dữ liệu đầu vào là PKL gốc của WESAD.
    with Path(path).open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    if not isinstance(data, dict) or not {"signal", "label"}.issubset(data):
        raise ValueError(f"{path}: expected signal and label keys.")
    labels = np.asarray(data["label"])
    if labels.ndim != 1 or not labels.size or not np.isfinite(labels).all():
        raise ValueError(f"{path}: labels must be a finite, nonempty 1D array.")
    if not isinstance(data["signal"], dict):
        raise ValueError(f"{path}: signal must be a mapping.")
    return data


def validate_features(frame, feature_columns=None, expected_subjects=None):
    if frame.empty or frame.columns.duplicated().any() or set(REQUIRED) - set(frame.columns):
        raise ValueError("Empty feature table, duplicate columns or missing required metadata.")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Required metadata cannot be missing.")
    if frame.duplicated(["subject_id", "window_id"]).any():
        raise ValueError("Duplicate (subject_id, window_id).")
    subjects = set(frame.subject_id.astype(str))
    if subjects - set(SUBJECTS):
        raise ValueError("Unknown WESAD subject.")
    if expected_subjects is not None and subjects != set(expected_subjects):
        raise ValueError(f"Expected {sorted(expected_subjects)}, found {sorted(subjects)}.")
    if not frame.label_original.isin([1, 2, 3]).all():
        raise ValueError("label_original must contain only 1, 2, 3.")
    starts = pd.to_numeric(frame.start_s, errors="raise").to_numpy(dtype=float)
    ends = pd.to_numeric(frame.end_s, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (starts < 0).any() or (ends <= starts).any():
        raise ValueError("Invalid window times.")
    if "label_binary" in frame and not np.array_equal(frame.label_binary, (frame.label_original == 2).astype(int)):
        raise ValueError("label_binary does not match original labels.")
    columns = list(feature_columns) if feature_columns is not None else [c for c in frame if c not in METADATA]
    if not columns or len(columns) != len(set(columns)) or set(columns) & METADATA or set(columns) - set(frame.columns):
        raise ValueError("Feature schema is empty, duplicate, missing or contains metadata.")
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Features contain Inf; use explicit missing values when justified.")
    return columns


def load_feature_table(path, schema=None, expected_subjects=None):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("Missing or duplicate CSV header columns.")
    if schema is None:
        candidate = path.parent / "feature_schema.json"
        schema = candidate if candidate.exists() else None
    columns = read_json(schema)["feature_columns"] if schema is not None else None
    frame = pd.read_csv(path, dtype={"subject_id": str, "window_id": str})
    columns = validate_features(frame, columns, expected_subjects)
    return frame.reset_index(drop=True), columns
