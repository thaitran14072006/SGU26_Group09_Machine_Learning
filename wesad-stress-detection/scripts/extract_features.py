"""Doc tung PKL WESAD, chia cua so, luu dac trung thong ke tho.

Chay tu thu muc goc: python scripts/extract_features.py
Khong scaler, khong chia train/test va chua loc nhieu sinh ly.
"""

import argparse
import gc
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wesad.preprocessing import iter_windows


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_statistics(signals):
    """Bay dac trung moi kenh; khong goi day la HRV."""
    result = {}
    for sensor, values in signals.items():
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        prefix = sensor.lower()
        result.update({
            f"{prefix}_mean": float(x.mean()),
            f"{prefix}_std": float(x.std(ddof=0)),
            f"{prefix}_min": float(x.min()),
            f"{prefix}_max": float(x.max()),
            f"{prefix}_rms": float(np.sqrt(np.mean(x * x))),
            f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
            f"{prefix}_range": float(np.ptp(x)),
        })
    return result


def process_subject(path, sensors, config):
    # Chi doc file pickle tu nguon tin cay (WESAD goc).
    with path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")
    try:
        labels = np.asarray(data["label"])
        chest = data["signal"]["chest"]
        signals = {}
        for sensor in sensors:
            values = np.asarray(chest[sensor])
            if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
                raise ValueError(f"{sensor}: phien ban nay chi nhan kenh don.")
            signals[sensor] = values.reshape(-1)
        rows = []
        for window in iter_windows(
            signals, labels,
            sampling_rate_hz=config["sampling_rate_hz"],
            window_seconds=config["window_seconds"],
            step_seconds=config["step_seconds"],
            allowed_labels=config["labels"],
        ):
            row = {
                "subject_id": path.stem,
                "window_id": f"{path.stem}_{window.start_index}",
                "start_s": window.start_s,
                "end_s": window.end_s,
                "label_original": window.label,
                "quality_flag": "finite_only_not_physiological_QA",
            }
            row.update(extract_statistics(window.signals))
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        del data
        # Cac view tin hieu duoc giai phong khi ham ket thuc.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/features_chest.yaml")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["device"] != "chest" or config["sampling_rate_hz"] != 700:
        raise ValueError("Phien ban nay chi ho tro chest va labels WESAD o 700 Hz.")
    if config["label_rule"] != "homogeneous":
        raise ValueError("Phien ban nay chi ho tro nhan dong nhat.")
    sensors = config["sensors"]
    if not sensors or not set(sensors).issubset({"ECG", "EDA", "Resp", "Temp", "EMG"}):
        raise ValueError("Chon kenh don chest: ECG, EDA, Resp, Temp hoac EMG.")
    raw_dir = resolve_path(config["data_root"])
    output_dir = resolve_path(config["output_dir"])
    subjects = config.get("subjects")
    if subjects is not None:
        if not subjects or any(not re.fullmatch(r"S\d+", str(s)) for s in subjects):
            raise ValueError("subjects phai la danh sach nhu [S2, S3] hoac null.")
        paths = [raw_dir / s / f"{s}.pkl" for s in subjects]
    else:
        paths = sorted(
            (p for p in raw_dir.glob("S*/S*.pkl")
             if re.fullmatch(r"S\d+", p.stem) and p.parent.name == p.stem),
            key=lambda p: int(p.stem[1:]),
        )
    if not paths:
        raise FileNotFoundError(f"Khong tim thay S*/S*.pkl trong {raw_dir}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Giu cau hinh de tranh ghep nham dac trung giua cac lan chay.
    effective = dict(config)
    effective["data_root"] = str(raw_dir.resolve())
    effective["output_dir"] = str(output_dir.resolve())
    effective["subjects"] = [p.stem for p in paths]
    script_hash = hashlib.sha256(Path(__file__).read_bytes())
    script_hash.update((PROJECT_ROOT / "src/wesad/preprocessing.py").read_bytes())
    effective["code_sha256"] = script_hash.hexdigest()
    snapshot = output_dir / "config.yaml"
    if snapshot.exists() and yaml.safe_load(snapshot.read_text(encoding="utf-8")) != effective:
        raise ValueError("Thu muc ket qua da co cau hinh/code khac. Chon output_dir moi.")
    snapshot.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    marker = output_dir / "run_status.json"
    marker.write_text(json.dumps({"status": "running"}), encoding="utf-8")

    summaries = []
    feature_paths = []
    for path in paths:
        start = perf_counter()
        print(f"Dang xu ly {path.stem}...", flush=True)
        frame = process_subject(path, sensors, config)
        gc.collect()
        if frame.empty:
            raise ValueError(f"{path.stem}: khong co cua so hop le. Kiem tra du lieu/cau hinh.")
        target = output_dir / f"{path.stem}.csv"
        temp = target.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(target)
        feature_paths.append(target)
        summaries.append({
            "subject_id": path.stem,
            "n_windows": len(frame),
            "n_baseline": int((frame.label_original == 1).sum()),
            "n_stress": int((frame.label_original == 2).sum()),
            "n_amusement": int((frame.label_original == 3).sum()),
            "seconds": round(perf_counter() - start, 2),
        })
        print(f"  Da luu {len(frame)} cua so -> {target.name}", flush=True)
        del frame
    # Chi ghep bang DAC TRUNG cua nhung nguoi trong lan chay nay.
    merged = pd.concat([pd.read_csv(p) for p in feature_paths], ignore_index=True)
    temp = output_dir / "all_subjects.csv.tmp"
    merged.to_csv(temp, index=False)
    temp.replace(output_dir / "all_subjects.csv")
    pd.DataFrame(summaries).to_csv(output_dir / "quality_report.csv", index=False)
    marker.write_text(json.dumps({"status": "complete", "n_windows": len(merged)}), encoding="utf-8")
    print(f"Hoan tat: {len(paths)} nguoi, {len(merged)} cua so.")


if __name__ == "__main__":
    main()
