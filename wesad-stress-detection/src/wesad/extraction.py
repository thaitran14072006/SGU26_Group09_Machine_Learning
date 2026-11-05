"""Trích xuất tuần tự, lưu bảng đặc trưng và trạng thái từng subject.

subject_processor nhận đường dẫn PKL, cấu hình và trả về DataFrame.
"""

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from wesad.data import load_subject, subject_paths, validate_features
from wesad.preprocessing import iter_windows
from wesad.utils import (code_hash, environment, file_hash, fingerprint, import_callable,
                         log_event, provenance, read_json, resolve_path, write_csv, write_json)


def extract_statistics(signals):
    result = {}
    for sensor, values in signals.items():
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        for name, value in {"mean": x.mean(), "std": x.std(ddof=0), "min": x.min(),
                            "max": x.max(), "rms": np.sqrt(np.mean(x * x)),
                            "iqr": np.percentile(x, 75) - np.percentile(x, 25),
                            "range": np.ptp(x)}.items():
            result[f"{sensor.lower()}_{name}"] = float(value)
    return result


def process_subject(path, config):
    if config.get("device") != "chest" or config.get("sampling_rate_hz") != 700:
        raise ValueError("Raw statistics support chest/700 Hz only; use subject_processor for other devices.")
    sensors = config.get("sensors", [])
    if not sensors or len(set(sensors)) != len(sensors) or set(sensors) - {"ECG", "EDA", "Resp", "Temp", "EMG"}:
        raise ValueError("Choose single-channel chest sensors; ACC/wrist require the team's processor.")
    data = load_subject(path)
    try:
        chest = data["signal"]["chest"]
        signals = {}
        for sensor in sensors:
            values = np.asarray(chest[sensor])
            if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
                raise ValueError(f"{sensor}: expected a single channel.")
            signals[sensor] = values.reshape(-1)
        rows = []
        for window in iter_windows(signals, data["label"],
                                   sampling_rate_hz=config["sampling_rate_hz"],
                                   window_seconds=config["window_seconds"],
                                   step_seconds=config["step_seconds"],
                                   allowed_labels=config["labels"],
                                   label_rule=config["label_rule"],
                                   min_label_fraction=config.get("min_label_fraction", 0.9)):
            row = {"subject_id": Path(path).stem,
                   "window_id": f"{Path(path).stem}_{window.start_index}",
                   "start_s": window.start_s, "end_s": window.end_s,
                   "label_original": window.label, "label_binary": int(window.label == 2),
                   "label_fraction": window.label_fraction,
                   "quality_flag": "finite_only_not_physiological_QA"}
            row.update(extract_statistics(window.signals))
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        del data


def run_extraction(config, subjects=None, resume=False):
    processing = {k: v for k, v in config.items() if k not in {"data_root", "output_dir", "subjects"}}
    if config.get("label_rule") not in {"homogeneous", "majority"}:
        raise ValueError("label_rule must be homogeneous or majority.")
    paths = subject_paths(resolve_path(config["data_root"]), subjects if subjects is not None else config.get("subjects"))
    out = resolve_path(config["output_dir"])
    signature = fingerprint({"processing": processing, "code": code_hash(), "environment": environment()})
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "config.json"
    if snapshot.exists() and read_json(snapshot)["signature"] != signature:
        raise ValueError("Output contains different processing/code/environment. Choose a new output_dir.")
    processor = import_callable(config["subject_processor"]) if config.get("subject_processor") else process_subject
    write_json(snapshot, {"signature": signature, "processing": processing, **provenance(), "environment": environment()})
    write_json(out / "run_status.json", {"status": "running", "subjects": [p.stem for p in paths]})
    summaries, frames, schema_columns = [], [], None
    try:
        for path in paths:
            started = perf_counter()
            target, manifest_path = out / f"{path.stem}.csv", out / f"{path.stem}.json"
            source_hash = file_hash(path)
            manifest, frame, skipped = None, None, False
            if resume and target.exists() and manifest_path.exists():
                try:
                    manifest = read_json(manifest_path)
                    if (manifest.get("status") == "complete" and manifest.get("signature") == signature and manifest.get("source_sha256") == source_hash
                            and manifest.get("csv_sha256") == file_hash(target)):
                        frame = pd.read_csv(target, dtype={"subject_id": str, "window_id": str})
                        validate_features(frame, manifest["feature_columns"], [path.stem])
                        if len(frame) != manifest["n_windows"]:
                            raise ValueError("Incomplete cached rows.")
                        skipped = True
                except (ValueError, KeyError, OSError):
                    frame = None
            if not skipped:
                frame = processor(path, config)
                columns = validate_features(frame, expected_subjects=[path.stem])
                if not np.allclose(frame.end_s - frame.start_s, config["window_seconds"]):
                    raise ValueError("Processor returned incorrect window duration.")
                write_csv(target, frame)
                manifest = {"status": "complete", "signature": signature, "source_sha256": source_hash,
                            "csv_sha256": file_hash(target), "feature_columns": columns,
                            "n_windows": len(frame), "seconds": round(perf_counter() - started, 3)}
                if "extraction_qa" in frame.attrs:
                    manifest["extraction_qa"] = frame.attrs["extraction_qa"]
                write_json(manifest_path, manifest)
            columns = validate_features(frame)
            if schema_columns is not None and columns != schema_columns:
                raise ValueError("Different feature order across subjects.")
            schema_columns = columns
            summary = {"subject_id": path.stem, "n_windows": len(frame), "resumed": skipped,
                       "seconds": manifest["seconds"]}
            summary.update(manifest.get("extraction_qa", {}))
            summary.update({f"n_label_{label}": int((frame.label_original == label).sum()) for label in (1, 2, 3)})
            summaries.append(summary)
            frames.append(frame)  # Chỉ giữ bảng đặc trưng để ghép cuối lượt.
            log_event(out, f"{path.stem}: {'SKIP verified' if skipped else 'EXTRACTED'} {len(frame)} windows")
            gc.collect()
        merged = pd.concat(frames, ignore_index=True)
        validate_features(merged, schema_columns, [p.stem for p in paths])
        write_csv(out / "all_subjects.csv", merged)
        write_csv(out / "quality_report.csv", pd.DataFrame(summaries))
        write_json(out / "feature_schema.json", {"feature_columns": schema_columns,
                   "feature_version": config["feature_version"], "processing_signature": signature,
                   "sensor_groups": {
                       "chest": [c for c in schema_columns if c.startswith("chest_")],
                       "wrist": [c for c in schema_columns if c.startswith("wrist_")],
                       "combined": schema_columns},
                   "description": "Raw statistics unless a team subject_processor is configured."})
        write_json(out / "checksums.json", {p.name: file_hash(p) for p in
                   [out / "all_subjects.csv", out / "feature_schema.json", *[out / f"{s.stem}.csv" for s in paths]]})
        write_json(out / "run_status.json", {"status": "complete", "subjects": [p.stem for p in paths],
                   "n_windows": len(merged), "scope": "selected_subjects_only"})
        return merged
    except Exception as exc:
        log_event(out, f"FAILED: {exc}")
        write_json(out / "run_status.json", {"status": "failed", "error": str(exc),
                   "completed_subjects": [s["subject_id"] for s in summaries]})
        raise

