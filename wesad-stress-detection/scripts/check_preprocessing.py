"""Kiểm tra tiền xử lý từng PKL; không tạo feature sinh lý hoặc điểm mô hình."""

import argparse
import gc
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wesad.data import load_subject, subject_paths
from wesad.preprocessing import prepare_signals
from wesad.utils import (environment, file_hash, provenance, read_config,
                         resolve_path, write_csv, write_json)
from wesad.windowing import WindowAudit, iter_multirate_windows


def main():
    # ASCII help also works in Windows terminals using legacy cp1252 output.
    parser = argparse.ArgumentParser(description="Check filtering and native-rate windows, one WESAD PKL at a time.")
    parser.add_argument("--config", default="configs/preprocessing_nhu_draft.yaml")
    parser.add_argument("--local")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--output", required=True, help="New diagnostic output directory")
    args = parser.parse_args()
    config = read_config(args.config, args.local)
    paths = subject_paths(resolve_path(config["data_root"]), args.subjects or config.get("subjects"))
    out = resolve_path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory; diagnostic runs are not overwritten.")
    out.mkdir(parents=True, exist_ok=True)
    # Do not publish a private raw path in the diagnostic config snapshot.
    write_json(out / "config.json", {"processing": {k: v for k, v in config.items() if k != "data_root"},
               "environment": environment(), **provenance()})
    write_json(out / "run_status.json", {"status": "running", "scope": "preprocessing_diagnostics"})
    reports = []
    try:
        for path in paths:
            started = perf_counter()
            data = load_subject(path)
            signals = prepare_signals(data, config)
            audit = WindowAudit()
            # Only lightweight metadata are retained, never all signal windows.
            rows = [w.metadata(path.stem) for w in iter_multirate_windows(
                signals, data["label"], audit=audit, **config["window"])]
            label_fs = config["window"]["label_sampling_rate_hz"]
            channels = {}
            for name, signal in signals.items():
                raw_spec = config["channels"][name]
                raw = np.asarray(data["signal"][raw_spec["device"]][raw_spec["sensor"]])
                values = np.asarray(signal.values)
                channels[name] = {"samples": len(values), "sampling_rate_hz": signal.sampling_rate_hz,
                    "duration_s": len(values) / signal.sampling_rate_hz,
                    "duration_difference_from_labels_s": len(values) / signal.sampling_rate_hz - len(data["label"]) / label_fs,
                    "raw_nonfinite_samples": int((~np.isfinite(raw)).reshape(len(raw), -1).any(axis=1).sum()),
                    "processed_nonfinite_samples": int((~np.isfinite(values)).reshape(len(values), -1).any(axis=1).sum()),
                    "raw_flat": bool(np.isfinite(raw).all() and np.all(np.ptp(raw, axis=0) == 0)),
                    "units": "raw_unverified"}
            columns = ["subject_id", "window_id", "start_s", "end_s", "label_original",
                       "label_binary", "label_fraction", "quality_flag"]
            write_csv(out / f"{path.stem}_windows.csv", pd.DataFrame(rows, columns=columns))
            counts = {k: v for k, v in vars(audit).items() if k != "reasons"}
            counts["reasons"] = dict(audit.reasons)
            reports.append({"subject_id": path.stem, "source_sha256": file_hash(path),
                "audit": counts, "channels": channels,
                "n_label": {str(label): sum(row["label_original"] == label for row in rows) for label in (1, 2, 3)},
                "seconds": round(perf_counter() - started, 3)})
            write_json(out / f"{path.stem}_quality.json", reports[-1])
            print(f"{path.stem}: {audit.accepted}/{audit.candidates} windows; diagnostics only")
            del data, signals, rows, raw, values, signal
            gc.collect()
        write_json(out / "run_status.json", {"status": "complete", "scope": "preprocessing_diagnostics",
            "subjects": [p.stem for p in paths], "physiological_qa_validated": False,
            "ready_for_training": False})
    except Exception as exc:
        write_json(out / "run_status.json", {"status": "failed", "scope": "preprocessing_diagnostics",
                   "completed_subjects": [r["subject_id"] for r in reports], "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
