"""Audit G2 preprocessing sequentially; output diagnostics, not model features."""

import argparse
import gc
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from wesad.data import load_subject, subject_paths
from wesad.preprocessing_g2 import acc_emg_block_statistics, iter_preprocessed_windows
from wesad.utils import (environment, file_hash, fingerprint, provenance, read_config,
                         resolve_path, write_csv, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocessing_g2_basic.yaml")
    parser.add_argument("--local")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = read_config(args.config, args.local)
    out = resolve_path(args.output_dir or config["output_dir"])
    paths = subject_paths(resolve_path(config["data_root"]), args.subjects or config["subjects"])
    # Keep evidence from each run; resume/merging belongs to the shared CLI owner.
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new empty output directory; existing evidence is preserved.")
    processing = {k: v for k, v in config.items() if k not in {"data_root", "subjects", "output_dir"}}
    snapshot = {"processing": processing, **provenance(), "environment": environment(),
                "subjects": [p.stem for p in paths], "scope": "G2_preprocessing_diagnostic_only"}
    snapshot["signature"] = fingerprint(snapshot)
    write_json(out / "config.json", snapshot)
    status = {"status": "running", "subjects": snapshot["subjects"], "completed_subjects": []}
    write_json(out / "run_status.json", status)
    summaries = []
    try:
        for path in paths:
            started = perf_counter()
            data, qa, rows = load_subject(path), {}, []
            try:
                for window in iter_preprocessed_windows(data, config, qa):
                    row = {"subject_id": path.stem, "window_id": f"{path.stem}_{window.start_index}",
                           "start_s": window.start_s, "end_s": window.end_s,
                           "label_original": window.label, "label_binary": int(window.label == 2),
                           "label_fraction": window.label_fraction,
                           "quality_flag": "finite_filtered_interior_not_physiological_QA"}
                    row.update(acc_emg_block_statistics(window))
                    rows.append(row)
            finally:
                del data
                gc.collect()
            if not rows:
                raise ValueError(f"{path.stem}: no windows survived; QA={qa}")
            frame = pd.DataFrame(rows)
            target = out / f"{path.stem}_acc_emg_diagnostic.csv"
            write_csv(target, frame)
            report = {"subject_id": path.stem, **qa,
                      "label_counts": {str(k): int(v) for k, v in frame.label_original.value_counts().sort_index().items()},
                      "source_sha256": file_hash(path), "diagnostic_csv_sha256": file_hash(target),
                      "signature": snapshot["signature"], "seconds": round(perf_counter() - started, 3),
                      "n_block_diagnostic_columns": len(frame.columns) - 8,
                      "units": "recorded_PKL_scale; IAV=recorded_scale*seconds; no_conversion"}
            write_json(out / f"{path.stem}.json", report)
            summaries.append({k: v for k, v in report.items() if not isinstance(v, dict)})
            write_csv(out / "quality_report.csv", pd.DataFrame(summaries))
            status["completed_subjects"].append(path.stem)
            write_json(out / "run_status.json", status)
            print(f"{path.stem}: {qa['n_kept_windows']} kept; {report['seconds']} seconds", flush=True)
        status["status"] = "complete"
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(out / "run_status.json", status)


if __name__ == "__main__":
    main()
