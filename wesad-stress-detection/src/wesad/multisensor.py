"""Common windows at native WESAD rates; raw statistics, not physiological features.

No interpolation, unit conversion, filtering or imputation is performed here.
ACC retains its recorded scale and gravity; x/y/z and magnitude stay separate.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from wesad.data import load_subject
from wesad.extraction import extract_statistics
from wesad.preprocessing import iter_windows

# (native Hz, recorded axes); label timestamps use the synchronized 700 Hz grid.
SENSOR_SPECS = {
    "chest": {"ECG": (700, 1), "EDA": (700, 1), "EMG": (700, 1),
              "Resp": (700, 1), "Temp": (700, 1), "ACC": (700, 3)},
    "wrist": {"BVP": (64, 1), "EDA": (4, 1), "TEMP": (4, 1), "ACC": (32, 3)},
}
STATS = ("mean", "std", "min", "max", "rms", "iqr", "range")
METADATA = ["subject_id", "window_id", "start_s", "end_s", "label_original",
            "label_binary", "label_fraction", "quality_flag"]


def configured_sensors(config):
    sensors = config.get("sensors")
    if config.get("sampling_rate_hz") != 700:
        raise ValueError("Multisensor label grid must be 700 Hz.")
    if not isinstance(sensors, dict) or not sensors or set(sensors) - set(SENSOR_SPECS):
        raise ValueError("sensors must map chest/wrist to nonempty sensor lists.")
    selected = []
    for device, specs in SENSOR_SPECS.items():
        if device not in sensors:
            continue
        names = sensors[device]
        if (not isinstance(names, list) or not names or any(not isinstance(s, str) for s in names)
                or len(names) != len(set(names)) or set(names) - set(specs)):
            raise ValueError(f"Invalid {device} sensors; choose {list(specs)} (case-sensitive).")
        # Canonical feature order independent of mapping/list input order.
        selected.extend((device, sensor, *specs[sensor]) for sensor in specs if sensor in names)
    return selected


def channel_names(device, sensor, axes):
    prefix = f"{device}_{sensor.lower()}"
    return [prefix] if axes == 1 else [f"{prefix}_{axis}" for axis in ("x", "y", "z", "magnitude")]


def process_subject(path, config):
    """Read one PKL; emit one feature row per common, label-valid window.

    All selected sensors must cover the same half-open interval [start_s, end_s).
    A short/nonfinite channel excludes that window for every sensor group.
    Window IDs always refer to the original 700 Hz label index.
    """
    specs = configured_sensors(config)
    data = load_subject(path)
    try:
        labels = np.asarray(data["label"])
        signals = {}
        columns = []
        for device, sensor, rate, axes in specs:
            try:
                values = np.asarray(data["signal"][device][sensor])
            except KeyError as exc:
                raise ValueError(f"{Path(path).stem}: missing required {device}/{sensor}.") from exc
            if values.ndim == 1 and axes == 1:
                values = values.reshape(-1, 1)
            if (values.ndim != 2 or values.shape[1] != axes or not len(values)
                    or not np.issubdtype(values.dtype, np.number)):
                raise ValueError(f"{device}/{sensor}: expected nonempty numeric (n, {axes}) signal.")
            signals[(device, sensor)] = values
            columns.extend(f"{name}_{stat}" for name in channel_names(device, sensor, axes) for stat in STATS)

        # iter_windows validates parameters and derives labels on the ORIGINAL grid.
        # The label-grid signal is a view, not an upsampled sensor array.
        windows = iter_windows({"label_grid": labels}, labels,
            sampling_rate_hz=700, window_seconds=config["window_seconds"],
            step_seconds=config["step_seconds"], allowed_labels=config["labels"],
            label_rule=config["label_rule"], min_label_fraction=config.get("min_label_fraction", .90))
        qa = {"n_candidate_windows": 0, "n_label_valid_windows": 0,
              "n_rejected_label": 0, "n_rejected_incomplete_signal": 0,
              "n_rejected_nonfinite_signal": 0, "n_kept_windows": 0}
        rows = []
        for window in windows:
            qa["n_label_valid_windows"] += 1
            chunks, reason = {}, None
            for device, sensor, rate, axes in specs:
                # Sample k belongs to [a,b) iff a <= k/rate < b.
                # Ceil integer arithmetic handles fractional boundaries exactly.
                first = (window.start_index * rate + 699) // 700
                last = (window.end_index * rate + 699) // 700
                values = signals[(device, sensor)]
                if last > len(values) or last <= first:
                    reason = "n_rejected_incomplete_signal"
                    break
                chunk = values[first:last]
                if not np.isfinite(chunk).all():
                    reason = "n_rejected_nonfinite_signal"
                    break
                names = channel_names(device, sensor, axes)
                if axes == 1:
                    chunks[names[0]] = chunk[:, 0]
                else:
                    for i, name in enumerate(names[:3]):
                        chunks[name] = chunk[:, i]
                    # Convert before squaring to avoid integer overflow.
                    chunks[names[3]] = np.linalg.norm(chunk.astype(np.float64), axis=1)
            if reason is not None:
                qa[reason] += 1
                continue
            row = {"subject_id": Path(path).stem,
                   "window_id": f"{Path(path).stem}_{window.start_index}",
                   "start_s": window.start_s, "end_s": window.end_s,
                   "label_original": window.label, "label_binary": int(window.label == 2),
                   "label_fraction": window.label_fraction,
                   "quality_flag": "finite_only_not_physiological_QA"}
            row.update(extract_statistics(chunks))
            rows.append(row)

        size = int(round(700 * config["window_seconds"]))
        step = int(round(700 * config["step_seconds"]))
        qa["n_candidate_windows"] = max(0, (len(labels) - size) // step + 1)
        qa["n_rejected_label"] = qa["n_candidate_windows"] - qa["n_label_valid_windows"]
        qa["n_kept_windows"] = len(rows)
        if not rows:
            raise ValueError(f"{Path(path).stem}: no common windows survived. QA: {qa}")
        frame = pd.DataFrame(rows, columns=METADATA + columns)
        frame.attrs["extraction_qa"] = qa
        return frame
    finally:
        del data
