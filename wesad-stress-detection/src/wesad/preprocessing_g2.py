"""G2 basic preprocessing handoff, separate from the agreed raw112 extractor.

Native-rate, offline filters follow preprocessing_g1_design.yaml. No imputation,
unit conversion or physiological acceptance is inferred from finite values.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from wesad.extraction import extract_statistics
from wesad.multisensor import SENSOR_SPECS, channel_names
from wesad.preprocessing import Window, iter_windows


def filter_finite_segments(values, rate, specification, order=4):
    """Filter independent finite runs; leave gaps, short runs and edges as NaN."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Filtering requires one channel at a time.")
    edge_seconds = specification["edge_seconds"]
    if not np.isfinite(edge_seconds) or edge_seconds < 0:
        raise ValueError("edge_seconds must be finite and nonnegative.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("Filter order must be a positive integer.")
    sos = butter(order, specification["cutoff_hz"], btype=specification["kind"],
                 fs=rate, output="sos")
    # Explicit SciPy SOS default padding formula, also used for short-run checks.
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    edge = int(np.ceil(edge_seconds * rate))
    result = np.full(x.shape, np.nan)
    valid = np.isfinite(x)
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if last - first <= max(padlen, 2 * edge):
            continue
        filtered = sosfiltfilt(sos, x[first:last], padtype="odd", padlen=padlen)
        stop = last - edge
        result[first + edge:stop] = filtered[edge:len(filtered) - edge if edge else None]
    return result


def prepare_signals(data, config):
    """Return flat channels and native rates for Mai's future extractor integration."""
    channels, rates, raw = {}, {}, {}
    for device, specs in SENSOR_SPECS.items():
        for sensor, (rate, axes) in specs.items():
            values = np.asarray(data["signal"][device][sensor])
            if values.ndim == 1 and axes == 1:
                values = values[:, None]
            if (values.ndim != 2 or values.shape[1] != axes or not len(values)
                    or not np.issubdtype(values.dtype, np.number)):
                raise ValueError(f"Invalid shape/type for {device}/{sensor}.")
            names = channel_names(device, sensor, axes)
            for axis, name in enumerate(names[:axes]):
                x = values[:, axis]
                raw[name] = x
                rates[name] = rate
                spec = config["filters"].get(name)
                channels[name] = (filter_finite_segments(x, rate, spec, config["filter_order"])
                                  if spec else x)
            if axes == 3:
                name = names[3]
                raw[name] = np.linalg.norm(values.astype(np.float64), axis=1)
                channels[name], rates[name] = raw[name], rate
    return channels, rates, raw


def iter_preprocessed_windows(data, config, qa):
    """Yield common 60/5 windows on the original clock, recording first rejection.

    All selected channels must be usable. No bridging gaps or upsampling to 700 Hz.
    A rejection is counted once in channel order, not once per affected channel.
    """
    if (config["window_seconds"] != 60 or config["step_seconds"] != 5
            or config["labels"] != [1, 2, 3] or config["min_label_fraction"] != .9):
        raise ValueError("G2 basic handoff requires 60/5, labels 1/2/3 and majority 90%.")
    channels, rates, raw = prepare_signals(data, config)
    labels = np.asarray(data["label"])
    qa.update(n_candidate_windows=max(0, (len(labels) - 42000) // 3500 + 1),
              n_label_valid_windows=0, n_rejected_label=0, n_rejected_incomplete_signal=0,
              n_rejected_nonfinite_signal=0, n_rejected_filter_edge_or_short_segment=0,
              n_kept_windows=0)
    for window in iter_windows({"label_grid": labels}, labels):
        qa["n_label_valid_windows"] += 1
        chunks, reason = {}, None
        for name, values in channels.items():
            rate = rates[name]
            first = (window.start_index * rate + 699) // 700
            last = (window.end_index * rate + 699) // 700
            if last > len(values):
                reason = "n_rejected_incomplete_signal"
                break
            if not np.isfinite(raw[name][first:last]).all():
                reason = "n_rejected_nonfinite_signal"
                break
            chunk = values[first:last]
            if not np.isfinite(chunk).all():
                reason = "n_rejected_filter_edge_or_short_segment"
                break
            chunks[name] = chunk
        if reason:
            qa[reason] += 1
            continue
        qa["n_kept_windows"] += 1
        yield Window(window.start_index, window.end_index, window.start_s, window.end_s,
                     window.label, chunks, window.label_fraction)
    qa["n_rejected_label"] = qa["n_candidate_windows"] - qa["n_label_valid_windows"]


def acc_emg_block_statistics(window):
    """Mean of twelve 5-second block statistics, in recorded PKL units.

    Diagnostic handoff only: these columns do not replace raw112 or Mai's schema.
    IAV uses sum(abs(x))/fs, a rectangle integral in recorded units * seconds.
    """
    result = {}
    for name, values in window.signals.items():
        if "_acc_" not in name and name != "chest_emg":
            continue
        rate = 32 if name.startswith("wrist_") else 700
        if len(values) != 60 * rate or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected a finite complete 60-second channel.")
        blocks = np.asarray(values, dtype=np.float64).reshape(12, 5 * rate)
        per_block = [extract_statistics({name: block}) for block in blocks]
        for key in per_block[0]:
            result[key + "_block5_mean"] = float(np.mean([row[key] for row in per_block]))
        result[name + "_iav_block5_mean"] = float(np.mean(np.sum(np.abs(blocks), axis=1) / rate))
    if not result or not np.isfinite(list(result.values())).all():
        raise ValueError("Missing ACC/EMG channels or nonfinite block statistics.")
    return result
