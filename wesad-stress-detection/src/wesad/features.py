"""G2 physiological_68_v1: feature kernels and a G1-design reference adapter.

The adapter is an integration aid, not a claim that Nhu has delivered/reviewed G2.
Filtering uses whole finite recording segments, before label/window selection.
All physiological detectors are fixed heuristics, not clinical signal validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, get_window, sosfiltfilt, welch

from wesad.data import load_subject, validate_features
from wesad.multisensor import SENSOR_SPECS
from wesad.preprocessing import iter_windows
from wesad.utils import file_hash, fingerprint, import_callable, read_json, resolve_path

VERSION = "physiological_68_v1"
# Binds this implementation's numeric choices and column meanings, independently
# of the config's file hash. Changing the design requires reviewing the kernels.
DESIGN_SIGNATURE = "9b4fa43becd779c7b908288a7c80bc68e3b28103887c3d29800b6ceeddf3f5cc"
META = ["subject_id", "window_id", "start_s", "end_s", "label_original",
        "label_binary", "label_fraction", "quality_flag"]
OK = "heuristic_checks_passed_not_physiological_validation"


@dataclass
class Channel:
    """Adapter interface. Indices and events are on this channel's native grid."""
    values: np.ndarray
    fs: int
    covered: np.ndarray
    usable: np.ndarray
    peaks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    tonic: np.ndarray | None = None
    phasic: np.ndarray | None = None
    events: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


def finite_segments(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def filter_segments(x, fs, cutoff, kind, edge_seconds):
    """Never filter across a missing sample. Short/edge regions remain unusable."""
    sos = butter(4, cutoff, btype=kind, fs=fs, output="sos")
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    values = np.full(len(x), np.nan)
    usable = np.zeros(len(x), dtype=bool)
    edge = int(math.ceil(edge_seconds * fs))
    for a, b in finite_segments(np.isfinite(x)):
        if b - a <= padlen or b - a <= 2 * edge:
            continue
        values[a:b] = sosfiltfilt(sos, x[a:b], padtype="odd", padlen=int(padlen))
        usable[a + edge:b - edge] = True
    return values, usable


def detect_beats(x, fs, usable, ecg=False):
    result = []
    distance = int(math.ceil(.30 * fs))
    # Use unmasked finite segments for filtering/detection, then restrict peaks.
    for a, b in finite_segments(np.isfinite(x)):
        y = x[a:b]
        if ecg:
            width = int(round(.10 * fs))
            if len(y) < width:
                continue
            envelope = np.convolve(y * y, np.ones(width) / width, mode="same")
            candidates, _ = find_peaks(envelope, distance=distance,
                                      prominence=max(.5 * float(envelope.std()), 1e-12))
            refined = {max(0, p - width) + int(np.argmax(np.abs(y[max(0, p-width):min(len(y), p+width+1)])))
                       for p in candidates}
            # Deterministic amplitude priority; adjacent accepted peaks >=300ms.
            occupied = np.zeros(len(y), dtype=bool)
            selected = []
            for p in sorted(refined, key=lambda i: (-abs(y[i]), i)):
                if not occupied[p]:
                    selected.append(p)
                    occupied[max(0, p-distance+1):min(len(y), p+distance)] = True
            peaks = np.sort(selected).astype(int)
        else:
            peaks, _ = find_peaks(y, distance=distance,
                                 prominence=max(.5 * float(y.std()), 1e-12))
        result.extend(int(a + p) for p in peaks if usable[a + p])
    return np.asarray(result, dtype=int)


def detect_scr(phasic, fs, usable):
    events = []
    for a, b in finite_segments(np.isfinite(phasic)):
        y = phasic[a:b]
        peaks, _ = find_peaks(y, distance=int(math.ceil(fs)), prominence=.01)
        previous = -1
        for peak in peaks:
            left = max(0, peak - int(4 * fs), previous + 1)
            if left >= peak:
                continue
            region = y[left:peak]
            # Last argmin resolves a flat onset deterministically.
            onset = left + len(region) - 1 - int(np.argmin(region[::-1]))
            rise = (peak - onset) / fs
            amplitude = float(y[peak] - y[onset])
            if .25 <= rise <= 4 and amplitude >= .01 and usable[a+onset:a+peak+1].all():
                events.append((a + onset, a + peak, amplitude))
                previous = peak
    return np.asarray(events, dtype=float).reshape(-1, 3)


def prepare_subject(data, config, contract):
    """Reference input adapter following G1. Replace via preprocessing_provider.

    Input is official synchronized PKL: chest is already calibrated, wrist ACC
    is still in 1/64g. Conversion occurs exactly once here, never in kernels.
    Returns every required channel; keeps rates, signal coverage and edge masks.
    """
    channels = {}
    methods = contract["methods"]
    n_labels = len(data["label"])
    for device, sensors in SENSOR_SPECS.items():
        for sensor, (fs, axes) in sensors.items():
            try:
                raw = np.asarray(data["signal"][device][sensor])
            except KeyError as exc:
                raise ValueError(f"Missing required {device}/{sensor}") from exc
            if raw.ndim == 1 and axes == 1:
                raw = raw[:, None]
            if raw.ndim != 2 or raw.shape[1] != axes or not len(raw) or not np.issubdtype(raw.dtype, np.number):
                raise ValueError(f"Invalid {device}/{sensor} shape or dtype")
            # Short channels are accounted for by common-window rejection later.
            if len(raw) / fs > n_labels / 700 + 1/fs + 1/700:
                raise ValueError(f"{device}/{sensor} extends beyond synchronized label duration")
            x = np.array(raw, dtype=np.float64, copy=True)
            if sensor == "ACC" and device == "wrist":
                x /= 64.0
            covered = np.isfinite(x).all(axis=1)
            usable = covered.copy()
            key = f"{device}_{sensor.lower()}"
            if axes == 1:
                x = x[:, 0]
            if sensor in ("ECG", "BVP", "EMG", "Resp"):
                cutoffs = {"ECG": [5, 40], "BVP": [.5, 5], "EMG": [20, 250], "Resp": [.1, .35]}[sensor]
                edges = {"ECG": 10, "BVP": 10, "EMG": 1, "Resp": 30}[sensor]
                x, usable = filter_segments(x, fs, cutoffs, "bandpass", edges)
            channel = Channel(x, fs, covered, usable)
            if sensor == "ECG":
                detection, mask = filter_segments(np.asarray(raw[:, 0], dtype=float), fs, [5, 20], "bandpass", 10)
                channel.peaks = detect_beats(detection, fs, mask, ecg=True)
            elif sensor == "BVP":
                channel.peaks = detect_beats(x, fs, usable)
            elif sensor == "Resp":
                found = []
                for a, b in finite_segments(np.isfinite(x)):
                    y = x[a:b]
                    peaks, _ = find_peaks(y, distance=int(2*fs), prominence=max(.5*float(y.std()), 1e-12))
                    found.extend(int(a+p) for p in peaks if usable[a+p])
                channel.peaks = np.asarray(found, dtype=int)
            elif sensor == "EDA":
                x, mask = filter_segments(x, fs, methods["eda"]["clean_lowpass_hz"][device], "lowpass", 10)
                tonic, tonic_mask = filter_segments(x, fs, .05, "lowpass", 60)
                channel.values, channel.usable = x, mask & tonic_mask
                channel.tonic, channel.phasic = tonic, x - tonic
                channel.events = detect_scr(channel.phasic, fs, channel.usable)
            channels[key] = channel
    return channels


def interval_features(peaks, fs):
    rr = np.diff(peaks) / fs
    valid = (rr >= .30) & (rr <= 2.0)
    if valid.any():
        median = float(np.median(rr[valid]))
        valid &= np.abs(rr - median) <= .20 * median + 1e-12
    accepted = rr[valid]
    # Difference in native samples first: exactly 50ms must not count as >50ms
    # through floating subtraction of two rounded second-valued intervals.
    pairs = np.diff(np.diff(peaks))[valid[:-1] & valid[1:]] / fs
    if len(accepted) < 20 or (valid.mean() if len(rr) else 0) < .8 or len(pairs) < 10:
        return np.full(6, np.nan), "insufficient_or_unreliable_beats"
    rate = 60 / accepted
    return np.array([rate.mean(), rate.std(ddof=0), accepted.mean()*1000,
                     accepted.std(ddof=1)*1000, np.sqrt(np.mean(pairs**2))*1000,
                     np.mean(np.abs(pairs) > .05)]), None


def band_powers(x, fs, bands):
    length = 4 * fs
    f, power = welch(x, fs=fs, window=get_window("hann", length, fftbins=True),
                     nperseg=length, noverlap=2*fs, nfft=length, detrend="constant",
                     scaling="density", return_onesided=True, average="mean")
    return [float(np.trapezoid(power[(f >= a) & (f <= b)], f[(f >= a) & (f <= b)])) for a, b in bands]


def block_features(x, fs, emg=False):
    if len(x) != 60*fs:
        raise ValueError("ACC/EMG require exactly twelve complete 5-second blocks")
    blocks = x.reshape(12, 5*fs)
    if emg:
        measures = [np.sqrt(np.mean(blocks**2, axis=1)), np.mean(np.abs(blocks), axis=1),
                    np.abs(np.diff(blocks, axis=1)).sum(axis=1)/5, np.abs(blocks).sum(axis=1)/fs]
    else:
        measures = [blocks.mean(axis=1), blocks.std(axis=1, ddof=0), np.abs(blocks).sum(axis=1)/fs]
    return [float(values.mean()) for values in measures]


def window_features(channels, start, end, subject, contract):
    """Pure feature layer: accepts an adapter's native-rate Channel dictionary."""
    result, flags = {}, set()
    for key, channel in channels.items():
        fs = channel.fs
        a, b = (start*fs + 699)//700, (end*fs + 699)//700
        x = channel.values[a:b]
        sensor = key.split("_", 1)[1]
        names = [name for name in contract["feature_columns"] if name.startswith(key + "_")]
        flat = float(np.std(x)) <= 1e-10
        if sensor in ("ecg", "bvp"):
            if flat:
                result.update(dict.fromkeys(names, np.nan))
                flags.add(key + ":flat_or_unusable")
                continue
            peaks = channel.peaks[(channel.peaks >= a) & (channel.peaks < b)]
            values, reason = interval_features(peaks, fs)
            result.update(zip(names[:6], values))
            result.update(zip(names[6:], band_powers(x, fs, contract["methods"]["psd"]["bands_hz"][key])))
            if reason:
                flags.add(key + ":" + reason)
        elif sensor == "eda":
            if flat:
                result.update(dict.fromkeys(names, np.nan))
                flags.add(key + ":flat_or_unusable")
                continue
            tonic, phasic = channel.tonic[a:b], channel.phasic[a:b]
            events = channel.events
            events = events[(events[:, 0] >= a) & (events[:, 1] < b)]
            count = len(events)
            amp_mean = float(events[:, 2].mean()) if count else np.nan
            amp_max = float(events[:, 2].max()) if count else np.nan
            values = [tonic.mean(), tonic.std(ddof=0), phasic.mean(), phasic.std(ddof=0),
                      float(count), float(count), amp_mean, amp_max]
            result.update(zip(names, values))
            if not count:
                flags.add(key + ":no_scr_events")
        elif sensor == "resp":
            peaks = channel.peaks[(channel.peaks >= a) & (channel.peaks < b)]
            intervals = np.diff(peaks) / fs
            valid = (intervals >= 2) & (intervals <= 15)
            if flat or valid.sum() < 3 or (valid.mean() if len(valid) else 0) < .8:
                result.update(dict.fromkeys(names, np.nan))
                flags.add(key + ":insufficient_cycles")
            else:
                rates = 60 / intervals[valid]
                result.update(zip(names, [rates.mean(), rates.std(ddof=0)]))
        elif sensor == "temp":
            if subject == "S2" and key == "chest_temp":
                result.update(dict.fromkeys(names, np.nan))
                flags.add(key + ":suspect_contact")
            else:
                t = np.arange(len(x), dtype=float) / fs
                t -= t.mean()
                result.update(zip(names, [x.mean(), np.dot(t, x-x.mean()) / np.dot(t, t)]))
                if flat:
                    flags.add(key + ":flat_for_review")
        elif sensor == "acc":
            for i, axis in enumerate(("x", "y", "z", "magnitude")):
                values = x[:, i] if i < 3 else np.linalg.norm(x, axis=1)
                group = [n for n in names if n.startswith(f"{key}_{axis}_")]
                result.update(zip(group, block_features(values, fs)))
        elif sensor == "emg":
            if flat:
                result.update(dict.fromkeys(names, np.nan))
                flags.add(key + ":flat_or_unusable")
            else:
                result.update(zip(names, block_features(x, fs, emg=True)))
    if set(result) != set(contract["feature_columns"]):
        raise ValueError("Adapter/feature layer did not produce exactly the contract columns")
    return result, flags


def read_contract(config):
    path = resolve_path(config["feature_contract"])
    if file_hash(path) != config["feature_contract_sha256"]:
        raise ValueError("Feature contract changed; update the versioned config before extracting")
    contract = read_json(path)
    design = {"methods": contract["methods"], "features": [
        {k: v for k, v in f.items() if k not in ("status", "pending")} for f in contract["features"]]}
    if fingerprint(design) != DESIGN_SIGNATURE:
        raise ValueError("Contract design differs from this version's implemented kernels")
    if config.get("feature_version") != VERSION or contract["feature_version"] != VERSION:
        raise ValueError("Expected physiological_68_v1")
    for key, value in {"window_seconds": 60, "step_seconds": 5, "label_rule": "majority",
                       "min_label_fraction": .90, "labels": [1, 2, 3]}.items():
        if config.get(key) != value:
            raise ValueError(f"{key} differs from locked physiological contract")
    if config.get("input_units") != "official_wesad_synchronized_pkl":
        raise ValueError("Declare official_wesad_synchronized_pkl units; no scale guessing")
    return contract


def process_subject(path, config):
    contract = read_contract(config)
    data = load_subject(path)
    subject = Path(path).stem
    labels = np.asarray(data["label"])
    provider = import_callable(config["preprocessing_provider"]) if config.get("preprocessing_provider") else prepare_subject
    channels = provider(data, config, contract)
    expected = {f"{device}_{sensor.lower()}" for device, sensors in SENSOR_SPECS.items() for sensor in sensors}
    if set(channels) != expected:
        raise ValueError("Adapter must provide all ten chest/wrist channels for common-window comparison")
    for device, sensors in SENSOR_SPECS.items():
        for sensor, (fs, _) in sensors.items():
            channel = channels[f"{device}_{sensor.lower()}"]
            if channel.fs != fs or len(channel.values) != len(channel.covered) or len(channel.values) != len(channel.usable):
                raise ValueError("Invalid adapter native rate/coverage masks")
    # Release original PKL sensor arrays; only prepared signals and labels remain.
    del data
    qa = {"n_candidate_windows": max(0, (len(labels)-42000)//3500+1), "n_label_valid_windows": 0,
          "n_rejected_label": 0, "n_rejected_incomplete_signal": 0,
          "n_rejected_nonfinite_signal": 0, "n_rejected_filter_edge": 0,
          "n_kept_windows": 0, "n_flagged_windows": 0}
    rows = []
    for window in iter_windows({"labels": labels}, labels):
        qa["n_label_valid_windows"] += 1
        reason = None
        # Priority is stable and distinguishes raw missingness from filter edges.
        for channel in channels.values():
            a = (window.start_index*channel.fs + 699)//700
            b = (window.end_index*channel.fs + 699)//700
            if b > len(channel.values) or b-a != 60*channel.fs:
                reason = "n_rejected_incomplete_signal"
                break
            if not channel.covered[a:b].all():
                reason = "n_rejected_nonfinite_signal"
                break
        if reason is None:
            for channel in channels.values():
                a = (window.start_index*channel.fs + 699)//700
                b = (window.end_index*channel.fs + 699)//700
                if not channel.usable[a:b].all():
                    reason = "n_rejected_filter_edge"
                    break
        if reason:
            qa[reason] += 1
            continue
        features, flags = window_features(channels, window.start_index, window.end_index, subject, contract)
        if np.isinf(list(features.values())).any():
            raise ValueError(f"{subject}: infinite feature output")
        if any(np.isnan(v) for v in features.values()) and not flags:
            raise ValueError("Missing physiological feature requires an explicit quality reason")
        qa["n_flagged_windows"] += bool(flags)
        row = dict(zip(META, [subject, f"{subject}_{window.start_index}", window.start_s, window.end_s,
                             window.label, int(window.label == 2), window.label_fraction,
                             ";".join(sorted(flags)) if flags else OK]))
        row.update(features)
        rows.append(row)
    if not rows:
        raise ValueError(f"{subject}: no eligible common windows; QA={qa}")
    qa["n_rejected_label"] = qa["n_candidate_windows"] - qa["n_label_valid_windows"]
    qa["n_kept_windows"] = len(rows)
    frame = pd.DataFrame(rows, columns=contract["output_column_order"])
    validate_features(frame, contract["feature_columns"], [subject])
    qa["n_missing_feature_values"] = int(frame[contract["feature_columns"]].isna().sum().sum())
    frame.attrs["extraction_qa"] = qa
    frame.attrs["feature_contract"] = {key: contract[key] for key in
                                       ("schema_version", "feature_version", "feature_columns", "metadata_columns",
                                        "output_column_order", "features", "sensor_groups", "unit_policy", "methods")}
    return frame
