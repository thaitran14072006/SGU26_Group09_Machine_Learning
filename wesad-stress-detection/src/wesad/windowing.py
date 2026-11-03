"""Cắt cửa sổ cùng mốc thời gian, giữ tần số gốc từng cảm biến."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def sample_count(seconds, sampling_rate_hz):
    if (not np.isfinite(seconds) or seconds < 0 or not np.isfinite(sampling_rate_hz)
            or sampling_rate_hz <= 0):
        raise ValueError("Time must be nonnegative and sampling rate positive.")
    count = seconds * sampling_rate_hz
    if abs(count - round(count)) > 1e-7:
        raise ValueError("Time grid must map to integer native sample indices.")
    return int(round(count))


@dataclass(frozen=True)
class SampledSignal:
    values: np.ndarray
    sampling_rate_hz: float


@dataclass(frozen=True)
class AlignedWindow:
    start_index: int  # index on the original label timeline
    start_s: float
    end_s: float
    label: int
    label_fraction: float
    signals: dict[str, SampledSignal]

    def metadata(self, subject_id):
        return {"subject_id": subject_id, "window_id": f"{subject_id}_{self.start_index}",
                "start_s": self.start_s, "end_s": self.end_s,
                "label_original": self.label, "label_binary": int(self.label == 2),
                "label_fraction": self.label_fraction,
                "quality_flag": "finite_alignment_only_not_physiological_QA"}


@dataclass
class WindowAudit:
    candidates: int = 0
    label_rejected: int = 0
    signal_rejected: int = 0
    accepted: int = 0
    reasons: Counter = field(default_factory=Counter)


def iter_multirate_windows(signals, labels, *, label_sampling_rate_hz=700,
                          window_seconds=60, step_seconds=5,
                          allowed_labels=(1, 2, 3), min_label_fraction=0.9,
                          audit=None):
    """Common complete-case windows; all requested channels share one set.

    Audit counts once per rejected window and once per channel/reason.
    No resampling, interpolation, learned normalization or physiological QA.
    Exhaust the iterator before reading final audit counts.
    """
    if not signals or not set(allowed_labels) or set(allowed_labels) - {1, 2, 3}:
        raise ValueError("Need signals and target labels within {1, 2, 3}.")
    if not np.isfinite(min_label_fraction) or not 0.5 < min_label_fraction <= 1:
        raise ValueError("min_label_fraction must be in (0.5, 1].")
    size = sample_count(window_seconds, label_sampling_rate_hz)
    step = sample_count(step_seconds, label_sampling_rate_hz)
    if min(size, step) < 1:
        raise ValueError("Window and step must be positive.")
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError("Labels must be a nonempty finite integer vector.")
    for name, signal in signals.items():
        x = np.asarray(signal.values)
        if x.ndim not in (1, 2) or not x.size:
            raise ValueError(f"{name}: expected nonempty samples x axes.")
        sample_count(window_seconds, signal.sampling_rate_hz)
        sample_count(step_seconds, signal.sampling_rate_hz)
    audit = audit if audit is not None else WindowAudit()
    for start in range(0, len(y) - size + 1, step):
        audit.candidates += 1
        values, counts = np.unique(y[start:start + size], return_counts=True)
        index = int(counts.argmax())
        label, fraction = int(values[index]), float(counts[index] / size)
        if label not in allowed_labels or fraction < min_label_fraction:
            audit.label_rejected += 1
            continue
        start_s, end_s = start / label_sampling_rate_hz, (start + size) / label_sampling_rate_hz
        chunks, reasons = {}, []
        for name, signal in signals.items():
            left = sample_count(start_s, signal.sampling_rate_hz)
            right = sample_count(end_s, signal.sampling_rate_hz)
            if right > len(signal.values):
                reasons.append(f"{name}:incomplete")
                continue
            chunk = np.asarray(signal.values)[left:right]
            if not np.isfinite(chunk).all():
                reasons.append(f"{name}:nonfinite_or_filter_edge")
            else:
                chunks[name] = SampledSignal(chunk, signal.sampling_rate_hz)
        if reasons:
            audit.signal_rejected += 1
            audit.reasons.update(reasons)
            continue
        audit.accepted += 1
        yield AlignedWindow(start, start_s, end_s, label, fraction, chunks)
