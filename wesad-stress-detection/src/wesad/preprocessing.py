"""Cửa sổ ngực cũ và các hàm tiền xử lý theo tần số gốc cho G2.

Lọc mới chỉ chạy khi được gọi trực tiếp; không thay đổi extractor raw cũ.
"""

from dataclasses import dataclass
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class Window:
    start_index: int
    end_index: int
    start_s: float
    end_s: float
    label: int
    signals: dict[str, np.ndarray]
    label_fraction: float = 1.0


def iter_windows(
    signals: Mapping[str, np.ndarray],
    labels: np.ndarray,
    sampling_rate_hz: float = 700,
    window_seconds: float = 60,
    step_seconds: float = 60,
    allowed_labels: Sequence[int] = (1, 2, 3),
    label_rule: str = "homogeneous",
    min_label_fraction: float = 0.90,
) -> Iterator[Window]:
    """Trả về cửa sổ đủ độ dài, đạt quy tắc nhãn và có tín hiệu hữu hạn.

    Tín hiệu đầu vào đã căn chỉnh và có cùng tần số với labels.
    Lưới bắt đầu tại 0; phần dư cuối bản ghi và cửa sổ có NaN/Inf bị loại.
    Hỗ trợ nhãn đồng nhất hoặc nhãn đa số theo min_label_fraction.
    Kiểm tra này chưa phát hiện tín hiệu phẳng hoặc nhiễu sinh lý.
    """
    parameters = (sampling_rate_hz, window_seconds, step_seconds)
    if any(not np.isfinite(x) or x <= 0 for x in parameters):
        raise ValueError("Tan so, cua so va buoc truot phai duong va huu han.")
    sizes = [sampling_rate_hz * x for x in (window_seconds, step_seconds)]
    if any(not np.isclose(x, round(x)) for x in sizes):
        raise ValueError("Cua so/buoc truot phai tuong ung so diem nguyen.")
    window_size, step_size = [int(round(x)) for x in sizes]
    if min(window_size, step_size) < 1:
        raise ValueError("Cua so/buoc truot phai co it nhat mot diem.")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels phai la mang mot chieu.")
    if not signals:
        raise ValueError("Can it nhat mot tin hieu.")
    arrays = {name: np.asarray(value) for name, value in signals.items()}
    for name, value in arrays.items():
        if value.ndim not in (1, 2) or len(value) != len(labels):
            raise ValueError(f"Tin hieu {name} khong khop chieu/dai voi labels.")
    allowed = set(allowed_labels)
    if label_rule not in {"homogeneous", "majority"}:
        raise ValueError("label_rule must be homogeneous or majority.")
    if not np.isfinite(min_label_fraction) or not 0.5 < min_label_fraction <= 1:
        raise ValueError("min_label_fraction must be in (0.5, 1].")

    for start in range(0, len(labels) - window_size + 1, step_size):
        end = start + window_size
        segment_labels = labels[start:end]
        if label_rule == "homogeneous":
            label, fraction = segment_labels[0], 1.0
            if label not in allowed or not np.all(segment_labels == label):
                continue
        else:
            values, counts = np.unique(segment_labels, return_counts=True)
            position = int(counts.argmax())
            label, fraction = values[position], float(counts[position] / window_size)
            if label not in allowed or fraction < min_label_fraction:
                continue
        chunks = {name: value[start:end] for name, value in arrays.items()}
        if not all(np.isfinite(value).all() for value in chunks.values()):
            continue
        yield Window(
            start_index=start,
            end_index=end,
            start_s=start / sampling_rate_hz,
            end_s=end / sampling_rate_hz,
            label=int(label),
            signals=chunks,
            label_fraction=fraction,
        )


def filter_signal(values, sampling_rate_hz, *, kind, cutoff_hz, order=4, edge_seconds):
    """Lọc SOS hai chiều từng đoạn hữu hạn, giữ NaN tại khoảng thiếu/biên.

    Không nối qua NaN/Inf. edge_seconds là vùng bỏ ở mỗi đầu từng đoạn,
    không phải cam kết rằng mọi quá độ đã hết. Hàm không sửa đơn vị hay scale.
    Bộ lọc offline có dùng mẫu tương lai trong cùng bản ghi chưa gán nhãn.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (1, 2) or not x.size:
        raise ValueError("Expected nonempty samples or samples x axes.")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer.")
    if not np.isfinite(edge_seconds) or edge_seconds < 0:
        raise ValueError("edge_seconds must be finite and nonnegative.")
    cutoff = np.atleast_1d(np.asarray(cutoff_hz, dtype=float))
    expected = {"lowpass": 1, "highpass": 1, "bandpass": 2}
    if (kind not in expected or cutoff.ndim != 1 or cutoff.size != expected[kind]
            or not np.isfinite(cutoff).all() or (cutoff <= 0).any()
            or (cutoff >= sampling_rate_hz / 2).any() or (np.diff(cutoff) <= 0).any()):
        raise ValueError("Invalid cutoff; require ordered frequencies strictly below Nyquist.")
    sos = butter(order, cutoff if cutoff.size == 2 else cutoff[0],
                 btype=kind, fs=sampling_rate_hz, output="sos")
    padlen = int(3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                            (sos[:, 5] == 0).sum())))
    edge = int(np.ceil(edge_seconds * sampling_rate_hz))
    finite = np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(axis=1)
    boundaries = np.diff(np.r_[False, finite, False].astype(np.int8))
    result = np.full(x.shape, np.nan, dtype=float)
    for start, end in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
        if end - start <= max(padlen + 1, 2 * edge):
            continue
        filtered = sosfiltfilt(sos, x[start:end], axis=0, padtype="odd", padlen=padlen)
        stop = end - edge
        result[start + edge:stop] = filtered[edge:len(filtered) - edge if edge else None]
    return result


def prepare_signals(data, config):
    """Chuẩn bị các kênh gốc, không tự chuyển đổi đơn vị.

    Returns dict[str, SampledSignal]. Config lists device, sensor, rate and axes.
    A caller must verify physical units before using these arrays as g/mV/etc.
    """
    from wesad.windowing import SampledSignal

    channels = config.get("channels", {})
    if not channels:
        raise ValueError("Config needs channels.")
    result = {}
    for name, spec in channels.items():
        fs = spec["sampling_rate_hz"]
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"{name}: invalid sampling rate.")
        x = np.asarray(data["signal"][spec["device"]][spec["sensor"]])
        axes = spec["axes"]
        if x.ndim == 1 and axes == 1:
            pass
        elif x.ndim != 2 or x.shape[1] != axes:
            raise ValueError(f"{name}: expected samples x {axes} axes.")
        if not x.size:
            raise ValueError(f"{name}: empty signal.")
        # Raw channels may be views of the one loaded PKL; never modify them.
        if spec.get("filter"):
            x = filter_signal(x, fs, **spec["filter"])
        result[name] = SampledSignal(x, fs)
    return result


def five_second_features(values, sampling_rate_hz, *, sensor):
    """ACC/EMG 60 s -> 12 blocks of 5 s -> mean, in INPUT units.

    Names match Mai's draft without chest_/wrist_ prefix. No scaling, gravity
    removal, imputation or skipping bad blocks. Physical units remain the
    caller's responsibility. Nonfinite input must be handled by window QA.
    """
    from wesad.windowing import sample_count

    x = np.asarray(values, dtype=float)
    block_size = sample_count(5, sampling_rate_hz)
    if len(x) != 12 * block_size or not np.isfinite(x).all():
        raise ValueError("Expected finite full 60-second signal, no skipped blocks.")
    if sensor == "acc":
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("ACC must have three axes.")
        augmented = np.column_stack([x, np.linalg.norm(x, axis=1)])
        blocks = augmented.reshape(12, block_size, 4)
        result = {}
        for index, axis in enumerate(("x", "y", "z", "magnitude")):
            features = {"mean": blocks[:, :, index].mean(axis=1),
                        "std": blocks[:, :, index].std(axis=1, ddof=0),
                        "abs_integral": np.abs(blocks[:, :, index]).sum(axis=1) / sampling_rate_hz}
            for stat, value in features.items():
                result[f"acc_{axis}_5s_{stat}_mean"] = float(value.mean())
        return result
    if sensor != "emg" or (x.ndim != 1 and (x.ndim != 2 or x.shape[1] != 1)):
        raise ValueError("Choose acc or single-channel emg.")
    blocks = x.reshape(12, block_size)
    return {"emg_5s_rms_mean": float(np.sqrt(np.mean(blocks ** 2, axis=1)).mean()),
            "emg_5s_mav_mean": float(np.abs(blocks).mean(axis=1).mean()),
            "emg_5s_waveform_length_per_s_mean": float(np.abs(np.diff(blocks, axis=1)).sum(axis=1).mean() / 5),
            "emg_5s_abs_integral_mean": float(np.abs(blocks).sum(axis=1).mean() / sampling_rate_hz)}
