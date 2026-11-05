"""Chia cua so tren truc thoi gian goc; chua loc nhieu sinh ly."""

from dataclasses import dataclass
from collections.abc import Iterator, Mapping, Sequence

import numpy as np


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
    step_seconds: float = 5,
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
