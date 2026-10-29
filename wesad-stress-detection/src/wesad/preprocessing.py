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


def iter_windows(
    signals: Mapping[str, np.ndarray],
    labels: np.ndarray,
    sampling_rate_hz: float = 700,
    window_seconds: float = 60,
    step_seconds: float = 60,
    allowed_labels: Sequence[int] = (1, 2, 3),
) -> Iterator[Window]:
    """Yield cua so day du, dong nhat nhan va co tin hieu huu han.

    Chi dung cho tin hieu da can chinh, cung tan so voi labels.
    Luoi cua so bat dau tai 0, giu nguyen truc thoi gian goc.
    Khong noi cac doan sau khi loai nhan; bo phan du cuoi ban ghi.
    Cua so chua NaN/Inf bi loai, nhung tin hieu phang van can QA rieng.
    Mang cua so la view neu co the; khong sao chep toan bo ban ghi.
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

    for start in range(0, len(labels) - window_size + 1, step_size):
        end = start + window_size
        segment_labels = labels[start:end]
        label = segment_labels[0]
        if label not in allowed or not np.all(segment_labels == label):
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
        )
