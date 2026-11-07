"""Describe missing BVP features without changing signals, thresholds or labels."""

import numpy as np

BVP_INTERVAL_COLUMNS = ["wrist_bvp_pulse_rate_mean_bpm", "wrist_bvp_pulse_rate_std_bpm",
                        "wrist_bvp_mean_ibi_ms", "wrist_bvp_sd_ibi_ms",
                        "wrist_bvp_rmssd_ibi_ms", "wrist_bvp_pnn50_ibi"]


def summarize_bvp(frame):
    """One row per subject/label present; distinguish missing intervals from PSD.

    This is descriptive QA. Missingness may depend on label; no decision about
    physiological validity or detector tuning follows from these counts alone.
    """
    if frame.empty or frame.duplicated(["subject_id", "window_id"]).any():
        raise ValueError("Empty table or duplicate subject/window keys.")
    if not frame.label_original.isin([1, 2, 3]).all():
        raise ValueError("Expected original labels 1/2/3.")
    interval = frame[BVP_INTERVAL_COLUMNS]
    if np.isinf(interval.to_numpy(dtype=float)).any():
        raise ValueError("BVP intervals contain Inf.")
    missing = interval.isna()
    if not (missing.all(axis=1) == missing.any(axis=1)).all():
        raise ValueError("The locked BVP interval group must be available or missing together.")
    flags = frame.quality_flag.fillna("").str.split(";")
    reason = "wrist_bvp:insufficient_or_unreliable_beats"
    flat = "wrist_bvp:flat_or_unusable"
    has_reason = flags.apply(lambda tokens: reason in tokens or flat in tokens)
    if not (missing.all(axis=1) == has_reason).all():
        raise ValueError("BVP interval missingness and quality reasons disagree.")
    rows = []
    for (subject, label), group in frame.groupby(["subject_id", "label_original"], sort=True):
        total = len(group)
        absent = int(group[BVP_INTERVAL_COLUMNS].isna().all(axis=1).sum())
        rows.append({"subject_id": subject, "label_original": int(label), "n_windows": total,
                     "n_bvp_interval_available": total - absent, "n_bvp_interval_missing": absent,
                     "bvp_interval_missing_fraction": absent / total})
    return rows
