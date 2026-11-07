"""Nhu's basic G2 provider for Mai's locked physiological68 feature interface.

Official synchronized PKL only. This provider owns native-rate filtering,
calibration and masks; Mai's feature kernels receive prepared channels once.
Raw112 and the earlier G2 diagnostic processor keep their original definitions.
"""

import math

import numpy as np
from scipy.signal import find_peaks

from wesad.features import Channel, VERSION, detect_beats, detect_scr, finite_segments
from wesad.multisensor import SENSOR_SPECS
from wesad.preprocessing_g2 import filter_finite_segments


def _filtered(values, rate, kind, cutoff):
    # Keep the full padded finite segment for cascaded filters and detection.
    # Eligibility masks remove margins once on the original finite segment.
    return filter_finite_segments(values, rate,
                                  {"kind": kind, "cutoff_hz": cutoff, "edge_seconds": 0})


def _interior(covered, rate, seconds):
    usable = np.zeros(len(covered), dtype=bool)
    edge = int(math.ceil(rate * seconds))
    for first, last in finite_segments(covered):
        if last - first > 2 * edge:
            usable[first + edge:last - edge if edge else last] = True
    return usable


def prepare_subject(data, config, contract):
    """Return ten native-rate Channels as required by features.process_subject.

    Methods are the explicit heuristic design in the versioned contract, not
    a physiological validation. Missing raw samples are never interpolated.
    """
    if (config.get("input_units") != "official_wesad_synchronized_pkl"
            or config.get("feature_version") != VERSION
            or contract.get("feature_version") != VERSION):
        raise ValueError("Provider requires the physiological68 official synchronized PKL contract.")
    methods = contract["methods"]
    edges = methods["filter"]["edges_seconds_each_end"]
    channels = {}
    for device, sensors in SENSOR_SPECS.items():
        for sensor, (rate, axes) in sensors.items():
            key = f"{device}_{sensor.lower()}"
            original = np.asarray(data["signal"][device][sensor])
            if original.ndim == 1 and axes == 1:
                original = original[:, None]
            if (original.ndim != 2 or original.shape[1] != axes or not len(original)
                    or not np.issubdtype(original.dtype, np.number)):
                raise ValueError(f"{key}: expected nonempty numeric (n, {axes}) array.")
            values = original.astype(np.float64, copy=True)
            covered = np.isfinite(values).all(axis=1)
            # Arrays preserve the original sample indices, including invalid rows.
            if axes == 1:
                values = values[:, 0]
            elif device == "wrist":
                # Explicit contract calibration, applied once to a new array.
                values /= 64.0
            channel = Channel(values, rate, covered, covered.copy())
            kind = sensor.lower()
            if kind == "ecg":
                channel.values = _filtered(values, rate, "bandpass", methods["ecg"]["bandpass_hz"])
                detection = _filtered(values, rate, "bandpass", methods["ecg"]["detection_bandpass_hz"])
                channel.usable = (_interior(covered, rate, edges["ecg"])
                                  & np.isfinite(channel.values) & np.isfinite(detection))
                channel.peaks = detect_beats(detection, rate, channel.usable, ecg=True)
            elif kind == "bvp":
                channel.values = _filtered(values, rate, "bandpass", methods["bvp"]["bandpass_hz"])
                channel.usable = _interior(covered, rate, edges["bvp"]) & np.isfinite(channel.values)
                channel.peaks = detect_beats(channel.values, rate, channel.usable)
            elif kind == "emg":
                channel.values = _filtered(values, rate, "bandpass", methods["acc_emg"]["emg_bandpass_hz"])
                channel.usable = _interior(covered, rate, edges["emg"]) & np.isfinite(channel.values)
            elif kind == "eda":
                eda = methods["eda"]
                channel.values = _filtered(values, rate, "lowpass", eda["clean_lowpass_hz"][device])
                channel.tonic = _filtered(channel.values, rate, "lowpass", eda["tonic_lowpass_hz"])
                channel.phasic = channel.values - channel.tonic
                margin = max(edges["eda_cleaned"], edges["eda_tonic_phasic"])
                channel.usable = (_interior(covered, rate, margin) & np.isfinite(channel.values)
                                  & np.isfinite(channel.tonic) & np.isfinite(channel.phasic))
                channel.events = detect_scr(channel.phasic, rate, channel.usable)
            elif kind == "resp":
                channel.values = _filtered(values, rate, "bandpass", methods["resp"]["bandpass_hz"])
                channel.usable = _interior(covered, rate, edges["resp"]) & np.isfinite(channel.values)
                peaks = []
                for first, last in finite_segments(np.isfinite(channel.values)):
                    segment = channel.values[first:last]
                    found, _ = find_peaks(segment, distance=int(math.ceil(2 * rate)),
                                          prominence=max(.5 * float(segment.std()), 1e-12))
                    peaks.extend(int(first + p) for p in found if channel.usable[first + p])
                channel.peaks = np.asarray(peaks, dtype=int)
            channels[key] = channel
    return channels
