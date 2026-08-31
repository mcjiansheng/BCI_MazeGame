"""Signal-quality indicators for setup checks, not impedance estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


@dataclass(frozen=True)
class ChannelQuality:
    rms_uv: float
    peak_to_peak_uv: float
    flatline_fraction: float
    line_noise_ratio: float
    finite_fraction: float
    status: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _band_power(frequencies: np.ndarray, spectrum: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies <= high)
    if mask.sum() < 2:
        return 0.0
    return float(trapezoid(spectrum[mask], frequencies[mask]))


def assess_signal_quality(data: np.ndarray, sample_rate: int) -> list[ChannelQuality]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be channels x samples")
    results: list[ChannelQuality] = []
    for channel in values:
        finite = np.isfinite(channel)
        finite_fraction = float(finite.mean()) if channel.size else 0.0
        clean = channel[finite]
        if clean.size < max(16, sample_rate // 2):
            results.append(ChannelQuality(0.0, 0.0, 1.0, 1.0, finite_fraction, "bad"))
            continue
        centered = clean - np.median(clean)
        rms = float(np.sqrt(np.mean(centered**2)))
        peak_to_peak = float(np.ptp(clean))
        tolerance = max(1e-6, np.std(clean) * 1e-4)
        flatline = float(np.mean(np.abs(np.diff(clean)) <= tolerance))
        frequencies, spectrum = welch(
            centered,
            fs=sample_rate,
            nperseg=min(clean.size, sample_rate * 2),
        )
        signal_power = _band_power(frequencies, spectrum, 4.0, 40.0)
        line_power = _band_power(frequencies, spectrum, 48.0, 52.0)
        line_ratio = line_power / max(signal_power + line_power, 1e-12)
        if finite_fraction < 1.0 or flatline > 0.2 or peak_to_peak < 1.0 or peak_to_peak > 1000.0:
            status = "bad"
        elif line_ratio > 0.35 or peak_to_peak > 500.0:
            status = "warning"
        else:
            status = "good"
        results.append(
            ChannelQuality(rms, peak_to_peak, flatline, line_ratio, finite_fraction, status)
        )
    return results


def format_quality_table(channel_names: tuple[str, ...], quality: list[ChannelQuality]) -> str:
    rows = ["channel status rms_uV p2p_uV flat_% line50_%"]
    for name, item in zip(channel_names, quality):
        rows.append(
            f"{name:>7} {item.status:>7} {item.rms_uv:7.1f} {item.peak_to_peak_uv:7.1f} "
            f"{item.flatline_fraction * 100:6.1f} {item.line_noise_ratio * 100:8.1f}"
        )
    return "\n".join(rows)
