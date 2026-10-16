"""High-harmonic spectra from the dipole, the dipole velocity or the acceleration."""

from __future__ import annotations

from typing import Optional

import numpy as np

_POWER = {"dipole": 2, "velocity": 1, "acceleration": 0}


def harmonic_spectrum(
    time: np.ndarray,
    signal: np.ndarray,
    kind: str = "dipole",
    omega0: Optional[float] = None,
    window: str = "hann",
    pad: int = 8,
) -> tuple:
    """Emitted spectrum |omega^2 d(omega)|^2 of a uniformly sampled time series.

    Parameters
    ----------
    time, signal:
        Uniform time grid and the dipole <d>(t), its velocity or its
        acceleration (``kind``). The dipole is multiplied by omega^2 and the
        velocity by omega, so all three give the same acceleration form.
    omega0:
        Laser frequency; when given the frequency axis is returned in harmonic
        orders omega / omega0.
    window:
        ``"hann"`` (default) or ``"none"``.
    pad:
        Zero-padding factor of the FFT.

    Returns
    -------
    (frequency or harmonic order, spectrum)
    """
    if kind not in _POWER:
        raise ValueError(f"kind must be one of {sorted(_POWER)}")
    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("harmonic_spectrum needs a uniform time grid (interpolate first)")
    if window == "hann":
        y = y * np.hanning(len(y))
    elif window != "none":
        raise ValueError("window must be 'hann' or 'none'")
    n = max(1, int(pad)) * len(y)
    amp = np.fft.rfft(y, n) * dt
    w = 2.0 * np.pi * np.fft.rfftfreq(n, dt)
    spec = np.abs(w ** _POWER[kind] * amp) ** 2
    return (w / omega0 if omega0 else w), spec


def log_ratio(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Median |log10(a / b)| (in decades) over ``mask`` and where b > 0."""
    m = (b > 0) & (a > 0)
    if mask is not None:
        m &= mask
    return float(np.median(np.abs(np.log10(a[m] / b[m])))) if m.any() else np.nan


__all__ = ["harmonic_spectrum", "log_ratio"]
