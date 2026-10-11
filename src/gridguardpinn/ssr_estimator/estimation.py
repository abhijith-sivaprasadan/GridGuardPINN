"""Independent time-domain pencil and frequency-domain Welch estimators."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import hankel, svd
from scipy.signal import find_peaks, peak_widths, welch


@dataclass
class Mode:
    frequency_hz: float
    decay_per_s: float | None
    damping_ratio: float | None
    amplitude: float
    amplitude_convention: str
    damping_note: str

    def as_dict(self):
        return asdict(self)


def _input(signal, sample_rate, band):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 64 or not np.all(np.isfinite(x)):
        raise ValueError("A finite one-dimensional signal of at least 64 samples is required")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 < band[0] < band[1] < sample_rate / 2:
        raise ValueError("Frequency band must lie strictly below Nyquist")
    return x - np.mean(x)


def matrix_pencil(signal, sample_rate, poles=12, band=(5.0, 55.0)):
    """Fit sum c_k exp(lambda_k t); amplitude is 2|c_k| at window start.

    Pole count is explicit, never selected using published target frequencies.
    Both growing and decaying modes are retained. Mean removal can add a pole;
    a constant column is included in the final full-window least-squares fit.
    """
    x = _input(signal, sample_rate, band)
    rows = min(160, len(x) // 3)
    if not isinstance(poles, int) or not 2 <= poles < rows:
        raise ValueError("poles must be an integer in [2, Hankel rows)")
    h = hankel(x[:rows], x[rows - 1:])
    u, singular, vh = svd(h[:, :-1], full_matrices=False)
    rank = min(poles, int(np.sum(singular > singular[0] * 1e-10)))
    if rank == 0:
        return []
    reduced = (u[:, :rank].T @ h[:, 1:] @ vh[:rank].T) / singular[:rank]
    roots = np.linalg.eigvals(reduced).astype(complex)
    roots = roots[np.abs(roots) > 1e-12]
    exponents = np.log(roots) * sample_rate
    # Extremely fast numerical poles make the Vandermonde overflow and are
    # outside the resolvable observation window; raise rather than hide them.
    time = np.arange(len(x)) / sample_rate
    if np.any(exponents.real * time[-1] > 600):
        raise ValueError("Unstable pencil fit; shorten the window or reduce pole order")
    basis = np.exp(time[:, None] * exponents)
    coefficients = np.linalg.lstsq(np.column_stack([basis, np.ones(len(x))]), x,
                                  rcond=None)[0][:-1]
    modes = []
    for exponent, coefficient in zip(exponents, coefficients):
        frequency = exponent.imag / (2 * np.pi)
        if band[0] <= frequency <= band[1]:
            decay = -exponent.real
            modes.append(Mode(float(frequency), float(decay),
                              float(decay / abs(exponent)), float(2 * abs(coefficient)),
                              "peak sinusoidal amplitude at analysis-window start",
                              "exponential fit; positive decay is damping, negative is growth"))
    return sorted(modes, key=lambda m: m.frequency_hz)


def welch_modes(signal, sample_rate, band=(5.0, 55.0), prominence=0.01):
    """Hann Welch peaks and local spectral RMS amplitudes, without target modes.

    Half-power damping is not identifiable for these short nonstationary
    ringdowns: no fabricated zeta is returned. Spectral widths are retained as
    diagnostics, not relabeled as modal damping.
    """
    x = _input(signal, sample_rate, band)
    nperseg = len(x) // 2
    frequencies, psd = welch(x, fs=sample_rate, window="hann", nperseg=nperseg,
                             noverlap=nperseg // 2, detrend="constant")
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    scale = np.max(psd[mask])
    if scale == 0:
        return [], {"bin_width_hz": sample_rate / nperseg, "widths_hz": []}
    peaks, _ = find_peaks(psd, prominence=prominence * scale)
    peaks = peaks[mask[peaks]]
    widths = peak_widths(psd, peaks, rel_height=0.5)[0] * sample_rate / nperseg
    modes = []
    spacing = sample_rate / nperseg
    for peak in peaks:
        log_power = np.log(np.maximum(psd[peak - 1:peak + 2], np.finfo(float).tiny))
        correction = 0.5 * (log_power[0] - log_power[2]) / (
            log_power[0] - 2 * log_power[1] + log_power[2])
        frequency = (peak + correction) * spacing
        # +/- two bins captures a Hann main lobe; cap at midpoint to neighboring
        # peaks so a resolved mode's energy is never counted twice.
        others = peaks[peaks != peak]
        radius = min(2, int(np.min(abs(others - peak)) // 2)) if len(others) else 2
        energy = np.sum(psd[max(0, peak - radius):peak + radius + 1]) * spacing
        modes.append(Mode(float(frequency), None, None, float(np.sqrt(2 * energy)),
                          "sqrt(2) times local Welch band RMS over the analysis window",
                          "unresolved: finite Hann window and nonstationary ringdown; "
                          "spectral width is not an independent decay measurement"))
    return modes, {"bin_width_hz": float(spacing), "widths_hz": widths.tolist(),
                   "prominence_fraction": prominence}


def compare_modes(modes, references, decrements=None, tolerance_hz=0.5):
    """One-to-one matching within a declared tolerance; misses remain in output."""
    available = list(modes)
    rows = []
    for index, reference in enumerate(references):
        nearest = min(available, key=lambda m: abs(m.frequency_hz - reference), default=None)
        matched = nearest is not None and abs(nearest.frequency_hz - reference) <= tolerance_hz
        row = {"mode": index + 1, "published_frequency_hz": float(reference),
               "status": "resolved" if matched else "missed", "estimate": None}
        if matched:
            candidates = [m.as_dict() for m in available
                          if abs(m.frequency_hz - reference) <= tolerance_hz]
            if len(candidates) > 1:
                row["status"] = "ambiguous"
                row["candidate_estimates"] = candidates
            available.remove(nearest)
            row.update(estimate=nearest.as_dict(),
                       absolute_frequency_error_hz=float(abs(nearest.frequency_hz - reference)),
                       frequency_error_percent=float(100 * abs(nearest.frequency_hz - reference)
                                                     / reference))
            if decrements is not None:
                reference_decay = float(decrements[index])
                row["published_decay_per_s"] = reference_decay
                row["absolute_decay_error_per_s"] = (
                    None if nearest.decay_per_s is None else abs(nearest.decay_per_s-reference_decay)
                )
                row["decay_error_percent"] = (
                    None if nearest.decay_per_s is None or reference_decay == 0
                    else 100 * abs(nearest.decay_per_s-reference_decay) / abs(reference_decay)
                )
        rows.append(row)
    return rows, [mode.as_dict() for mode in available]
