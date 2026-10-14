"""Torsional vibration and stick-slip dynamics analysis.

Implements:
- Stick-Slip Severity Index (SSI) calculation and classification.
- Drillstring torsional natural resonance frequency (f0 = c_s / (4 * L)).
- High-frequency FFT spectral decomposition and Welch Power Spectral Density (PSD).
- Resonance proximity detection.

References:
    - Kyllingstad, A., & Halsey, G. W. (1988). A Study of Drillstring Torque Fluctuations.
      SPE Drilling Engineering, 3(04), 367-374.
    - Brett, J. F. (1992). The Genesis of Torsional Drillstring Vibrations.
      SPE Drilling Engineering, 7(03), 168-174.
    - Jansen, J. D., & van den Steen, L. (1995). Active Damping of Torsional Drillstring
      Vibrations. SPE Drilling & Completion, 10(04), 250-254.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

try:
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Shear wave velocity in steel: ~3170 m/s (~10,400 ft/s)
STEEL_SHEAR_WAVE_VELOCITY_M_S = 3170.0
STEEL_SHEAR_WAVE_VELOCITY_FT_S = 10400.26


@dataclass
class SpectralAnalysisResult:
    """Spectral analysis outcome container."""
    frequencies: np.ndarray
    spectrum: np.ndarray
    dominant_frequency: float
    peak_value: float
    total_power: float
    spectrum_type: str  # 'fft_amplitude' or 'welch_psd'


def compute_stick_slip_index(
    rpm_max: Union[float, int, np.ndarray, pd.Series],
    rpm_min: Union[float, int, np.ndarray, pd.Series],
    rpm_avg: Union[float, int, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Stick-Slip Severity Index (SSI).

    Formula:
        SSI = (RPM_max - RPM_min) / (2 * RPM_avg)

    Physical Interpretation:
        - SSI = 0.0: Perfectly steady rotation (no torsional oscillation).
        - SSI < 0.5: Mild torsional vibration (safe operational regime).
        - 0.5 <= SSI < 1.0: Moderate stick-slip (cyclic deceleration/acceleration).
        - SSI >= 1.0: Severe stick-slip (bit comes to a complete halt periodically).
        - SSI > 1.0: Critical stick-slip with backward / reverse rotation (RPM_min < 0).

    Args:
        rpm_max: Maximum RPM during the evaluation window.
        rpm_min: Minimum RPM during the evaluation window.
        rpm_avg: Mean RPM during the evaluation window.

    Returns:
        Stick-Slip Severity Index (dimensionless). Returns np.nan when rpm_avg <= 0.
    """
    is_series = any(isinstance(x, pd.Series) for x in (rpm_max, rpm_min, rpm_avg))
    series_index = None
    for x in (rpm_max, rpm_min, rpm_avg):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (rpm_max, rpm_min, rpm_avg)
    )

    max_arr = np.asarray(rpm_max, dtype=float)
    min_arr = np.asarray(rpm_min, dtype=float)
    avg_arr = np.asarray(rpm_avg, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = avg_arr > 0
        diff = max_arr - min_arr
        ssi = np.where(valid, diff / (2.0 * avg_arr), np.nan)

    if is_scalar:
        return float(ssi) if not np.isnan(ssi) else float("nan")
    if is_series:
        return pd.Series(ssi, index=series_index)
    return ssi


def compute_rolling_stick_slip_index(
    rpm: Union[np.ndarray, pd.Series],
    window_samples: int = 50,
) -> Union[np.ndarray, pd.Series]:
    """Calculate rolling Stick-Slip Severity Index over a sliding window.

    Args:
        rpm: Continuous RPM time series.
        window_samples: Number of samples in the rolling calculation window.

    Returns:
        Rolling SSI series matching input type.
    """
    if window_samples < 2:
        raise ValueError("window_samples must be at least 2.")

    is_series = isinstance(rpm, pd.Series)
    s = pd.Series(rpm) if not is_series else rpm

    rolling = s.rolling(window=window_samples, min_periods=window_samples // 2)
    rpm_max = rolling.max()
    rpm_min = rolling.min()
    rpm_avg = rolling.mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 2.0 * rpm_avg
        ssi_series = np.where(denom > 0, (rpm_max - rpm_min) / denom, np.nan)

    if is_series:
        return pd.Series(ssi_series, index=s.index)
    return np.asarray(ssi_series)


def classify_stick_slip_severity(
    ssi: Union[float, int, np.ndarray, pd.Series],
) -> Union[str, np.ndarray, pd.Series]:
    """Classify stick-slip severity according to standard industry thresholds.

    Thresholds:
        - SSI < 0.5: 'Mild'
        - 0.5 <= SSI < 1.0: 'Moderate'
        - SSI >= 1.0: 'Severe'

    Args:
        ssi: Stick-Slip Severity Index value(s).

    Returns:
        Severity string ('Mild', 'Moderate', 'Severe') or array/Series thereof.
    """
    is_series = isinstance(ssi, pd.Series)
    series_index = getattr(ssi, "index", None)
    is_scalar = not isinstance(ssi, (np.ndarray, pd.Series, list, tuple))

    ssi_arr = np.asarray(ssi, dtype=float)

    def _cat(val: float) -> str:
        if np.isnan(val):
            return "Unknown"
        if val < 0.5:
            return "Mild"
        elif val < 1.0:
            return "Moderate"
        else:
            return "Severe"

    if is_scalar:
        return _cat(float(ssi_arr))

    vfunc = np.vectorize(_cat, otypes=[object])
    res = vfunc(ssi_arr)

    if is_series:
        return pd.Series(res, index=series_index)
    return res


def drillstring_torsional_natural_frequency(
    drillstring_length: float,
    shear_wave_velocity: float = STEEL_SHEAR_WAVE_VELOCITY_M_S,
    length_unit: str = "m",
    harmonics_count: int = 3,
) -> Tuple[float, float, List[float]]:
    """Estimate drillstring natural torsional resonance frequency and harmonics.

    For a drillpipe string fixed at surface (rotary table / top drive) and free/cutting
    at bottom, the fundamental torsional resonance frequency f0 (quarter-wave mode) is:
        f0 = c_s / (4 * L)

    Harmonics (quarter-wave odd modes):
        f_n = (2n - 1) * f0,  for n = 1, 2, 3...

    Args:
        drillstring_length: Measured drillstring length (L).
        shear_wave_velocity: Speed of shear waves in drillstring material (default ~3170 m/s for steel).
        length_unit: Length unit ('m' for meters or 'ft' for feet).
        harmonics_count: Number of harmonic frequencies to compute (default 3).

    Returns:
        Tuple containing:
            - f0: Fundamental resonance frequency in Hz.
            - T0: Fundamental period in seconds (1 / f0).
            - harmonics: List of first N resonance frequencies [f0, f1, f2, ...] in Hz.

    Raises:
        ValueError: If drillstring_length <= 0 or shear_wave_velocity <= 0.
    """
    if drillstring_length <= 0:
        raise ValueError("drillstring_length must be strictly positive.")
    if shear_wave_velocity <= 0:
        raise ValueError("shear_wave_velocity must be strictly positive.")

    # Harmonize units
    if length_unit.lower() in ("ft", "feet"):
        l_meters = drillstring_length * 0.3048
    elif length_unit.lower() in ("m", "meters"):
        l_meters = drillstring_length
    else:
        raise ValueError(f"Unsupported length_unit '{length_unit}'. Use 'm' or 'ft'.")

    # Fundamental frequency: f0 = c_s / (4 * L)
    f0 = shear_wave_velocity / (4.0 * l_meters)
    t0 = 1.0 / f0

    harmonics = [(2 * n - 1) * f0 for n in range(1, harmonics_count + 1)]
    return f0, t0, harmonics


def analyze_torsional_fft(
    signal: Union[np.ndarray, pd.Series, Sequence[float]],
    sampling_rate: float,
    detrend: bool = True,
) -> SpectralAnalysisResult:
    """Perform Fast Fourier Transform (FFT) on high-frequency torque or RPM time-series.

    Args:
        signal: 1D array-like signal (e.g. Surface Torque in ft-lbf or Downhole RPM).
        sampling_rate: Sampling frequency in Hertz (Hz, samples per second).
        detrend: If True, subtract the signal mean to eliminate zero-frequency DC offset.

    Returns:
        SpectralAnalysisResult containing frequencies, amplitude spectrum, dominant frequency,
        peak amplitude, and total power.
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be strictly positive.")

    data = np.asarray(signal, dtype=float)
    n = len(data)
    if n < 4:
        raise ValueError("Signal length must be at least 4 samples for FFT analysis.")

    # Detrend
    if detrend:
        data = data - np.mean(data)

    # Windowing (Hann window to reduce spectral leakage)
    window = np.hanning(n)
    windowed_data = data * window
    scale_factor = 2.0 / np.sum(window)

    # Real FFT
    fft_vals = np.fft.rfft(windowed_data)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    amplitudes = np.abs(fft_vals) * scale_factor

    # Ignore zero-frequency DC component when finding peak
    if len(frequencies) > 1:
        peak_idx = 1 + np.argmax(amplitudes[1:])
        dom_freq = float(frequencies[peak_idx])
        peak_amp = float(amplitudes[peak_idx])
    else:
        dom_freq = 0.0
        peak_amp = float(amplitudes[0])

    total_power = float(np.sum(amplitudes ** 2))

    return SpectralAnalysisResult(
        frequencies=frequencies,
        spectrum=amplitudes,
        dominant_frequency=dom_freq,
        peak_value=peak_amp,
        total_power=total_power,
        spectrum_type="fft_amplitude",
    )


def analyze_torsional_welch(
    signal: Union[np.ndarray, pd.Series, Sequence[float]],
    sampling_rate: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> SpectralAnalysisResult:
    """Perform Welch Power Spectral Density (PSD) analysis on torsional drilling signals.

    Welch's method splits the signal into overlapping segments and averages their
    periodograms, providing significantly lower variance than raw FFT for noisy drilling data.

    Args:
        signal: High-frequency torque or RPM series.
        sampling_rate: Sampling rate in Hz.
        nperseg: Length of each Welch segment. If None, min(256, len(signal)) is used.
        noverlap: Number of points of overlap between segments. If None, nperseg // 2.

    Returns:
        SpectralAnalysisResult with frequencies, PSD (unit^2/Hz), dominant peak frequency,
        peak PSD value, and total spectral power.
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be strictly positive.")

    data = np.asarray(signal, dtype=float)
    n = len(data)
    if n < 4:
        raise ValueError("Signal length must be at least 4 samples for PSD analysis.")

    if nperseg is None:
        nperseg = min(256, n)
    else:
        nperseg = min(nperseg, n)

    if SCIPY_AVAILABLE:
        freqs, psd = sp_signal.welch(
            data,
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
        )
    else:
        # Fallback periodogram if scipy is not available
        detrended = data - np.mean(data)
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
        psd = (np.abs(fft_vals) ** 2) / (n * sampling_rate)

    # Exclude DC component
    if len(freqs) > 1:
        peak_idx = 1 + np.argmax(psd[1:])
        dom_freq = float(freqs[peak_idx])
        peak_psd = float(psd[peak_idx])
    else:
        dom_freq = 0.0
        peak_psd = float(psd[0])

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_power = float(np.sum(psd) * df)

    return SpectralAnalysisResult(
        frequencies=freqs,
        spectrum=psd,
        dominant_frequency=dom_freq,
        peak_value=peak_psd,
        total_power=total_power,
        spectrum_type="welch_psd",
    )


def check_resonance_risk(
    observed_frequency: float,
    drillstring_length: float,
    tolerance: float = 0.15,
    shear_wave_velocity: float = STEEL_SHEAR_WAVE_VELOCITY_M_S,
    length_unit: str = "m",
) -> Dict[str, Union[bool, float, str]]:
    """Determine whether an observed oscillation frequency aligns with drillstring natural resonance.

    Args:
        observed_frequency: Dominant peak frequency observed in FFT/Welch (Hz).
        drillstring_length: Current measured length of drillstring.
        tolerance: Fractional frequency tolerance band (default 0.15 = +/-15%).
        shear_wave_velocity: Shear wave velocity in drillstring.
        length_unit: 'm' or 'ft'.

    Returns:
        Dictionary with:
            - 'is_resonance': Boolean
            - 'f0_fundamental': Fundamental theoretical frequency (Hz)
            - 'matched_mode': Description (e.g. 'fundamental', 'harmonic_2', 'none')
            - 'frequency_error_pct': Percentage difference from closest mode
    """
    f0, _, harmonics = drillstring_torsional_natural_frequency(
        drillstring_length=drillstring_length,
        shear_wave_velocity=shear_wave_velocity,
        length_unit=length_unit,
        harmonics_count=3,
    )

    modes = [("fundamental", f0)] + [
        (f"harmonic_{i + 1}", h) for i, h in enumerate(harmonics[1:], start=1)
    ]

    best_mode = "none"
    min_err = float("inf")
    is_res = False

    for mode_name, freq in modes:
        err = abs(observed_frequency - freq) / freq
        if err < min_err:
            min_err = err
            best_mode = mode_name
        if err <= tolerance:
            is_res = True
            break

    return {
        "is_resonance": is_res,
        "f0_fundamental": f0,
        "matched_mode": best_mode if is_res else "none",
        "frequency_error_pct": min_err * 100.0,
    }
