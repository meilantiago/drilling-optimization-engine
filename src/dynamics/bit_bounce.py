"""Axial vibration and bit bounce dynamics analysis.

Implements:
- Rolling standard deviation and variance of Weight on Bit (WOB).
- Dimensionless Axial Vibration Index (AVI).
- Detection and characterization of axial shock events (off-bottom bounce and impact spikes).

References:
    - Dareing, D. W., & Livesay, B. J. (1968). Longitudinal and Angular Drill String
      Vibrations with Roller Bit Drilling. Journal of Engineering for Industry, 90(4), 671-679.
    - Dunayevsky, V. A., Abbassian, F., & Judzis, A. (1993). Dynamic Stability of
      Drillstrings Under Random Excitation. SPE 26344.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class AxialShockSummary:
    """Summary of axial vibration and shock detection."""
    total_samples: int
    shock_events_count: int
    bounce_events_count: int
    shock_frequency_ratio: float
    max_impact_ratio: float
    mean_vibration_index: float
    max_vibration_index: float
    severity_category: str  # 'Low', 'Moderate', 'High', 'Severe'


def compute_axial_vibration_index(
    wob: Union[np.ndarray, pd.Series],
    window_samples: int = 50,
    use_variance: bool = False,
) -> Union[np.ndarray, pd.Series]:
    """Calculate the Axial Vibration Index (AVI) using rolling WOB dispersion.

    The normalized axial vibration index reflects relative load oscillation:
        AVI = sigma_WOB / mu_WOB  (coefficient of variation)
    Or if use_variance is True:
        AVI_var = (sigma_WOB / mu_WOB)^2

    Physical Interpretation:
        - AVI < 0.25: Smooth axial drilling (stable cutter engagement).
        - 0.25 <= AVI < 0.50: Moderate axial vibration.
        - 0.50 <= AVI < 1.00: High axial vibration, intermittent bit unloading.
        - AVI >= 1.00: Severe bit bounce; bit periodically loses contact with bottom.

    Args:
        wob: Continuous Weight on Bit array or Series (lbf or klbf).
        window_samples: Window length in data points for rolling statistics (default 50).
        use_variance: If True, returns normalized variance (AVI^2); default False (sigma/mean).

    Returns:
        Rolling axial vibration index series.
    """
    if window_samples < 2:
        raise ValueError("window_samples must be at least 2.")

    is_series = isinstance(wob, pd.Series)
    s = pd.Series(wob, dtype=float) if not is_series else wob.astype(float)

    rolling = s.rolling(window=window_samples, min_periods=max(2, window_samples // 3))
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = rolling_mean > 0
        avi = np.where(valid, rolling_std / rolling_mean, np.nan)
        if use_variance:
            avi = avi ** 2

    if is_series:
        return pd.Series(avi, index=s.index)
    return np.asarray(avi)


def classify_bit_bounce_severity(
    axial_index: Union[float, np.ndarray, pd.Series],
) -> Union[str, np.ndarray, pd.Series]:
    """Classify bit bounce severity from the Axial Vibration Index.

    Args:
        axial_index: Value(s) of AVI (sigma_WOB / mu_WOB).

    Returns:
        Categorical severity ('Low', 'Moderate', 'High', 'Severe') or array/Series.
    """
    is_series = isinstance(axial_index, pd.Series)
    series_index = getattr(axial_index, "index", None)
    is_scalar = not isinstance(axial_index, (np.ndarray, pd.Series, list, tuple))

    avi_arr = np.asarray(axial_index, dtype=float)

    def _classify_single(val: float) -> str:
        if np.isnan(val):
            return "Unknown"
        if val < 0.25:
            return "Low"
        elif val < 0.50:
            return "Moderate"
        elif val < 1.00:
            return "High"
        else:
            return "Severe"

    if is_scalar:
        return _classify_single(float(avi_arr))

    vfunc = np.vectorize(_classify_single, otypes=[object])
    res = vfunc(avi_arr)

    if is_series:
        return pd.Series(res, index=series_index)
    return res


def detect_axial_shocks(
    wob: Union[np.ndarray, pd.Series],
    window_samples: int = 50,
    shock_sigma_threshold: float = 2.5,
    off_bottom_ratio: float = 0.15,
    max_wob_rating: Optional[float] = None,
) -> Tuple[pd.DataFrame, AxialShockSummary]:
    """Detect and quantify axial bit bounce shocks and off-bottom detachment events.

    Identifies two critical shock phenomena:
    1. Off-bottom bounce: WOB collapses below (off_bottom_ratio * mean_WOB),
       indicating cutters lifting free of rock.
    2. Impact shock: WOB rebounds sharply above (mean_WOB + shock_sigma_threshold * std_WOB)
       or exceeds maximum design rating max_wob_rating, causing impact cutter fracturing.

    Args:
        wob: Time-series of Weight on Bit (e.g. lbf).
        window_samples: Rolling window size for dynamic baseline (default 50).
        shock_sigma_threshold: Multiplier for std dev to classify an impact spike (default 2.5).
        off_bottom_ratio: Fraction of mean WOB below which an off-bottom bounce is flagged (default 0.15).
        max_wob_rating: Absolute maximum allowable WOB limit.

    Returns:
        Tuple of (results_df, summary_metrics):
            - results_df: pd.DataFrame with rolling stats, boolean event flags, and impact ratios.
            - summary_metrics: AxialShockSummary containing aggregate event counts and severity.
    """
    if window_samples < 2:
        raise ValueError("window_samples must be at least 2.")

    s = pd.Series(wob, dtype=float)
    rolling = s.rolling(window=window_samples, min_periods=max(2, window_samples // 3))
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()
    rolling_var = rolling.var()

    with np.errstate(divide="ignore", invalid="ignore"):
        avi = np.where(rolling_mean > 0, rolling_std / rolling_mean, np.nan)
        impact_ratio = np.where(rolling_mean > 0, s / rolling_mean, np.nan)

    # Off-bottom bounce detection: WOB drops close to zero while drilling
    is_bounce = (s <= off_bottom_ratio * rolling_mean) & (rolling_mean > 0)

    # Impact spike detection: WOB spikes significantly above rolling baseline
    shock_cutoff = rolling_mean + shock_sigma_threshold * rolling_std
    is_shock = (s >= shock_cutoff) & (rolling_std > 0)

    if max_wob_rating is not None:
        is_shock |= (s >= max_wob_rating)

    df_out = pd.DataFrame({
        "wob": s,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_var": rolling_var,
        "axial_vibration_index": avi,
        "impact_ratio": impact_ratio,
        "is_bounce": is_bounce,
        "is_shock": is_shock,
    })

    # Aggregate metrics
    n_samples = len(s)
    n_bounce = int(is_bounce.sum())
    n_shock = int(is_shock.sum())
    shock_freq = (n_bounce + n_shock) / max(1, n_samples)
    max_imp = float(np.nanmax(impact_ratio)) if n_samples > 0 else 1.0
    mean_avi = float(np.nanmean(avi)) if n_samples > 0 else 0.0
    max_avi = float(np.nanmax(avi)) if n_samples > 0 else 0.0

    # Overall severity
    if max_avi >= 1.0 or shock_freq >= 0.10:
        overall_severity = "Severe"
    elif max_avi >= 0.50 or shock_freq >= 0.03:
        overall_severity = "High"
    elif max_avi >= 0.25:
        overall_severity = "Moderate"
    else:
        overall_severity = "Low"

    summary = AxialShockSummary(
        total_samples=n_samples,
        shock_events_count=n_shock,
        bounce_events_count=n_bounce,
        shock_frequency_ratio=shock_freq,
        max_impact_ratio=max_imp,
        mean_vibration_index=mean_avi,
        max_vibration_index=max_avi,
        severity_category=overall_severity,
    )

    return df_out, summary
