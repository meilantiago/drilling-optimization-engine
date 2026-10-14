"""Tests for dynamics modules: Stick-Slip, Bit Bounce, and PDC Cutter Wear Model."""

import math
import numpy as np
import pandas as pd
import pytest

from src.dynamics.stick_slip import (
    SpectralAnalysisResult,
    analyze_torsional_fft,
    analyze_torsional_welch,
    check_resonance_risk,
    classify_stick_slip_severity,
    compute_rolling_stick_slip_index,
    compute_stick_slip_index,
    drillstring_torsional_natural_frequency,
)
from src.dynamics.bit_bounce import (
    AxialShockSummary,
    classify_bit_bounce_severity,
    compute_axial_vibration_index,
    detect_axial_shocks,
)
from src.dynamics.wear_model import (
    CutterWearState,
    compute_cumulative_dissipated_energy,
    estimate_cutter_wear_grade,
    evaluate_bit_wear_state,
)


# =====================================================================
# Stick-Slip Module Tests
# =====================================================================

def test_stick_slip_severity_index_scalar():
    """Test Stick-Slip Severity Index (SSI) calculation."""
    # Constant RPM (no stick slip)
    ssi_zero = compute_stick_slip_index(rpm_max=100.0, rpm_min=100.0, rpm_avg=100.0)
    assert ssi_zero == 0.0

    # Mild stick slip: fluctuates between 80 and 120 RPM (avg 100)
    # SSI = (120 - 80) / (2 * 100) = 40 / 200 = 0.2
    ssi_mild = compute_stick_slip_index(rpm_max=120.0, rpm_min=80.0, rpm_avg=100.0)
    assert pytest.approx(ssi_mild, rel=1e-5) == 0.2

    # Severe stick slip (bit stops): max 200, min 0, avg 100
    # SSI = (200 - 0) / (2 * 100) = 1.0
    ssi_severe = compute_stick_slip_index(rpm_max=200.0, rpm_min=0.0, rpm_avg=100.0)
    assert pytest.approx(ssi_severe, rel=1e-5) == 1.0

    # Critical stick slip with backward rotation: max 240, min -40, avg 100
    # SSI = (240 - (-40)) / (2 * 100) = 280 / 200 = 1.4
    ssi_critical = compute_stick_slip_index(rpm_max=240.0, rpm_min=-40.0, rpm_avg=100.0)
    assert pytest.approx(ssi_critical, rel=1e-5) == 1.4


def test_stick_slip_classification():
    """Test SSI classification against industry standard categories."""
    assert classify_stick_slip_severity(0.2) == "Mild"
    assert classify_stick_slip_severity(0.49) == "Mild"
    assert classify_stick_slip_severity(0.50) == "Moderate"
    assert classify_stick_slip_severity(0.85) == "Moderate"
    assert classify_stick_slip_severity(1.00) == "Severe"
    assert classify_stick_slip_severity(1.50) == "Severe"

    # Test array classification
    arr_ssi = np.array([0.1, 0.6, 1.2])
    cats = classify_stick_slip_severity(arr_ssi)
    assert list(cats) == ["Mild", "Moderate", "Severe"]


def test_rolling_stick_slip_index():
    """Test rolling window calculation of SSI."""
    # Synthetic oscillating RPM: 100 + 50 * sin(t)
    t = np.linspace(0, 10, 100)
    rpm_series = 100.0 + 50.0 * np.sin(2.0 * np.pi * t)
    rolling_ssi = compute_rolling_stick_slip_index(rpm_series, window_samples=20)

    assert len(rolling_ssi) == 100
    valid_vals = rolling_ssi[~np.isnan(rolling_ssi)]
    assert len(valid_vals) > 0
    # Expected SSI ~ 100 / 200 = 0.5
    assert np.all(valid_vals > 0)


def test_drillstring_torsional_natural_frequency():
    """Test f0 = c_s / (4 * L) in meters and feet."""
    # Case 1: L = 3170 m, c_s = 3170 m/s -> f0 = 3170 / (4 * 3170) = 0.25 Hz
    f0, t0, harmonics = drillstring_torsional_natural_frequency(
        drillstring_length=3170.0,
        shear_wave_velocity=3170.0,
        length_unit="m",
        harmonics_count=3,
    )
    assert pytest.approx(f0, rel=1e-4) == 0.25
    assert pytest.approx(t0, rel=1e-4) == 4.0
    assert len(harmonics) == 3
    # Harmonics: (2n-1)*f0 -> [0.25, 0.75, 1.25]
    assert pytest.approx(harmonics[0], rel=1e-4) == 0.25
    assert pytest.approx(harmonics[1], rel=1e-4) == 0.75
    assert pytest.approx(harmonics[2], rel=1e-4) == 1.25

    # Case 2: Length in feet
    # 10,000 ft = 3048 m -> f0 = 3170 / (4 * 3048) = 0.260 Hz
    f0_ft, _, _ = drillstring_torsional_natural_frequency(
        drillstring_length=10000.0,
        length_unit="ft",
    )
    expected_f0 = 3170.0 / (4.0 * (10000.0 * 0.3048))
    assert pytest.approx(f0_ft, rel=1e-4) == expected_f0


def test_fft_and_welch_spectral_analysis():
    """Test FFT and Welch PSD on a synthetic sinusoidal torsional signal."""
    sampling_rate = 10.0  # 10 Hz (standard MWD surface sampling)
    duration = 50.0       # 50 seconds -> 500 samples
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    target_freq = 0.25  # 0.25 Hz torsional oscillation
    signal = 5000.0 + 1500.0 * np.sin(2.0 * np.pi * target_freq * t)

    # FFT analysis
    fft_res = analyze_torsional_fft(signal, sampling_rate=sampling_rate, detrend=True)
    assert isinstance(fft_res, SpectralAnalysisResult)
    assert pytest.approx(fft_res.dominant_frequency, abs=0.05) == target_freq
    assert fft_res.peak_value > 0

    # Welch PSD analysis
    welch_res = analyze_torsional_welch(signal, sampling_rate=sampling_rate)
    assert isinstance(welch_res, SpectralAnalysisResult)
    assert pytest.approx(welch_res.dominant_frequency, abs=0.05) == target_freq


def test_check_resonance_risk():
    """Test resonance detection matching observed frequency with natural frequency."""
    drillstring_length = 3170.0  # m -> f0 = 0.25 Hz
    # Observed frequency near f0 (0.24 Hz is within 15% of 0.25 Hz)
    res_info = check_resonance_risk(
        observed_frequency=0.24,
        drillstring_length=drillstring_length,
        tolerance=0.15,
        length_unit="m",
    )
    assert res_info["is_resonance"] is True
    assert res_info["matched_mode"] == "fundamental"

    # Frequency far from resonance (e.g. 0.50 Hz)
    no_res_info = check_resonance_risk(
        observed_frequency=0.50,
        drillstring_length=drillstring_length,
        tolerance=0.10,
        length_unit="m",
    )
    assert no_res_info["is_resonance"] is False


# =====================================================================
# Bit Bounce Module Tests
# =====================================================================

def test_axial_vibration_index():
    """Test rolling Axial Vibration Index (AVI)."""
    # Smooth drilling: mean 20,000 lbf, tiny noise
    np.random.seed(42)
    wob_smooth = 20000.0 + np.random.normal(0, 500.0, 100)
    avi_smooth = compute_axial_vibration_index(wob_smooth, window_samples=20)
    valid_avi = avi_smooth[~np.isnan(avi_smooth)]
    assert np.all(valid_avi < 0.15)  # Very low vibration

    # Violent bouncing: oscillating between 0 and 40,000 lbf
    t = np.linspace(0, 10, 100)
    wob_bounce = 20000.0 + 20000.0 * np.sin(2.0 * np.pi * t)
    avi_bounce = compute_axial_vibration_index(wob_bounce, window_samples=20)
    valid_bounce = avi_bounce[~np.isnan(avi_bounce)]
    assert np.mean(valid_bounce) > 0.5  # High vibration


def test_classify_bit_bounce_severity():
    """Test classification of axial vibration index."""
    assert classify_bit_bounce_severity(0.10) == "Low"
    assert classify_bit_bounce_severity(0.35) == "Moderate"
    assert classify_bit_bounce_severity(0.75) == "High"
    assert classify_bit_bounce_severity(1.20) == "Severe"


def test_detect_axial_shocks():
    """Test shock detection for off-bottom bounce and impact spikes."""
    np.random.seed(42)
    # 200 samples of drilling data around 25000 lbf
    wob = np.full(200, 25000.0) + np.random.normal(0, 500, 200)

    # Inject off-bottom bounce (WOB collapses near 0)
    wob[50] = 500.0
    wob[51] = 0.0

    # Inject impact spike (WOB surges to 60,000 lbf)
    wob[120] = 60000.0

    df_shocks, summary = detect_axial_shocks(
        wob=wob,
        window_samples=30,
        shock_sigma_threshold=2.5,
        off_bottom_ratio=0.15,
    )

    assert isinstance(summary, AxialShockSummary)
    assert summary.bounce_events_count >= 1
    assert summary.shock_events_count >= 1
    assert df_shocks["is_bounce"].iloc[50] or df_shocks["is_bounce"].iloc[51]
    assert df_shocks["is_shock"].iloc[120]


# =====================================================================
# PDC Wear Model Tests
# =====================================================================

def test_cumulative_dissipated_energy():
    """Test discrete mechanical energy dissipation accumulation."""
    depths = np.array([5000.0, 5010.0, 5020.0, 5030.0])  # Delta z = 10 ft
    bit_dia = 8.5
    area = (np.pi / 4.0) * (8.5 ** 2)

    # Case 1: MSE equals UCS -> 0 dissipated energy
    mse_efficient = np.array([10000.0, 10000.0, 10000.0, 10000.0])
    ucs = 10000.0
    cum_e_eff = compute_cumulative_dissipated_energy(
        mse=mse_efficient, ucs=ucs, depth=depths, bit_diameter=bit_dia
    )
    assert np.all(cum_e_eff == 0.0)

    # Case 2: MSE exceeds UCS by 5000 psi
    # Delta E per step = 5000 psi * area * 10 ft
    excess = 5000.0
    step_energy = excess * area * 10.0
    mse_worn = np.array([15000.0, 15000.0, 15000.0, 15000.0])
    cum_e_worn = compute_cumulative_dissipated_energy(
        mse=mse_worn, ucs=ucs, depth=depths, bit_diameter=bit_dia
    )
    assert cum_e_worn[0] == 0.0
    assert pytest.approx(cum_e_worn[1], rel=1e-5) == step_energy
    assert pytest.approx(cum_e_worn[2], rel=1e-5) == 2.0 * step_energy
    assert pytest.approx(cum_e_worn[3], rel=1e-5) == 3.0 * step_energy


def test_estimate_cutter_wear_grade():
    """Test IADC 0 to 8 cutter wear grade estimation."""
    capacity = 1.0e8  # ft-lbf

    # Zero energy -> Grade 0
    w_cont, w_int = estimate_cutter_wear_grade(0.0, energy_capacity=capacity)
    assert w_cont == 0.0
    assert w_int == 0

    # 50% capacity -> Grade 4.0
    w_cont, w_int = estimate_cutter_wear_grade(0.5e8, energy_capacity=capacity)
    assert pytest.approx(w_cont, rel=1e-5) == 4.0
    assert w_int == 4

    # 100% capacity -> Grade 8.0
    w_cont, w_int = estimate_cutter_wear_grade(1.0e8, energy_capacity=capacity)
    assert pytest.approx(w_cont, rel=1e-5) == 8.0
    assert w_int == 8

    # Excess energy (> 100%) remains capped at Grade 8.0
    w_cont, w_int = estimate_cutter_wear_grade(2.0e8, energy_capacity=capacity)
    assert w_cont == 8.0
    assert w_int == 8


def test_evaluate_bit_wear_state():
    """Test full bit wear state evaluation and RUL estimation."""
    capacity = 1.0e8
    cum_e = 0.25e8  # 25% used -> grade 2.0
    drilled_ft = 500.0

    state = evaluate_bit_wear_state(
        cumulative_energy_ft_lbf=cum_e,
        energy_capacity_ft_lbf=capacity,
        drilled_depth_ft=drilled_ft,
    )

    assert isinstance(state, CutterWearState)
    assert pytest.approx(state.wear_grade_continuous, rel=1e-4) == 2.0
    assert state.iadc_wear_grade_integer == 2
    assert state.wear_condition == "Light"
    # Remaining capacity: 75% left -> remaining footage ~ 3 * 500 = 1500 ft
    assert pytest.approx(state.remaining_footage_estimate, rel=1e-4) == 1500.0
