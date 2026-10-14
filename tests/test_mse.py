"""Tests for physics modules: MSE, D-exponent, and Hydraulics."""

import math
import numpy as np
import pandas as pd
import pytest

from src.physics.mse import (
    dupriest_mse,
    mechanical_efficiency,
    mpa_to_psi,
    over_energy_ratio,
    psi_to_mpa,
    teale_mse,
)
from src.physics.d_exponent import (
    detect_pore_pressure_anomaly,
    eaton_pore_pressure,
    fit_normal_compaction_trend,
    jordan_shirley_d_exponent,
    rehm_mcclendon_dc_exponent,
)
from src.physics.hydraulics import (
    annular_velocity,
    bit_pressure_drop,
    hydraulic_horsepower_per_sq_in,
    jet_impact_force,
    nozzle_tfa,
    nozzle_velocity,
)


# =====================================================================
# MSE Module Tests
# =====================================================================

def test_unit_conversion():
    """Test psi to MPa and MPa to psi conversion roundtrips."""
    psi_val = 14503.773773
    mpa_val = psi_to_mpa(psi_val)
    assert pytest.approx(mpa_val, rel=1e-4) == 100.0
    back_to_psi = mpa_to_psi(mpa_val)
    assert pytest.approx(back_to_psi, rel=1e-5) == psi_val


def test_teale_mse_scalar_field_units():
    """Test Teale MSE calculation with known analytical field values."""
    wob = 20000.0       # lbf
    torque = 3000.0     # ft-lbf
    rpm = 120.0         # RPM
    rop = 60.0          # ft/hr
    bit_dia = 8.5       # inches

    area = (np.pi / 4.0) * (8.5 ** 2)  # ~56.745 in^2
    axial_expected = wob / area        # ~352.45 psi
    rot_expected = (120.0 * np.pi * rpm * torque) / (area * rop)  # ~39861.59 psi
    expected_mse = axial_expected + rot_expected                  # ~40214.04 psi

    computed = teale_mse(wob=wob, torque=torque, rpm=rpm, rop=rop, bit_diameter=bit_dia)
    assert isinstance(computed, float)
    assert pytest.approx(computed, rel=1e-4) == expected_mse


def test_teale_mse_output_mpa():
    """Test Teale MSE directly returning MPa."""
    wob = 20000.0
    torque = 3000.0
    rpm = 120.0
    rop = 60.0
    bit_dia = 8.5

    mse_psi = teale_mse(wob, torque, rpm, rop, bit_diameter=bit_dia, output_unit="psi")
    mse_mpa = teale_mse(wob, torque, rpm, rop, bit_diameter=bit_dia, output_unit="MPa")

    assert pytest.approx(mse_mpa, rel=1e-5) == psi_to_mpa(mse_psi)


def test_teale_mse_vector_numpy_and_pandas():
    """Test Teale MSE with NumPy arrays and Pandas Series."""
    wobs = np.array([15000.0, 20000.0, 25000.0])
    torques = np.array([2500.0, 3000.0, 3500.0])
    rpms = np.array([100.0, 120.0, 140.0])
    rops = np.array([40.0, 60.0, 80.0])

    # Numpy array input
    res_np = teale_mse(wobs, torques, rpms, rops, bit_diameter=8.5)
    assert isinstance(res_np, np.ndarray)
    assert len(res_np) == 3

    # Check first element matches scalar
    scalar_0 = teale_mse(wobs[0], torques[0], rpms[0], rops[0], bit_diameter=8.5)
    assert pytest.approx(res_np[0], rel=1e-5) == scalar_0

    # Pandas Series input
    s_wob = pd.Series(wobs, index=["a", "b", "c"])
    res_pd = teale_mse(s_wob, torques, rpms, rops, bit_diameter=8.5)
    assert isinstance(res_pd, pd.Series)
    assert list(res_pd.index) == ["a", "b", "c"]
    assert pytest.approx(res_pd.iloc[0], rel=1e-5) == scalar_0


def test_teale_mse_zero_and_negative_division():
    """Ensure ROP <= 0 or area <= 0 produces np.nan without raising ZeroDivisionError."""
    # Scalar ROP = 0
    res_zero = teale_mse(wob=20000.0, torque=3000.0, rpm=120.0, rop=0.0, bit_diameter=8.5)
    assert math.isnan(res_zero)

    # Negative ROP
    res_neg = teale_mse(wob=20000.0, torque=3000.0, rpm=120.0, rop=-10.0, bit_diameter=8.5)
    assert math.isnan(res_neg)

    # Vector with zeros
    rops = np.array([50.0, 0.0, 60.0])
    res_vec = teale_mse(20000.0, 3000.0, 120.0, rops, bit_diameter=8.5)
    assert np.isfinite(res_vec[0])
    assert np.isnan(res_vec[1])
    assert np.isfinite(res_vec[2])


def test_dupriest_mse_calculation():
    """Test Dupriest MSE with default EFF_m (0.35) and custom efficiency."""
    teale_val = 40000.0
    dup_default = dupriest_mse(mse_teale=teale_val)
    assert pytest.approx(dup_default, rel=1e-5) == 0.35 * 40000.0

    dup_custom = dupriest_mse(mse_teale=teale_val, eff_m=0.40)
    assert pytest.approx(dup_custom, rel=1e-5) == 16000.0

    # Compute from raw parameters
    dup_from_params = dupriest_mse(
        wob=20000.0, torque=3000.0, rpm=120.0, rop=60.0, bit_diameter=8.5, eff_m=0.35
    )
    raw_teale = teale_mse(20000.0, 3000.0, 120.0, 60.0, bit_diameter=8.5)
    assert pytest.approx(dup_from_params, rel=1e-5) == 0.35 * raw_teale


def test_mechanical_efficiency_and_over_energy_ratio():
    """Test EFF_m = UCS / MSE and OER = MSE / UCS."""
    teale_val = 40000.0
    ucs = 10000.0

    eff = mechanical_efficiency(teale_val, ucs)
    oer = over_energy_ratio(teale_val, ucs)

    assert pytest.approx(eff, rel=1e-5) == 0.25
    assert pytest.approx(oer, rel=1e-5) == 4.0
    assert pytest.approx(eff * oer, rel=1e-5) == 1.0

    # Zero handling
    assert math.isnan(mechanical_efficiency(0.0, 10000.0))
    assert math.isnan(over_energy_ratio(40000.0, 0.0))


# =====================================================================
# D-exponent Module Tests
# =====================================================================

def test_jordan_shirley_d_exponent():
    """Test Jordan & Shirley D-exponent calculation."""
    rop = 60.0       # ft/hr
    rpm = 100.0      # RPM
    wob = 30000.0    # lbf
    db = 8.5         # inches

    # Formula: D = log10(ROP / (60*RPM)) / log10(12*WOB / (1e6 * Db))
    num = np.log10(60.0 / (60.0 * 100.0))                 # log10(0.01) = -2.0
    den = np.log10((12.0 * 30000.0) / (1.0e6 * 8.5))     # log10(360000 / 8500000)
    expected_d = num / den

    d_calc = jordan_shirley_d_exponent(rop=rop, rpm=rpm, wob=wob, bit_diameter=db)
    assert pytest.approx(d_calc, rel=1e-4) == expected_d
    assert 1.0 < d_calc < 2.0


def test_rehm_mcclendon_dc_exponent():
    """Test mud weight correction on D-exponent."""
    d_val = 1.6
    normal_mw = 9.0
    actual_mw = 12.0
    expected_dc = 1.6 * (9.0 / 12.0)

    dc_calc = rehm_mcclendon_dc_exponent(d_val, actual_mud_weight=actual_mw, normal_mud_weight=normal_mw)
    assert pytest.approx(dc_calc, rel=1e-5) == expected_dc


def test_pore_pressure_anomaly_detection():
    """Test detection of abnormal pore pressure trend deviations."""
    depths = np.linspace(4000, 10000, 61)
    # Normal trend: Dc increases with depth
    normal_dc = 0.00015 * depths + 0.6
    # Induce overpressure drop below 8000 ft
    actual_dc = normal_dc.copy()
    actual_dc[depths >= 8000] -= 0.35

    df_anom = detect_pore_pressure_anomaly(
        depth=depths,
        dc_exponent=actual_dc,
        threshold=0.15,
    )

    assert "is_overpressured" in df_anom.columns
    assert "delta_dc" in df_anom.columns
    # Check that points below 8000 ft are flagged as overpressured
    overpressured_depths = df_anom.loc[df_anom["is_overpressured"], "depth"]
    assert (overpressured_depths >= 8000).all()
    assert len(overpressured_depths) > 0


def test_eaton_pore_pressure():
    """Test Eaton's pore pressure gradient calculation from Dc."""
    dc = 1.2
    dc_norm = 1.6
    obg = 1.0        # psi/ft
    ppg_norm = 0.465 # psi/ft

    expected_ppg = obg - (obg - ppg_norm) * ((dc / dc_norm) ** 1.2)
    ppg = eaton_pore_pressure(dc_exponent=dc, dc_normal=dc_norm, overburden_gradient=obg, normal_pore_gradient=ppg_norm)
    assert pytest.approx(ppg, rel=1e-4) == expected_ppg
    assert ppg > ppg_norm  # Overpressured zone has higher PPG


# =====================================================================
# Hydraulics Module Tests
# =====================================================================

def test_nozzle_tfa():
    """Test nozzle Total Flow Area calculation."""
    # Three 12/32" nozzles
    nozzles = [12, 12, 12]
    expected_area = 3.0 * (np.pi / 4.0) * ((12.0 / 32.0) ** 2)
    tfa = nozzle_tfa(nozzles, in_32nds=True)
    assert pytest.approx(tfa, rel=1e-5) == expected_area

    # Invalid input
    with pytest.raises(ValueError):
        nozzle_tfa([])
    with pytest.raises(ValueError):
        nozzle_tfa([-12, 12])


def test_nozzle_velocity():
    """Test nozzle exit velocity."""
    q = 500.0    # gpm
    tfa = 0.44   # in^2
    # v_n = 0.32086 * Q / TFA
    expected_vn = (0.32086 * 500.0) / 0.44
    vn = nozzle_velocity(flow_rate=q, tfa=tfa)
    assert pytest.approx(vn, rel=1e-4) == expected_vn


def test_bit_pressure_drop_and_hsi():
    """Test bit nozzle pressure drop and Hydraulic Horsepower per square inch."""
    mw = 10.0   # ppg
    q = 500.0   # gpm
    tfa = 0.44  # in^2
    cd = 0.95
    db = 8.5    # inches

    dp = bit_pressure_drop(mud_weight=mw, flow_rate=q, tfa=tfa, cd=cd)
    expected_dp = (10.0 * (500.0 ** 2)) / (12032.0 * (0.95 ** 2) * (0.44 ** 2))
    assert pytest.approx(dp, rel=1e-4) == expected_dp

    # HSI = (dp * Q) / (1714 * A_b)
    area_bit = (np.pi / 4.0) * (8.5 ** 2)
    expected_hsi = (dp * q) / (1714.0 * area_bit)
    hsi = hydraulic_horsepower_per_sq_in(delta_p_bit=dp, flow_rate=q, bit_diameter=db)
    assert pytest.approx(hsi, rel=1e-4) == expected_hsi


def test_jet_impact_force_and_annular_velocity():
    """Test jet impact force and annular velocity."""
    mw = 10.0
    q = 500.0
    tfa = 0.44
    vn = nozzle_velocity(q, tfa)

    # Jet impact force
    fj = jet_impact_force(mud_weight=mw, flow_rate=q, nozzle_vel=vn)
    expected_fj = (10.0 * 500.0 * vn) / 1930.4
    assert pytest.approx(fj, rel=1e-4) == expected_fj

    # Annular velocity
    va_ft_min = annular_velocity(flow_rate=500.0, hole_diameter=8.5, pipe_outer_diameter=5.0, in_feet_per_minute=True)
    expected_va = (24.5 * 500.0) / (8.5 ** 2 - 5.0 ** 2)
    assert pytest.approx(va_ft_min, rel=1e-4) == expected_va

    va_ft_s = annular_velocity(flow_rate=500.0, hole_diameter=8.5, pipe_outer_diameter=5.0, in_feet_per_minute=False)
    assert pytest.approx(va_ft_s, rel=1e-4) == expected_va / 60.0
