"""
Unit tests for UnitConverter module.
"""

import numpy as np
import pytest
from src.data.unit_converter import UnitConverter


def test_length_conversions():
    # 100 ft -> meters
    m = UnitConverter.ft_to_m(100.0)
    assert pytest.approx(m, rel=1e-4) == 30.48
    # 30.48 m -> ft
    ft = UnitConverter.m_to_ft(30.48)
    assert pytest.approx(ft, rel=1e-4) == 100.0

    # 8.5 inches -> mm
    mm = UnitConverter.in_to_mm(8.5)
    assert pytest.approx(mm, rel=1e-4) == 215.9
    assert pytest.approx(UnitConverter.mm_to_in(215.9), rel=1e-4) == 8.5


def test_rop_conversions():
    rop_ft_hr = 50.0
    rop_m_hr = UnitConverter.ft_hr_to_m_hr(rop_ft_hr)
    assert pytest.approx(rop_m_hr, rel=1e-4) == 15.24
    assert pytest.approx(UnitConverter.m_hr_to_ft_hr(rop_m_hr), rel=1e-4) == 50.0


def test_wob_conversions():
    wob_klbf = 25.0
    wob_kn = UnitConverter.klbf_to_kn(wob_klbf)
    assert pytest.approx(wob_kn, rel=1e-3) == 111.205
    assert pytest.approx(UnitConverter.kn_to_klbf(wob_kn), rel=1e-3) == 25.0

    tonnes = UnitConverter.klbf_to_tonnes(wob_klbf)
    assert pytest.approx(tonnes, rel=1e-3) == 11.3398
    assert pytest.approx(UnitConverter.tonnes_to_klbf(tonnes), rel=1e-3) == 25.0


def test_torque_conversions():
    torque_ft_lbf = 15000.0
    torque_nm = UnitConverter.ft_lbf_to_nm(torque_ft_lbf)
    assert pytest.approx(torque_nm, rel=1e-3) == 20337.269
    assert pytest.approx(UnitConverter.nm_to_ft_lbf(torque_nm), rel=1e-3) == 15000.0


def test_pressure_conversions():
    psi = 10000.0
    bar = UnitConverter.psi_to_bar(psi)
    assert pytest.approx(bar, rel=1e-3) == 689.475
    assert pytest.approx(UnitConverter.bar_to_psi(bar), rel=1e-3) == 10000.0

    mpa = UnitConverter.psi_to_mpa(psi)
    assert pytest.approx(mpa, rel=1e-3) == 68.9475
    assert pytest.approx(UnitConverter.mpa_to_psi(mpa), rel=1e-3) == 10000.0


def test_flow_conversions():
    gpm = 600.0
    lpm = UnitConverter.gpm_to_lpm(gpm)
    assert pytest.approx(lpm, rel=1e-3) == 2271.247
    assert pytest.approx(UnitConverter.lpm_to_gpm(lpm), rel=1e-3) == 600.0


def test_numpy_array_support():
    arr = np.array([10.0, 20.0, 30.0])
    res = UnitConverter.ft_to_m(arr)
    assert len(res) == 3
    assert pytest.approx(res[0], rel=1e-4) == 3.048
