"""
End-to-End Integration Test for DrillOpt-Engine
===============================================
Validates the complete drilling mechanics and optimization pipeline:
1. Data generation & export (LAS & Parquet)
2. Ingestion & schema harmonization
3. Rig state classification
4. Physics engine: Teale MSE, Dupriest MSE, D-exponent, Hydraulics
5. Dynamics engine: Stick-Slip FFT, Bit Bounce shocks, PDC Wear
6. Optimization engine: Founder Point detection & Operating Envelope Road Map
7. Business ROI & Cost Savings calculation
"""

from pathlib import Path
import numpy as np
import pytest

from src.data.generator import DrillingDataGenerator
from src.data.loader import DrillingDataLoader
from src.data.rig_state import RigState, RigStateClassifier
from src.physics.mse import teale_mse, dupriest_mse, mechanical_efficiency
from src.physics.d_exponent import jordan_shirley_d_exponent, rehm_mcclendon_dc_exponent
from src.physics.hydraulics import nozzle_tfa, hydraulic_horsepower_per_sq_in
from src.dynamics.stick_slip import compute_stick_slip_index, analyze_torsional_fft, drillstring_torsional_natural_frequency
from src.dynamics.bit_bounce import compute_axial_vibration_index, detect_axial_shocks
from src.dynamics.wear_model import evaluate_bit_wear_state
from src.optimizer.founder_point import FounderPointAnalyzer
from src.optimizer.road_map import RoadMapOptimizer


def test_complete_drilling_pipeline_end_to_end(tmp_path: Path):
    # Step 1: Generate synthetic drilling telemetry
    generator = DrillingDataGenerator(bit_diameter_in=8.5, random_seed=42)
    df_raw = generator.generate_profile(start_depth_m=2000.0, end_depth_m=2150.0, depth_step_m=0.5)
    assert len(df_raw) > 200
    assert "depth" in df_raw.columns

    # Step 2: Export to LAS and Parquet and verify loader roundtrip
    las_path = tmp_path / "test_well.las"
    parquet_path = tmp_path / "test_well.parquet"

    generator.export_dataset(df_raw, parquet_path=parquet_path, las_path=las_path)

    assert las_path.exists()
    assert parquet_path.exists()

    df_loaded_las = DrillingDataLoader.load_file(las_path)
    df_loaded_parquet = DrillingDataLoader.load_file(parquet_path)

    assert len(df_loaded_las) == len(df_raw)
    assert len(df_loaded_parquet) == len(df_raw)

    # Step 3: Classify Rig States
    classifier = RigStateClassifier()
    df_classified = classifier.classify_dataframe(df_loaded_parquet)
    assert "rig_state" in df_classified.columns
    assert "is_on_bottom" in df_classified.columns
    active_drilling_count = (df_classified["rig_state"] == RigState.DRILLING_ROTARY.value).sum()
    assert active_drilling_count > 150

    # Step 4: Physics Engine Execution
    df_active = classifier.filter_active_drilling(df_classified)

    # MSE Calculations
    bit_diam_in = 8.5
    wob_klbf = df_active["wob"]
    rpm = df_active["rpm"]
    torque_ft_lbf = df_active["torque"]
    rop_m_hr = df_active["rop"]

    wob_lbf = wob_klbf * 1000.0
    rop_ft_hr = rop_m_hr / 0.3048

    mse_psi = teale_mse(wob=wob_lbf, torque=torque_ft_lbf, rpm=rpm, rop=rop_ft_hr, bit_diameter=bit_diam_in)
    assert np.all(np.isfinite(mse_psi))
    assert np.mean(mse_psi) > 5000.0  # Physically realistic MSE values

    dup_mse = dupriest_mse(mse_psi, eff_m=0.35)
    assert np.all(dup_mse < mse_psi)

    # D-exponent & Hydraulics
    d_exp = jordan_shirley_d_exponent(rop=rop_ft_hr, rpm=rpm, wob=wob_lbf, bit_diameter=bit_diam_in)
    dc_exp = rehm_mcclendon_dc_exponent(d_exponent=d_exp, actual_mud_weight=10.5, normal_mud_weight=9.0)
    assert np.all(np.isfinite(d_exp))
    assert np.all(np.isfinite(dc_exp))

    from src.physics.hydraulics import bit_pressure_drop
    tfa = nozzle_tfa([16, 16, 16])  # 3 x 16/32" nozzles (TFA ~ 0.589 sq in)
    dp_bit = bit_pressure_drop(mud_weight=10.5, flow_rate=550.0, tfa=tfa)
    hsi = hydraulic_horsepower_per_sq_in(delta_p_bit=dp_bit, flow_rate=550.0, bit_diameter=bit_diam_in)
    assert 1.0 <= hsi <= 10.0  # Industry standard HSI window

    # Step 5: Dynamics Engine Execution
    rpm_vals = df_active["rpm"]
    ssi = compute_stick_slip_index(rpm_max=rpm_vals.max(), rpm_min=rpm_vals.min(), rpm_avg=rpm_vals.mean())
    assert ssi >= 0.0

    fft_result = analyze_torsional_fft(signal=df_active["torque"].values, sampling_rate=1.0)
    f0, t0, harmonics = drillstring_torsional_natural_frequency(drillstring_length=2000.0)
    assert 0.1 <= f0 <= 1.0
    assert fft_result.dominant_frequency >= 0.0

    shock_df, shock_summary = detect_axial_shocks(df_active["wob"].values)
    assert shock_summary.total_samples == len(df_active)

    from src.dynamics.wear_model import compute_cumulative_dissipated_energy
    depth_ft = df_active["depth"] / 0.3048
    cum_energy = compute_cumulative_dissipated_energy(mse=mse_psi, ucs=8000.0, depth=depth_ft, bit_diameter=bit_diam_in)
    wear_eval = evaluate_bit_wear_state(cumulative_energy_ft_lbf=float(cum_energy.iloc[-1]))
    assert 0.0 <= wear_eval.wear_grade_continuous <= 8.0

    # Step 6: Optimizer Engine - Founder Point Detection
    analyzer = FounderPointAnalyzer(bit_diameter_in=bit_diam_in)
    fp_result = analyzer.analyze(df_active)
    assert fp_result.regime in ["founder_detected", "pre_founder", "post_founder"]
    assert fp_result.wob_recommended > 0.0

    # Step 7: Operational Roadmap & Business ROI
    optimizer = RoadMapOptimizer(founder_analyzer=analyzer, default_day_rate_usd=60000.0)
    opt_result = optimizer.optimize_section(df_active, section_length_m=150.0)

    assert opt_result.recommended_wob_klbf > 0.0
    assert opt_result.projected_rop_m_hr >= opt_result.baseline_rop_m_hr
    assert opt_result.hours_saved >= 0.0
    assert opt_result.cost_savings_usd >= 0.0
