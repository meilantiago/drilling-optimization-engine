"""
Tests for Drilling Telemetry Generator & Optimizer Modules
===========================================================
Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.generator import (
    DrillingDataGenerator,
    FormationLayer,
    generate_benchmark_files,
    DEFAULT_FORMATIONS,
)
from src.data.loader import DrillingDataLoader
from src.optimizer.founder_point import (
    FounderPointAnalyzer,
    FounderPointResult,
)
from src.optimizer.road_map import (
    RoadMapOptimizer,
    DrillingEnvelope,
    OptimizationResult,
)


class TestDrillingDataGenerator:
    """Test suite for high-fidelity drilling telemetry generator."""

    def test_profile_generation_channels_and_physics(self):
        generator = DrillingDataGenerator(random_seed=123)
        df = generator.generate_profile(
            start_depth_m=2000.0,
            end_depth_m=2100.0,
            depth_step_m=0.5,
            base_rpm=120.0,
            base_flow_gpm=550.0,
            add_founder_sweeps=True,
        )

        # Check required canonical columns
        expected_cols = [
            "depth", "rop", "wob", "rpm", "torque", "flow", "spp", "gamma",
            "formation", "ucs_psi", "founder_wob", "ssi", "mse_psi"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Column '{col}' missing from generated telemetry"

        # Check depth monotonic and correct range
        assert len(df) == 201
        assert df["depth"].iloc[0] == 2000.0
        assert df["depth"].iloc[-1] == 2100.0
        assert (np.diff(df["depth"]) > 0).all()

        # Check physical bounds (non-negative and realistic)
        assert (df["rop"] > 0).all()
        assert (df["wob"] > 0).all()
        assert (df["rpm"] >= 0).all()
        assert (df["torque"] > 0).all()
        assert (df["flow"] > 0).all()
        assert (df["spp"] > 0).all()
        assert (df["gamma"] >= 0).all()
        assert (df["mse_psi"] > 0).all()

    def test_stratified_formations_and_stick_slip(self):
        generator = DrillingDataGenerator(random_seed=42)
        df = generator.generate_profile(
            start_depth_m=2000.0,
            end_depth_m=2450.0,
            depth_step_m=1.0,
            add_founder_sweeps=True,
        )

        formations_present = set(df["formation"].unique())
        assert "Shale" in formations_present
        assert "Sandstone" in formations_present
        assert "Limestone" in formations_present
        assert "Dolomite" in formations_present

        # Verify rock strength hierarchy: Dolomite > Limestone > Sandstone > Shale
        dolomite_ucs = df[df["formation"] == "Dolomite"]["ucs_psi"].iloc[0]
        limestone_ucs = df[df["formation"] == "Limestone"]["ucs_psi"].iloc[0]
        sandstone_ucs = df[df["formation"] == "Sandstone"]["ucs_psi"].iloc[0]
        shale_ucs = df[df["formation"] == "Shale"]["ucs_psi"].iloc[0]

        assert dolomite_ucs > limestone_ucs > sandstone_ucs > shale_ucs

        # Verify stick-slip occurs in hard/frictional formations (Limestone/Dolomite)
        limestone_ssi = df[df["formation"] == "Limestone"]["ssi"]
        shale_ssi = df[df["formation"] == "Shale"]["ssi"]
        assert limestone_ssi.max() > shale_ssi.max()
        assert (limestone_ssi > 0.5).any(), "Stick-slip oscillation expected in Limestone"

    def test_export_and_loader_roundtrip(self, tmp_path):
        generator = DrillingDataGenerator(random_seed=42)
        df = generator.generate_profile(
            start_depth_m=2000.0,
            end_depth_m=2050.0,
            depth_step_m=0.5,
        )

        parquet_file = tmp_path / "test_telemetry.parquet"
        las_file = tmp_path / "test_telemetry.las"

        p_out, l_out = generator.export_dataset(df, parquet_path=parquet_file, las_path=las_file)
        assert p_out.exists()
        assert l_out.exists()

        # Verify DrillingDataLoader can load both formats cleanly
        df_pq = DrillingDataLoader.load_file(parquet_file)
        assert "depth" in df_pq.columns
        assert "rop" in df_pq.columns
        assert len(df_pq) == len(df)

        df_las = DrillingDataLoader.load_file(las_file)
        assert "depth" in df_las.columns
        assert "rop" in df_las.columns
        assert len(df_las) == len(df)

    def test_generate_benchmark_files(self, tmp_path):
        out_dir = tmp_path / "processed"
        pq_path, las_path = generate_benchmark_files(
            output_dir=out_dir,
            start_depth=2000.0,
            end_depth=2100.0,
        )
        assert pq_path.exists()
        assert las_path.exists()
        assert pq_path.stat().st_size > 0
        assert las_path.stat().st_size > 0


class TestFounderPointAnalyzer:
    """Test suite for automated founder point detection."""

    def test_controlled_founder_point_detection(self):
        # Create synthetic step test with known founder at WOB = 30 klbf
        wob_vals = np.linspace(10.0, 45.0, 60)
        # Up to 30 klbf, ROP rises linearly to 25 m/hr; beyond 30 klbf, ROP saturates and drops to 15 m/hr
        rop_vals = np.where(
            wob_vals <= 30.0,
            5.0 + (20.0 / 20.0) * (wob_vals - 10.0),
            25.0 - 0.7 * np.maximum(0.0, wob_vals - 30.0) ** 1.3
        )
        # Add slight noise
        rng = np.random.default_rng(42)
        rop_noisy = rop_vals + rng.normal(0, 0.3, len(rop_vals))

        df_test = pd.DataFrame({
            "depth": np.linspace(2100.0, 2130.0, len(wob_vals)),
            "wob": wob_vals,
            "rop": np.maximum(0.5, rop_noisy),
            "rpm": np.full(len(wob_vals), 120.0),
            "torque": 200.0 * wob_vals,
        })

        analyzer = FounderPointAnalyzer(safety_factor=0.90)
        result = analyzer.analyze(df_test)

        assert result.regime == "founder_detected"
        # Identified founder WOB should be in close proximity to 30.0 klbf (within 15%)
        assert 26.0 <= result.wob_founder <= 34.0, f"Detected WOB {result.wob_founder} outside expected range"
        # Recommended WOB should be 90% of founder WOB
        assert result.wob_recommended <= result.wob_founder
        assert np.isclose(result.wob_recommended, round(result.wob_founder * 0.90, 2))
        assert result.rop_at_founder > 0
        assert result.confidence_score > 0.75

    def test_pre_founder_linear_regime(self):
        # Data strictly within linear cutting regime (no saturation)
        wob_vals = np.linspace(10.0, 25.0, 40)
        rop_vals = 0.8 * wob_vals + 2.0

        df_linear = pd.DataFrame({
            "depth": np.linspace(2000.0, 2020.0, len(wob_vals)),
            "wob": wob_vals,
            "rop": rop_vals,
            "rpm": np.full(len(wob_vals), 110.0),
            "torque": 180.0 * wob_vals,
        })

        analyzer = FounderPointAnalyzer()
        result = analyzer.analyze(df_linear)
        assert result.regime in ["pre_founder", "founder_detected"]
        assert result.d_rop_d_wob > 0

    def test_insufficient_samples(self):
        df_small = pd.DataFrame({
            "depth": [2000.0, 2001.0],
            "wob": [15.0, 16.0],
            "rop": [12.0, 13.0],
            "rpm": [100.0, 100.0],
            "torque": [3000.0, 3200.0],
        })
        analyzer = FounderPointAnalyzer(min_samples=10)
        res = analyzer.analyze(df_small)
        assert res.regime == "insufficient_data"
        assert np.isnan(res.wob_founder)

    def test_analyze_by_formation(self):
        generator = DrillingDataGenerator(random_seed=42)
        df = generator.generate_profile(
            start_depth_m=2000.0,
            end_depth_m=2300.0,
            depth_step_m=0.5,
            add_founder_sweeps=True,
        )

        analyzer = FounderPointAnalyzer()
        results = analyzer.analyze_by_formation(df)

        assert "Shale" in results
        assert "Sandstone" in results
        # Sandstone has higher founder point than Shale
        assert results["Sandstone"].wob_founder >= results["Shale"].wob_founder


class TestRoadMapOptimizer:
    """Test suite for operational envelope optimizer and financial ROI."""

    def test_stick_slip_mitigation_and_ssi_target(self):
        # Telemetry with severe stick-slip (SSI ~ 0.85)
        n = 50
        df_ss = pd.DataFrame({
            "depth": np.linspace(2200.0, 2225.0, n),
            "wob": np.full(n, 28.0),
            "rop": np.full(n, 12.0),
            "rpm": np.full(n, 90.0),
            "torque": np.full(n, 8500.0),
            "flow": np.full(n, 520.0),
            "ssi": np.full(n, 0.85),
            "spp": np.full(n, 3100.0),
        })

        optimizer = RoadMapOptimizer()
        res = optimizer.optimize_interval(df_ss, target_ssi_limit=0.45)

        # Verified SSI mitigated below threshold
        assert res.initial_ssi == 0.85
        assert res.projected_ssi <= 0.45
        assert res.recommended_rpm > 90.0, "RPM must be increased to mitigate stick-slip"
        assert res.projected_rop >= res.baseline_rop
        assert res.hours_saved >= 0.0
        assert res.cost_savings_usd >= 0.0

    def test_financial_roi_calculation(self):
        # Interval of 100 meters, baseline ROP = 10 m/hr -> 10 hours
        # Founder point analysis allows pushing ROP to 20 m/hr -> 5 hours
        # Hours saved = 5.0 hours
        # Rig day rate = $60,000/day -> $2,500/hour
        # Expected savings = 5.0 * $2,500 = $12,500 USD
        df_interval = pd.DataFrame({
            "depth": np.linspace(2000.0, 2100.0, 80),
            "wob": np.full(80, 18.0),
            "rop": np.full(80, 10.0),
            "rpm": np.full(80, 110.0),
            "torque": np.full(80, 6000.0),
            "flow": np.full(80, 500.0),
            "ssi": np.full(80, 0.20),
        })

        founder_res = FounderPointResult(
            wob_founder=30.0,
            wob_recommended=27.0,
            rop_at_founder=22.0,
            rop_at_recommended=20.0,
            mse_at_founder=35000.0,
            confidence_score=0.92,
            regime="founder_detected",
            d_rop_d_wob=0.8,
            wob_tested_range=(15.0, 32.0),
            depth_interval=(2000.0, 2100.0),
        )

        optimizer = RoadMapOptimizer(default_day_rate_usd=60000.0)
        res = optimizer.optimize_interval(df_interval, founder_result=founder_res)

        assert res.drilling_depth_m == 100.0
        assert res.recommended_wob_klbf <= 30.0
        assert res.rop_gain_pct > 0.0
        assert res.hours_saved > 0.0

        expected_hourly_rate = 60000.0 / 24.0
        assert np.isclose(res.cost_savings_usd, res.hours_saved * expected_hourly_rate, atol=1.0)
        assert res.envelope.wob_min_klbf < res.envelope.wob_opt_klbf <= res.envelope.wob_max_klbf
        assert res.envelope.rpm_min < res.envelope.rpm_opt <= res.envelope.rpm_max

    def test_topdrive_torque_limiting(self):
        # If baseline torque is already high, recommended WOB must not exceed topdrive torque limit
        n = 40
        df_high_torque = pd.DataFrame({
            "depth": np.linspace(2300.0, 2320.0, n),
            "wob": np.full(n, 35.0),
            "rop": np.full(n, 15.0),
            "rpm": np.full(n, 120.0),
            "torque": np.full(n, 26000.0),  # near topdrive limit
            "flow": np.full(n, 580.0),
            "ssi": np.full(n, 0.30),
        })

        # Set low torque limit of 27,000 ft-lbf
        optimizer = RoadMapOptimizer(topdrive_max_torque_ft_lbf=27000.0)
        res = optimizer.optimize_interval(df_high_torque)

        assert res.estimated_torque_ft_lbf <= 27000.0
        assert res.recommended_wob_klbf <= 35.0, "WOB should be constrained by torque limit"
