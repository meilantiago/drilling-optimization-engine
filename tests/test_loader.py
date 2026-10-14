"""
Unit tests for DrillingDataLoader module.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.data.loader import DrillingDataLoader, MNEMONIC_MAP


def test_mnemonic_map_coverage():
    assert "depth" in MNEMONIC_MAP
    assert "rop" in MNEMONIC_MAP
    assert "wob" in MNEMONIC_MAP
    assert "rpm" in MNEMONIC_MAP
    assert "torque" in MNEMONIC_MAP


def test_harmonize_schema():
    raw_df = pd.DataFrame({
        "DEPT": [1500.0, 1500.2, 1500.4],
        "ROPA": [25.0, 26.5, 24.8],
        "SWOB": [22.0, 22.5, 23.0],
        "SRPM": [110.0, 110.0, 110.0],
        "STOR": [12000.0, 12500.0, 12200.0],
        "FLOWIN": [600.0, 600.0, 600.0],
        "STANDPIPE_PRESSURE": [3200.0, 3250.0, 3220.0],
        "GR": [65.0, 70.0, 68.0],
    })

    harmonized = DrillingDataLoader.harmonize_schema(raw_df)

    expected_cols = ["depth", "rop", "wob", "rpm", "torque", "flow", "spp", "gamma"]
    for col in expected_cols:
        assert col in harmonized.columns, f"Expected {col} to be harmonized"

    coverage = DrillingDataLoader.inspect_coverage(harmonized)
    for col in ["depth", "rop", "wob", "rpm", "torque"]:
        assert coverage[col]["present"] is True
        assert coverage[col]["valid_percentage"] == 100.0


def test_csv_roundtrip(tmp_path: Path):
    file_path = tmp_path / "test_telemetry.csv"
    sample_df = pd.DataFrame({
        "DEPTH": [2000.0, 2001.0],
        "ROP": [15.0, 16.0],
        "WOB": [20.0, 22.0],
        "RPM": [120.0, 120.0],
        "TORQUE": [10000.0, 10500.0],
    })
    sample_df.to_csv(file_path, index=False)

    loaded = DrillingDataLoader.load_file(file_path)
    assert "depth" in loaded.columns
    assert len(loaded) == 2
    assert loaded["wob"].iloc[1] == 22.0
