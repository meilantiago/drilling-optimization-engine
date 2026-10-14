"""
Drilling Telemetry & Log Loader Module
======================================
Robust data ingestion for industry-standard drilling logs:
- LAS files (CWLS Log ASCII Standard v1.2 / v2.0) via `lasio`
- CSV and Parquet continuous high-frequency surface & downhole telemetry
- Mnemonic alias mapping and automatic schema harmonization

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import lasio
import numpy as np
import pandas as pd


# Canonical alias dictionary mapping industry curve mnemonics to standard names
MNEMONIC_MAP: Dict[str, List[str]] = {
    "depth": [
        "DEPTH", "DEPT", "MD", "MEASURED_DEPTH", "BIT_DEPTH", "HOLE_DEPTH", "TVD"
    ],
    "rop": [
        "ROP", "ROPA", "ROP_AVG", "RATE_OF_PENETRATION", "ROP5", "ROPM"
    ],
    "wob": [
        "WOB", "SWOB", "DWOB", "WEIGHT_ON_BIT", "WOB_AVG", "WOB_SURF"
    ],
    "rpm": [
        "RPM", "SRPM", "DRPM", "SURF_RPM", "BIT_RPM", "ROTARY_SPEED"
    ],
    "torque": [
        "TORQUE", "TORQ", "STOR", "SURF_TORQUE", "BIT_TORQUE", "TORQUE_AVG"
    ],
    "flow": [
        "FLOW", "FLOW_IN", "FLOWIN", "PUMP_FLOW", "FLOW_RATE", "Q_PUMP"
    ],
    "spp": [
        "SPP", "STANDPIPE_PRESSURE", "PUMP_PRESS", "PUMP_PRESSURE", "PRESS_SPP"
    ],
    "gamma": [
        "GR", "GAM", "GAMMA", "GR_CAL", "GAMMA_RAY", "GRD"
    ],
}


class DrillingDataLoader:
    """Loads, inspects, and harmonizes drilling telemetry data from LAS, CSV, and Parquet."""

    @classmethod
    def load_file(cls, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads a file automatically inspecting its extension (.las, .csv, .parquet)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".las":
            return cls.load_las(path)
        elif ext == ".csv":
            return cls.load_csv(path)
        elif ext in [".parquet", ".pq"]:
            return cls.load_parquet(path)
        else:
            raise ValueError(f"Unsupported drilling file format: {ext}. Expected .las, .csv, or .parquet")

    @classmethod
    def load_las(cls, file_path: Union[str, Path]) -> pd.DataFrame:
        """Parses a CWLS LAS file using lasio and standardizes curve mnemonics."""
        las = lasio.read(str(file_path))
        df = las.df().reset_index()
        return cls.harmonize_schema(df)

    @classmethod
    def load_csv(cls, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads a CSV telemetry file and standardizes column names."""
        df = pd.read_csv(file_path)
        return cls.harmonize_schema(df)

    @classmethod
    def load_parquet(cls, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads a Parquet telemetry file and standardizes column names."""
        df = pd.read_parquet(file_path)
        return cls.harmonize_schema(df)

    @classmethod
    def harmonize_schema(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps raw column mnemonics to canonical names:
        depth, rop, wob, rpm, torque, flow, spp, gamma
        """
        raw_cols = {col.upper().strip(): col for col in df.columns}
        rename_dict = {}

        for canonical_name, aliases in MNEMONIC_MAP.items():
            for alias in aliases:
                if alias in raw_cols:
                    rename_dict[raw_cols[alias]] = canonical_name
                    break

        df_harmonized = df.rename(columns=rename_dict).copy()

        # Sort by depth if depth column is available
        if "depth" in df_harmonized.columns:
            df_harmonized = df_harmonized.sort_values(by="depth").reset_index(drop=True)

        return df_harmonized

    @staticmethod
    def inspect_coverage(df: pd.DataFrame) -> Dict[str, Union[bool, float]]:
        """Returns diagnostic metadata about which primary drilling curves are present and their validity."""
        required = ["depth", "rop", "wob", "rpm", "torque"]
        report = {}
        for col in required:
            exists = col in df.columns
            valid_pct = (df[col].notna().mean() * 100.0) if exists else 0.0
            report[col] = {"present": exists, "valid_percentage": round(valid_pct, 2)}
        return report
