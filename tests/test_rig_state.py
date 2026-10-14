"""
Unit tests for RigStateClassifier.
"""

import pandas as pd
import pytest
from src.data.rig_state import RigState, RigStateClassifier


def test_classify_row():
    classifier = RigStateClassifier(
        min_wob_klbf=2.0, min_rpm=15.0, min_flow_gpm=100.0, min_rop_m_hr=0.5
    )

    # Active Rotary Drilling: WOB=20, RPM=120, Flow=500, ROP=15
    state = classifier.classify_row(wob=20.0, rpm=120.0, flow=500.0, rop=15.0)
    assert state == RigState.DRILLING_ROTARY

    # Sliding (Mud motor drilling without surface rotation): WOB=20, RPM=0, Flow=500, ROP=10
    state = classifier.classify_row(wob=20.0, rpm=0.0, flow=500.0, rop=10.0)
    assert state == RigState.DRILLING_SLIDING

    # Reaming: WOB=0, RPM=100, Flow=400, ROP=0
    state = classifier.classify_row(wob=0.0, rpm=100.0, flow=400.0, rop=0.0)
    assert state == RigState.REAMING

    # Circulating: WOB=0, RPM=0, Flow=400, ROP=0
    state = classifier.classify_row(wob=0.0, rpm=0.0, flow=400.0, rop=0.0)
    assert state == RigState.CIRCULATING

    # Connection: WOB=0, RPM=0, Flow=0, ROP=0
    state = classifier.classify_row(wob=0.0, rpm=0.0, flow=0.0, rop=0.0)
    assert state == RigState.CONNECTION


def test_classify_dataframe():
    classifier = RigStateClassifier()
    df = pd.DataFrame({
        "depth": [1000.0, 1000.5, 1001.0, 1001.0],
        "wob": [25.0, 0.0, 0.0, 0.0],
        "rpm": [120.0, 80.0, 0.0, 0.0],
        "flow": [550.0, 450.0, 450.0, 0.0],
        "rop": [20.0, 0.0, 0.0, 0.0],
    })

    df_classified = classifier.classify_dataframe(df)

    assert "rig_state" in df_classified.columns
    assert "is_on_bottom" in df_classified.columns
    assert df_classified["rig_state"].iloc[0] == RigState.DRILLING_ROTARY.value
    assert df_classified["rig_state"].iloc[1] == RigState.REAMING.value
    assert df_classified["rig_state"].iloc[2] == RigState.CIRCULATING.value
    assert df_classified["rig_state"].iloc[3] == RigState.CONNECTION.value

    assert bool(df_classified["is_on_bottom"].iloc[0]) is True
    assert bool(df_classified["is_on_bottom"].iloc[1]) is False

    df_filtered = classifier.filter_active_drilling(df_classified)
    assert len(df_filtered) == 1
    assert df_filtered["depth"].iloc[0] == 1000.0
