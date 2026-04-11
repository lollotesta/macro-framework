import numpy as np
import pandas as pd

from src.backtest import (
    build_signal_diagnostics_dataset,
    compute_signal_bucket_summary,
    compute_signal_bucket_summary_by_side,
)


def test_build_signal_diagnostics_dataset_adds_expected_columns():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")

    backtest_df = pd.DataFrame(
        {
            "signal": [0, 1, 1, -1, 0],
            "position": [0, 1, 1, -1, 0],
            "daily_pnl_bp": [0.0, 1.0, -2.0, 3.0, 0.0],
        },
        index=idx,
    )

    zscore = pd.Series([0.2, 1.6, 2.1, -2.4, 0.3], index=idx, name="zscore")

    result = build_signal_diagnostics_dataset(backtest_df, zscore)

    expected_columns = {
        "signal",
        "position",
        "daily_pnl_bp",
        "zscore",
        "abs_zscore",
        "signal_side",
        "position_side",
        "signal_type",
        "is_active",
        "pnl_positive",
    }

    assert expected_columns.issubset(result.columns)
    assert result.loc[idx[1], "signal_side"] == "long"
    assert result.loc[idx[3], "signal_side"] == "short"
    assert result.loc[idx[0], "signal_side"] == "flat"
    assert result.loc[idx[1], "signal_type"] == "fade_cheapness"
    assert result.loc[idx[3], "signal_type"] == "fade_richness"
    assert result.loc[idx[0], "is_active"] == False
    assert result.loc[idx[1], "is_active"] == True


def test_compute_signal_bucket_summary_returns_expected_columns():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")

    diagnostics_df = pd.DataFrame(
        {
            "abs_zscore": [0.3, 0.8, 1.2, 1.8, 2.4, 3.2],
            "daily_pnl_bp": [1.0, -1.0, 2.0, 3.0, -2.0, 4.0],
            "is_active": [True, True, True, True, True, True],
        },
        index=idx,
    )

    result = compute_signal_bucket_summary(
        diagnostics_df,
        bins=[0, 1, 2, 3, np.inf],
        active_only=True,
    )

    expected_columns = {
        "count",
        "avg_pnl_bp",
        "median_pnl_bp",
        "total_pnl_bp",
        "pnl_vol_bp",
        "hit_ratio",
    }

    assert expected_columns.issubset(result.columns)
    assert result["count"].sum() == 6


def test_compute_signal_bucket_summary_active_only_filters_rows():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")

    diagnostics_df = pd.DataFrame(
        {
            "abs_zscore": [0.4, 1.1, 2.2, 2.8],
            "daily_pnl_bp": [1.0, 2.0, 3.0, 4.0],
            "is_active": [True, False, True, False],
        },
        index=idx,
    )

    result_active_only = compute_signal_bucket_summary(
        diagnostics_df,
        bins=[0, 1, 2, 3],
        active_only=True,
    )

    result_all_rows = compute_signal_bucket_summary(
        diagnostics_df,
        bins=[0, 1, 2, 3],
        active_only=False,
    )

    assert result_active_only["count"].sum() == 2
    assert result_all_rows["count"].sum() == 4


def test_compute_signal_bucket_summary_by_side_has_long_and_short_groups():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")

    diagnostics_df = pd.DataFrame(
        {
            "abs_zscore": [0.6, 1.3, 2.1, 0.7, 1.6, 2.4],
            "daily_pnl_bp": [1.0, 2.0, -1.0, -2.0, 3.0, 4.0],
            "position_side": ["long", "long", "long", "short", "short", "short"],
            "is_active": [True, True, True, True, True, True],
        },
        index=idx,
    )

    result = compute_signal_bucket_summary_by_side(
        diagnostics_df,
        bins=[0, 1, 2, 3],
        active_only=True,
    )

    assert "count" in result.columns
    assert "avg_pnl_bp" in result.columns
    assert "long" in result.index.get_level_values(0)
    assert "short" in result.index.get_level_values(0)
    assert result["count"].sum() == 6


def test_compute_signal_bucket_summary_by_side_excludes_flat_rows():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")

    diagnostics_df = pd.DataFrame(
        {
            "abs_zscore": [0.6, 1.3, 2.1, 1.1, 2.2],
            "daily_pnl_bp": [1.0, 2.0, -1.0, 5.0, 6.0],
            "position_side": ["long", "short", "flat", "long", "flat"],
            "is_active": [True, True, False, True, False],
        },
        index=idx,
    )

    result = compute_signal_bucket_summary_by_side(
        diagnostics_df,
        bins=[0, 1, 2, 3],
        active_only=False,
    )

    sides = set(result.index.get_level_values(0))
    assert "flat" not in sides
    assert result["count"].sum() == 3