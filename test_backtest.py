import numpy as np
import pandas as pd

from src.backtest import (
    backtest_signal,
    compute_performance_metrics,
    compute_yearly_breakdown,
    extract_trade_log,
    generate_signal,
)


def test_generate_signal_basic():
    z = pd.Series([0.0, 2.1, 1.5, 0.4, -2.2, -1.0, -0.4])
    signal = generate_signal(z, entry_threshold=2.0, exit_threshold=0.5)

    expected = pd.Series([0, 1, 1, 0, -1, -1, 0])
    assert signal.equals(expected)


def test_backtest_signal_basic():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    price = pd.Series([2.00, 2.02, 2.05, 2.03, 2.00], index=idx)
    signal = pd.Series([0, 1, 1, 0, 0], index=idx)

    bt = backtest_signal(signal, price)

    expected_daily_pnl = pd.Series([0.0, 0.0, 3.0, -2.0, 0.0], index=idx)
    expected_cum_pnl = expected_daily_pnl.cumsum()

    assert np.allclose(bt["daily_pnl_bp"].values, expected_daily_pnl.values)
    assert np.allclose(bt["cumulative_pnl_bp"].values, expected_cum_pnl.values)


def test_extract_trade_log_long_and_short():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    price = pd.Series([2.00, 2.02, 2.05, 2.03, 2.01, 1.99, 1.96, 1.98], index=idx)
    signal = pd.Series([0, 1, 1, 0, 0, -1, -1, 0], index=idx)
    zscore = pd.Series([0.0, 2.1, 1.2, 0.4, 0.0, -2.2, -1.5, -0.3], index=idx)

    bt = backtest_signal(signal, price)
    trades = extract_trade_log(bt, zscore=zscore)

    assert len(trades) == 2

    assert trades.loc[0, "side"] == "long"
    assert trades.loc[0, "signal_type"] == "fade_cheapness"
    assert np.isclose(trades.loc[0, "pnl_bp"], 1.0)

    assert trades.loc[1, "side"] == "short"
    assert trades.loc[1, "signal_type"] == "fade_richness"
    assert np.isclose(trades.loc[1, "pnl_bp"], 1.0)


def test_compute_performance_metrics():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    price = pd.Series([2.00, 2.02, 2.05, 2.03, 2.01, 1.99, 1.96, 1.98], index=idx)
    signal = pd.Series([0, 1, 1, 0, 0, -1, -1, 0], index=idx)

    bt = backtest_signal(signal, price)
    trades = extract_trade_log(bt)
    metrics = compute_performance_metrics(bt, trades)

    assert "overall" in metrics
    assert "long" in metrics
    assert "short" in metrics

    assert metrics["overall"]["number_of_trades"] == 2
    assert np.isclose(metrics["overall"]["cumulative_pnl_bp"], bt["cumulative_pnl_bp"].iloc[-1])


def test_compute_yearly_breakdown():
    idx = pd.to_datetime(
        ["2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02", "2025-01-03"]
    )
    price = pd.Series([2.00, 2.02, 2.05, 2.03, 2.00], index=idx)
    signal = pd.Series([0, 1, 1, 0, 0], index=idx)

    bt = backtest_signal(signal, price)
    trades = extract_trade_log(bt)
    yearly = compute_yearly_breakdown(bt, trades)

    assert 2024 in yearly.index or 2025 in yearly.index
    assert "cumulative_pnl_bp" in yearly.columns
    assert "sharpe_ratio" in yearly.columns
    assert "hit_ratio" in yearly.columns
    assert "number_of_trades" in yearly.columns