import pandas as pd
import numpy as np

from src.backtest import (
    generate_signal,
    backtest_signal,
    extract_trade_log,
    compute_performance_metrics,
    compute_yearly_breakdown,
    build_backtest_diagnostics,
    compute_drawdown_series,
    compute_rolling_sharpe,
    compute_yearly_pnl_series,
    plot_backtest,
    plot_pnl_and_risk,
    plot_yearly_pnl,
)

# -----------------------------
# 1. Build toy data
# -----------------------------
dates = pd.date_range("2020-01-01", periods=300, freq="B")

# Toy tradable series in percent units
price = pd.Series(
    2.00
    + np.cumsum(np.random.normal(0, 0.01, len(dates))),
    index=dates,
    name="price",
)

# Toy z-score with some larger swings to trigger trades
zscore = pd.Series(
    1.8 * np.sin(np.linspace(0, 12 * np.pi, len(dates)))
    + np.random.normal(0, 0.35, len(dates)),
    index=dates,
    name="zscore",
)

# -----------------------------
# 2. Generate signal and run backtest
# -----------------------------
signal = generate_signal(zscore, entry_threshold=1.5, exit_threshold=0.5)
bt = backtest_signal(signal, price)

trade_log = extract_trade_log(bt, zscore=zscore)
metrics = compute_performance_metrics(bt, trade_log)
yearly = compute_yearly_breakdown(bt, trade_log)

# -----------------------------
# 3. Run diagnostics
# -----------------------------
diag = build_backtest_diagnostics(bt, sharpe_window=60)
drawdown = compute_drawdown_series(bt["cumulative_pnl_bp"])
rolling_sharpe = compute_rolling_sharpe(bt["daily_pnl_bp"], window=60)
yearly_pnl = compute_yearly_pnl_series(bt)

# -----------------------------
# 4. Print checks
# -----------------------------
print("=== BACKTEST HEAD ===")
print(bt.head(), "\n")

print("=== DIAGNOSTICS HEAD ===")
print(diag[["daily_pnl_bp", "cumulative_pnl_bp", "drawdown_bp", "rolling_sharpe"]].head(10), "\n")

print("=== TRADE LOG HEAD ===")
print(trade_log.head(), "\n")

print("=== METRICS ===")
print(metrics, "\n")

print("=== YEARLY BREAKDOWN ===")
print(yearly, "\n")

print("=== YEARLY PNL SERIES ===")
print(yearly_pnl, "\n")

# -----------------------------
# 5. Basic assertions
# -----------------------------
assert "drawdown_bp" in diag.columns
assert "rolling_sharpe" in diag.columns
assert (diag["drawdown_bp"].dropna() <= 0).all()
assert yearly_pnl.index.nunique() >= 1

print("All basic checks passed.")

# -----------------------------
# 6. Plots
# -----------------------------
plot_backtest(bt, title="Execution View")
plot_pnl_and_risk(bt, sharpe_window=60, title="Risk Diagnostics")
plot_yearly_pnl(bt, title="Yearly Strategy PnL")

signal_diag = build_signal_diagnostics_dataset(bt, zscore)

bucket_summary = compute_signal_bucket_summary(
    signal_diag,
    bins=[0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
    active_only=True,
)

bucket_summary_by_side = compute_signal_bucket_summary_by_side(
    signal_diag,
    bins=[0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
    active_only=True,
)

print(bucket_summary)
print(bucket_summary_by_side)

plot_signal_bucket_summary(
    bucket_summary,
    column="avg_pnl_bp",
    title="Average Daily PnL by Absolute Z-Score Bucket",
)