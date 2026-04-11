import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_series(series: pd.Series, name: str) -> pd.Series:
    """
    Validate that an input is a non-empty pandas Series.
    """
    if series is None:
        raise ValueError(f"{name} must not be None")

    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")

    if series.empty:
        raise ValueError(f"{name} must not be empty")

    return series

def _safe_sharpe(daily_pnl_bp, annualization_factor=252):
    """Compute annualized Sharpe ratio from daily PnL in bp."""
    pnl = pd.Series(daily_pnl_bp).dropna()
    if pnl.empty:
        return np.nan

    vol = pnl.std(ddof=1)
    if pd.isna(vol) or vol == 0:
        return np.nan

    return float(np.sqrt(annualization_factor) * pnl.mean() / vol)


def _compute_drawdown_series(cumulative_pnl_bp):
    """Compute drawdown series from cumulative PnL in bp."""
    cumulative = pd.Series(cumulative_pnl_bp).copy()
    running_max = cumulative.cummax()
    return cumulative - running_max


def _compute_max_drawdown(cumulative_pnl_bp):
    """Compute maximum drawdown in basis points from cumulative PnL."""
    drawdown = _compute_drawdown_series(cumulative_pnl_bp).dropna()
    if drawdown.empty:
        return np.nan
    return float(drawdown.min())


def _build_trade_stats(trades):
    """Build summary stats for a subset of trades."""
    if trades.empty:
        return {
            "cumulative_pnl_bp": 0.0,
            "hit_ratio": np.nan,
            "number_of_trades": 0,
            "avg_holding_period": np.nan,
            "avg_pnl_bp": np.nan,
        }

    wins = (trades["pnl_bp"] > 0).sum()
    count = len(trades)

    return {
        "cumulative_pnl_bp": float(trades["pnl_bp"].sum()),
        "hit_ratio": float(wins / count),
        "number_of_trades": int(count),
        "avg_holding_period": float(trades["holding_period"].mean()),
        "avg_pnl_bp": float(trades["pnl_bp"].mean()),
    }


def generate_signal(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate mean-reversion positions from a z-score series.

    Signal convention
    -----------------
    - Positive z-score = cheapness
    - Negative z-score = richness

    Trading logic
    -------------
    - Enter long when zscore >= entry_threshold   (fade cheapness)
    - Enter short when zscore <= -entry_threshold (fade richness)
    - Exit long when zscore <= exit_threshold
    - Exit short when zscore >= -exit_threshold

    Parameters
    ----------
    zscore : pd.Series
        Residual z-score series.
    entry_threshold : float, default 2.0
        Absolute threshold to enter a position.
    exit_threshold : float, default 0.5
        Threshold used for exit.

    Returns
    -------
    pd.Series
        Position series in {-1, 0, +1}, where +1=long and -1=short.
    """
    _validate_series(zscore, "zscore")

    if entry_threshold <= 0:
        raise ValueError("entry_threshold must be > 0")

    if exit_threshold < 0:
        raise ValueError("exit_threshold must be >= 0")

    if exit_threshold >= entry_threshold:
        raise ValueError("exit_threshold must be smaller than entry_threshold")

    position = pd.Series(0, index=zscore.index, dtype=int)
    current_position = 0

    for i, value in enumerate(zscore):
        if pd.isna(value):
            position.iloc[i] = current_position
            continue

        if current_position == 0:
            if value >= entry_threshold:
                current_position = 1
            elif value <= -entry_threshold:
                current_position = -1

        elif current_position == 1:
            if value <= exit_threshold:
                current_position = 0

        elif current_position == -1:
            if value >= -exit_threshold:
                current_position = 0

        position.iloc[i] = current_position

    return position


def backtest_signal(signal, price):
    """
    Backtest a signal against a tradable price series.

    PnL is computed in basis points as:
    daily_pnl_bp = lagged_position * price.diff() * 100

    Parameters
    ----------
    signal : pd.Series
        Position signal in {-1, 0, +1}.
    price : pd.Series
        Tradable price/level series (e.g. rates in percent units).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        price, signal, position, price_change, daily_pnl_bp, cumulative_pnl_bp.
    """
    _validate_series(signal, "signal")
    _validate_series(price, "price")

    df = pd.concat(
        [price.rename("price"), signal.rename("signal")],
        axis=1,
        join="inner",
    ).sort_index()

    if df.empty:
        raise ValueError("signal and price have no overlapping index")

    df["signal"] = df["signal"].fillna(0).clip(-1, 1).astype(int)
    df["position"] = df["signal"]
    df["price_change"] = df["price"].diff()

    lagged_position = df["position"].shift(1).fillna(0)
    df["daily_pnl_bp"] = lagged_position * df["price_change"] * 100
    df["daily_pnl_bp"] = df["daily_pnl_bp"].fillna(0.0)
    df["cumulative_pnl_bp"] = df["daily_pnl_bp"].cumsum()

    return df


def extract_trade_log(backtest_df, zscore=None):
    """
    Extract completed trades from a backtest DataFrame.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Backtest output with at least 'price' and 'position' columns.
    zscore : pd.Series, optional
        Optional z-score series to annotate entry and exit z-scores.

    Returns
    -------
    pd.DataFrame
        Completed trade log with entry/exit details and trade PnL in bp.
    """
    if not isinstance(backtest_df, pd.DataFrame):
        raise TypeError("backtest_df must be a pandas DataFrame")

    required_columns = {"price", "position"}
    missing_columns = required_columns - set(backtest_df.columns)
    if missing_columns:
        raise ValueError(f"backtest_df is missing required columns: {sorted(missing_columns)}")

    if zscore is not None:
        _validate_series(zscore, "zscore")
        zscore = zscore.reindex(backtest_df.index)

    position = backtest_df["position"].fillna(0).astype(int)
    price = backtest_df["price"]

    trades = []
    open_trade = None

    for idx in backtest_df.index:
        pos = position.loc[idx]
        px = price.loc[idx]

        if pd.isna(px):
            continue

        if open_trade is None and pos != 0:
            open_trade = {
                "entry_date": idx,
                "entry_price": float(px),
                "side_sign": int(pos),
                "entry_zscore": np.nan if zscore is None else zscore.loc[idx],
            }
            continue

        if open_trade is not None and pos == 0:
            exit_price = float(px)
            side_sign = open_trade["side_sign"]
            pnl_bp = side_sign * (exit_price - open_trade["entry_price"]) * 100

            trades.append(
                {
                    "entry_date": open_trade["entry_date"],
                    "exit_date": idx,
                    "side": "long" if side_sign == 1 else "short",
                    "signal_type": "fade_cheapness" if side_sign == 1 else "fade_richness",
                    "entry_price": open_trade["entry_price"],
                    "exit_price": exit_price,
                    "holding_period": int(
                        backtest_df.index.get_loc(idx) - backtest_df.index.get_loc(open_trade["entry_date"])
                    ),
                    "pnl_bp": float(pnl_bp),
                    "entry_zscore": open_trade["entry_zscore"],
                    "exit_zscore": np.nan if zscore is None else zscore.loc[idx],
                }
            )
            open_trade = None

    base_columns = [
        "entry_date",
        "exit_date",
        "side",
        "signal_type",
        "entry_price",
        "exit_price",
        "holding_period",
        "pnl_bp",
    ]

    if zscore is not None:
        base_columns.extend(["entry_zscore", "exit_zscore"])

    if not trades:
        return pd.DataFrame(columns=base_columns)

    trade_log = pd.DataFrame(trades)

    if zscore is None:
        trade_log = trade_log[base_columns]
    else:
        trade_log = trade_log[base_columns]

    return trade_log


def compute_performance_metrics(backtest_df, trade_log):
    """
    Compute overall and side-specific backtest performance metrics.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Backtest DataFrame with daily and cumulative PnL columns.
    trade_log : pd.DataFrame
        Completed trade log output from extract_trade_log.

    Returns
    -------
    dict
        Nested dictionary with overall, long, and short statistics.
    """
    if not isinstance(backtest_df, pd.DataFrame):
        raise TypeError("backtest_df must be a pandas DataFrame")
    if not isinstance(trade_log, pd.DataFrame):
        raise TypeError("trade_log must be a pandas DataFrame")

    required_bt_cols = {"daily_pnl_bp", "cumulative_pnl_bp", "position"}
    missing_bt_cols = required_bt_cols - set(backtest_df.columns)
    if missing_bt_cols:
        raise ValueError(f"backtest_df is missing required columns: {sorted(missing_bt_cols)}")

    required_trade_cols = {"side", "holding_period", "pnl_bp"}
    missing_trade_cols = required_trade_cols - set(trade_log.columns)
    if missing_trade_cols:
        raise ValueError(f"trade_log is missing required columns: {sorted(missing_trade_cols)}")

    daily_pnl = backtest_df["daily_pnl_bp"].fillna(0.0)
    cumulative = backtest_df["cumulative_pnl_bp"].ffill().fillna(0.0)

    overall_trade_stats = _build_trade_stats(trade_log)
    overall = {
        "cumulative_pnl_bp": float(cumulative.iloc[-1]) if len(cumulative) else 0.0,
        "sharpe_ratio": _safe_sharpe(daily_pnl),
        "hit_ratio": overall_trade_stats["hit_ratio"],
        "max_drawdown_bp": _compute_max_drawdown(cumulative),
        "pnl_vol_bp": float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else np.nan,
        "number_of_trades": overall_trade_stats["number_of_trades"],
        "avg_holding_period": overall_trade_stats["avg_holding_period"],
    }

    long_trades = trade_log[trade_log["side"] == "long"]
    short_trades = trade_log[trade_log["side"] == "short"]

    long_daily_pnl = backtest_df["daily_pnl_bp"].where(backtest_df["position"].shift(1).fillna(0) > 0, 0.0)
    short_daily_pnl = backtest_df["daily_pnl_bp"].where(backtest_df["position"].shift(1).fillna(0) < 0, 0.0)

    long_trade_stats = _build_trade_stats(long_trades)
    short_trade_stats = _build_trade_stats(short_trades)

    long_metrics = {
        "cumulative_pnl_bp": float(long_daily_pnl.sum()),
        "sharpe_ratio": _safe_sharpe(long_daily_pnl),
        "hit_ratio": long_trade_stats["hit_ratio"],
        "max_drawdown_bp": _compute_max_drawdown(long_daily_pnl.cumsum()),
        "pnl_vol_bp": float(long_daily_pnl.std(ddof=1)) if len(long_daily_pnl) > 1 else np.nan,
        "number_of_trades": long_trade_stats["number_of_trades"],
        "avg_holding_period": long_trade_stats["avg_holding_period"],
    }

    short_metrics = {
        "cumulative_pnl_bp": float(short_daily_pnl.sum()),
        "sharpe_ratio": _safe_sharpe(short_daily_pnl),
        "hit_ratio": short_trade_stats["hit_ratio"],
        "max_drawdown_bp": _compute_max_drawdown(short_daily_pnl.cumsum()),
        "pnl_vol_bp": float(short_daily_pnl.std(ddof=1)) if len(short_daily_pnl) > 1 else np.nan,
        "number_of_trades": short_trade_stats["number_of_trades"],
        "avg_holding_period": short_trade_stats["avg_holding_period"],
    }

    return {
        "overall": overall,
        "long": long_metrics,
        "short": short_metrics,
    }


def compute_yearly_breakdown(backtest_df, trade_log):
    """
    Compute yearly performance using daily PnL and trades closed in each year.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Backtest DataFrame containing daily_pnl_bp with DatetimeIndex.
    trade_log : pd.DataFrame
        Trade log containing exit_date and pnl_bp.

    Returns
    -------
    pd.DataFrame
        Calendar-year performance summary.
    """
    if not isinstance(backtest_df, pd.DataFrame):
        raise TypeError("backtest_df must be a pandas DataFrame")
    if "daily_pnl_bp" not in backtest_df.columns:
        raise ValueError("backtest_df must contain 'daily_pnl_bp'")
    if not isinstance(backtest_df.index, pd.DatetimeIndex):
        raise TypeError("backtest_df index must be a pandas DatetimeIndex")
    if not isinstance(trade_log, pd.DataFrame):
        raise TypeError("trade_log must be a pandas DataFrame")

    pnl_by_year = backtest_df["daily_pnl_bp"].fillna(0.0).groupby(backtest_df.index.year)

    yearly_rows = []
    for year, pnl in pnl_by_year:
        year_stats = {
            "year": int(year),
            "cumulative_pnl_bp": float(pnl.sum()),
            "sharpe_ratio": _safe_sharpe(pnl),
            "hit_ratio": np.nan,
            "number_of_trades": 0,
        }

        if not trade_log.empty and "exit_date" in trade_log.columns:
            exits = pd.to_datetime(trade_log["exit_date"], errors="coerce")
            year_trades = trade_log[exits.dt.year == year]

            year_stats["number_of_trades"] = int(len(year_trades))
            if len(year_trades) > 0:
                year_stats["hit_ratio"] = float((year_trades["pnl_bp"] > 0).mean())

        yearly_rows.append(year_stats)

    if not yearly_rows:
        return pd.DataFrame(
            columns=["cumulative_pnl_bp", "sharpe_ratio", "hit_ratio", "number_of_trades"]
        )

    return pd.DataFrame(yearly_rows).set_index("year").sort_index()


def plot_backtest(backtest_df, title="Backtest"):
    """
    Plot price with entries/exits and cumulative PnL.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Backtest output from backtest_signal.
    title : str, default "Backtest"
        Plot title prefix.

    Returns
    -------
    tuple
        Matplotlib figure and axes.
    """
    if not isinstance(backtest_df, pd.DataFrame):
        raise TypeError("backtest_df must be a pandas DataFrame")

    required_columns = {"price", "position", "cumulative_pnl_bp"}
    missing_columns = required_columns - set(backtest_df.columns)
    if missing_columns:
        raise ValueError(f"backtest_df is missing required columns: {sorted(missing_columns)}")

    df = backtest_df.copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df["price"], label="Price")
    axes[0].set_ylabel("Price")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    long_entries = (df["position"] == 1) & (df["position"].shift(1).fillna(0) == 0)
    short_entries = (df["position"] == -1) & (df["position"].shift(1).fillna(0) == 0)
    exits = (df["position"] == 0) & (df["position"].shift(1).fillna(0) != 0)

    axes[0].scatter(df.index[long_entries], df.loc[long_entries, "price"], marker="^", s=50, label="Long Entry")
    axes[0].scatter(df.index[short_entries], df.loc[short_entries, "price"], marker="v", s=50, label="Short Entry")
    axes[0].scatter(df.index[exits], df.loc[exits, "price"], marker="o", s=40, label="Exit")
    axes[0].legend(loc="best")

    axes[1].plot(df.index, df["cumulative_pnl_bp"], label="Cumulative PnL (bp)")
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("PnL (bp)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.show()
    return fig, axes

def compute_drawdown_series(cumulative_pnl_bp: pd.Series) -> pd.Series:
    """
    Compute drawdown series from cumulative PnL in basis points.

    Parameters
    ----------
    cumulative_pnl_bp : pd.Series
        Cumulative PnL series in basis points.

    Returns
    -------
    pd.Series
        Drawdown series in basis points.
    """
    cumulative_pnl_bp = _validate_series(cumulative_pnl_bp, "cumulative_pnl_bp")

    running_max = cumulative_pnl_bp.cummax()
    drawdown = cumulative_pnl_bp - running_max
    drawdown.name = "drawdown_bp"

    return drawdown


def compute_rolling_sharpe(
    daily_pnl_bp: pd.Series,
    window: int = 60,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio from daily PnL in basis points.

    Parameters
    ----------
    daily_pnl_bp : pd.Series
        Daily PnL series in basis points.
    window : int, default 60
        Rolling window length.
    annualization_factor : int, default 252
        Annualization factor.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio series.
    """
    daily_pnl_bp = _validate_series(daily_pnl_bp, "daily_pnl_bp")

    if window <= 1:
        raise ValueError("window must be greater than 1")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    rolling_mean = daily_pnl_bp.rolling(window=window, min_periods=window).mean()
    rolling_std = daily_pnl_bp.rolling(window=window, min_periods=window).std()

    rolling_sharpe = rolling_mean / rolling_std
    rolling_sharpe = rolling_sharpe * np.sqrt(annualization_factor)
    rolling_sharpe = rolling_sharpe.where(rolling_std > 0)
    rolling_sharpe.name = f"rolling_sharpe_{window}d"

    return rolling_sharpe


def build_backtest_diagnostics(
    backtest_df: pd.DataFrame,
    sharpe_window: int = 60,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Add diagnostic time series to a backtest DataFrame.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of backtest_signal().
    sharpe_window : int, default 60
        Rolling Sharpe window length.
    annualization_factor : int, default 252
        Annualization factor for rolling Sharpe.

    Returns
    -------
    pd.DataFrame
        Backtest DataFrame with diagnostic columns added.
    """
    required_columns = {"daily_pnl_bp", "cumulative_pnl_bp"}
    missing = required_columns - set(backtest_df.columns)
    if missing:
        raise ValueError(f"backtest_df is missing required columns: {sorted(missing)}")

    df = backtest_df.copy()

    df["drawdown_bp"] = compute_drawdown_series(df["cumulative_pnl_bp"])
    df["rolling_sharpe"] = compute_rolling_sharpe(
        df["daily_pnl_bp"],
        window=sharpe_window,
        annualization_factor=annualization_factor,
    )

    return df


def compute_yearly_pnl_series(backtest_df: pd.DataFrame) -> pd.Series:
    """
    Compute yearly PnL in basis points from daily PnL.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of backtest_signal().

    Returns
    -------
    pd.Series
        Yearly PnL series in basis points indexed by calendar year.
    """
    if "daily_pnl_bp" not in backtest_df.columns:
        raise ValueError("backtest_df must contain 'daily_pnl_bp'")

    if not isinstance(backtest_df.index, pd.DatetimeIndex):
        raise ValueError("backtest_df index must be a DatetimeIndex")

    yearly_pnl = backtest_df["daily_pnl_bp"].groupby(backtest_df.index.year).sum()
    yearly_pnl.name = "yearly_pnl_bp"

    return yearly_pnl


def plot_pnl_and_risk(
    backtest_df: pd.DataFrame,
    sharpe_window: int = 60,
    annualization_factor: int = 252,
    title: str = "Backtest Analytics",
    figsize: tuple = (12, 10),
) -> None:
    """
    Plot cumulative PnL, drawdown, and rolling Sharpe.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of backtest_signal().
    sharpe_window : int, default 60
        Rolling Sharpe window length.
    annualization_factor : int, default 252
        Annualization factor for rolling Sharpe.
    title : str, default "Backtest Analytics"
        Plot title.
    figsize : tuple, default (12, 10)
        Figure size.
    """
    diagnostics_df = build_backtest_diagnostics(
        backtest_df,
        sharpe_window=sharpe_window,
        annualization_factor=annualization_factor,
    )

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(diagnostics_df.index, diagnostics_df["cumulative_pnl_bp"], label="Cumulative PnL")
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Cumulative PnL (bp)")
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(diagnostics_df.index, diagnostics_df["drawdown_bp"], label="Drawdown")
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Drawdown (bp)")
    axes[1].legend()

    axes[2].plot(diagnostics_df.index, diagnostics_df["rolling_sharpe"], label="Rolling Sharpe")
    axes[2].axhline(0, linestyle="--", linewidth=1)
    axes[2].set_ylabel(f"Rolling Sharpe ({sharpe_window}d)")
    axes[2].set_xlabel("Date")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_yearly_pnl(
    backtest_df: pd.DataFrame,
    title: str = "Yearly PnL",
    figsize: tuple = (10, 5),
) -> None:
    """
    Plot yearly PnL in basis points.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of backtest_signal().
    title : str, default "Yearly PnL"
        Plot title.
    figsize : tuple, default (10, 5)
        Figure size.
    """
    yearly_pnl = compute_yearly_pnl_series(backtest_df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(yearly_pnl.index.astype(str), yearly_pnl.values)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("PnL (bp)")
    ax.set_xlabel("Year")

    plt.tight_layout()
    plt.show()