import pandas as pd
import numpy as np
import statsmodels.api as sm


def run_ols(
    y: pd.Series,
    X: pd.DataFrame,
    add_constant: bool = True
):
    """
    Run OLS regression.
    """
    if add_constant:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model


def build_regression_output(
    y: pd.Series,
    X: pd.DataFrame
) -> dict:
    """
    Full regression engine:
    - OLS
    - fair value
    - residual
    - z-score
    """

    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    y_aligned = df["y"]
    X_aligned = df.drop(columns="y")

    model = run_ols(y_aligned, X_aligned)

    params = model.params
    tstats = model.tvalues
    pvalues = model.pvalues
    r2 = model.rsquared

    fitted = model.fittedvalues
    residual = model.resid

    zscore = (residual - residual.mean()) / residual.std()

    latest = {
        "y": y_aligned.iloc[-1],
        "fair_value": fitted.iloc[-1],
        "residual": residual.iloc[-1],
        "zscore": zscore.iloc[-1]
    }

    table = pd.DataFrame({
        "coef": params,
        "tstat": tstats,
        "pvalue": pvalues
    })

    return {
        "model": model,
        "table": table,
        "r2": r2,
        "fitted": fitted,
        "residual": residual,
        "zscore": zscore,
        "latest": latest
    }


def summarize_latest(results: dict) -> pd.Series:
    """
    Clean latest output for trading view.
    """
    latest = results["latest"]

    summary = pd.Series({
        "y": latest["y"],
        "fair_value": latest["fair_value"],
        "residual": latest["residual"],
        "zscore": latest["zscore"],
        "r2": results["r2"]
    })

    return summary


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def rolling_regression(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = 20
) -> dict:
    """
    Run rolling OLS and return window-end snapshots.

    Notes
    -----
    `rolling_fitted_values` and `rolling_residuals` store values only at the
    last observation of each rolling window (other rows remain NaN).
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    df = pd.concat([y.rename("y"), X], axis=1).dropna()

    if window > len(df):
        raise ValueError("window cannot be larger than the number of valid observations")

    y_aligned = df["y"]
    X_aligned = df.drop(columns="y")

    index = y_aligned.index
    rolling_betas = pd.DataFrame(index=index, columns=["const", *X_aligned.columns], dtype=float)
    rolling_r2 = pd.Series(np.nan, index=index, dtype=float)
    rolling_fitted_values = pd.Series(np.nan, index=index, dtype=float)
    rolling_residuals = pd.Series(np.nan, index=index, dtype=float)

    for end in range(window - 1, len(df)):
        window_slice = slice(end - window + 1, end + 1)
        model = run_ols(y_aligned.iloc[window_slice], X_aligned.iloc[window_slice])

        idx = index[end]
        rolling_betas.loc[idx, model.params.index] = model.params.values
        rolling_r2.loc[idx] = model.rsquared
        rolling_fitted_values.loc[idx] = model.fittedvalues.iloc[-1]
        rolling_residuals.loc[idx] = model.resid.iloc[-1]

    return {
        "rolling_coefficients": rolling_betas,
        "rolling_r2": rolling_r2,
        "rolling_fitted_values": rolling_fitted_values,
        "rolling_residuals": rolling_residuals,
    }