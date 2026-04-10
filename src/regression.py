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

