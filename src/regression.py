import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def run_ols(y, X):
    """Run an OLS regression with an intercept term."""
    X_with_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_with_const, missing="drop")
    return model.fit()


def build_regression_output(model):
    return {
        "coefficients": model.params,
        "fitted_values": model.fittedvalues,
        "residuals": model.resid,
        "r_squared": model.rsquared,
    }


def summarize_latest(output):
    """Summarize coefficient values from the latest regression output."""
    coefficients = output["coefficients"]

    if isinstance(coefficients, pd.DataFrame):
        return coefficients.iloc[-1]

    return coefficients


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def compute_fair_value_deviation(actual: pd.Series, fair_value: pd.Series) -> pd.Series:
    """Compute deviation between observed value and fair value."""
    return actual - fair_value


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


def plot_fitted_vs_observed(
    observed: pd.Series,
    fitted: pd.Series,
    title: str = "Observed vs Fitted"
):
    """Plot observed series against fitted/fair value series."""
    frame = pd.concat(
        [observed.rename("observed"), fitted.rename("fitted")],
        axis=1
    ).dropna()

    plt.figure(figsize=(12, 6))
    plt.plot(frame.index, frame["observed"], label="Observed")
    plt.plot(frame.index, frame["fitted"], label="Fitted / Fair Value")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_zscore(
    zscore: pd.Series,
    title: str = "Z-Score Over Time",
    upper: float = 2.0,
    lower: float = -2.0
):
    """Plot z-score through time with threshold bands."""
    z = zscore.dropna()

    plt.figure(figsize=(12, 4))
    plt.plot(z.index, z.values, label="Z-Score")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axhline(upper, linestyle="--", linewidth=1)
    plt.axhline(lower, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_actual_fair_value_and_zscore(
    actual: pd.Series,
    fair_value: pd.Series,
    zscore: pd.Series,
    title: str = "Fair Value & Z-Score",
    upper: float = 2.0,
    lower: float = -2.0
):
    """Plot observed vs fair value and z-score in two panels."""

    import matplotlib.pyplot as plt

    df = pd.concat(
        [
            actual.rename("actual"),
            fair_value.rename("fair_value"),
            zscore.rename("zscore"),
        ],
        axis=1
    ).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Top panel: actual vs fair value ---
    axes[0].plot(df.index, df["actual"], label="Actual")
    axes[0].plot(df.index, df["fair_value"], label="Fair Value")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True)

    # --- Bottom panel: z-score ---
    axes[1].plot(df.index, df["zscore"], label="Z-Score")
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].axhline(upper, linestyle="--", linewidth=1)
    axes[1].axhline(lower, linestyle="--", linewidth=1)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def select_best_rolling_window(
    y: pd.Series,
    X: pd.DataFrame,
    windows: list[int],
    criterion: str = "ssr"
) -> dict:
    """
    Test multiple rolling regression windows and select the best one.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    windows : list[int]
        Candidate rolling window sizes.
    criterion : str, default "ssr"
        Selection criterion: "ssr" or "rmse".

    Returns
    -------
    dict
        Dictionary containing:
        - "best_window": best rolling window
        - "best_score": score of the selected window
        - "criterion": criterion used for selection
        - "comparison": DataFrame with window comparison
        - "best_output": rolling_regression output for best window
    """
    if not windows:
        raise ValueError("windows must be a non-empty list")

    if criterion not in {"ssr", "rmse"}:
        raise ValueError("criterion must be either 'ssr' or 'rmse'")

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    results = []
    outputs = {}

    for window in windows:
        output = rolling_regression(y=y, X=X, window=window)
        residuals = output["rolling_residuals"].dropna()

        if residuals.empty:
            ssr = np.nan
            rmse = np.nan
            n_obs = 0
        else:
            ssr = float((residuals ** 2).sum())
            rmse = float(np.sqrt((residuals ** 2).mean()))
            n_obs = int(len(residuals))

        results.append(
            {
                "window": window,
                "ssr": ssr,
                "rmse": rmse,
                "n_obs": n_obs,
            }
        )
        outputs[window] = output

    comparison = pd.DataFrame(results)

    if comparison[criterion].isna().all():
        raise ValueError("all candidate windows produced invalid results")

    comparison = comparison.sort_values(by=criterion).reset_index(drop=True)

    best_window = int(comparison.loc[0, "window"])
    best_score = float(comparison.loc[0, criterion])

    return {
        "best_window": best_window,
        "best_score": best_score,
        "criterion": criterion,
        "comparison": comparison,
        "best_output": outputs[best_window],
    }

def plot_window_selection(
    comparison: pd.DataFrame,
    criterion: str = "ssr",
    title: str = "Rolling Window Selection"
):
    """
    Plot rolling-window comparison results.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output comparison table from select_best_rolling_window.
    criterion : str, default "ssr"
        Metric to plot: "ssr" or "rmse".
    title : str
        Chart title.
    """
    if criterion not in {"ssr", "rmse"}:
        raise ValueError("criterion must be either 'ssr' or 'rmse'")

    if criterion not in comparison.columns:
        raise ValueError(f"{criterion} not found in comparison DataFrame")

    import matplotlib.pyplot as plt

    df = comparison.sort_values("window").copy()

    plt.figure(figsize=(10, 5))
    plt.plot(df["window"], df[criterion], marker="o")
    plt.title(title)
    plt.xlabel("Rolling Window")
    plt.ylabel(criterion.upper())
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_stationarity(series: pd.Series) -> dict:
    """
    Run Augmented Dickey-Fuller stationarity test on a time series.
    """
    clean_series = pd.Series(series).dropna()
    adf_result = adfuller(clean_series)

    return {
        "test_statistic": adf_result[0],
        "p_value": adf_result[1],
        "lags_used": adf_result[2],
        "n_obs": adf_result[3],
        "critical_values": adf_result[4],
        "is_stationary": adf_result[1] < 0.05,
    }


def granger_causality_test(y: pd.Series, X: pd.DataFrame, maxlag: int = 5) -> pd.DataFrame:
    """
    Test whether each predictor in X Granger-causes y.
    """
    if isinstance(maxlag, bool) or not isinstance(maxlag, int) or maxlag <= 0:
        raise ValueError("maxlag must be a positive integer")

    y_series = pd.Series(y, name="y")
    X_df = pd.DataFrame(X)
    records = []

    for variable in X_df.columns:
        test_df = pd.concat([y_series, X_df[variable]], axis=1).dropna()
        test_df.columns = ["y", variable]

        test_results = grangercausalitytests(test_df[["y", variable]], maxlag=maxlag, verbose=False)
        for lag, lag_result in test_results.items():
            p_value = lag_result[0]["ssr_ftest"][1]
            records.append(
                {
                    "variable": variable,
                    "lag": lag,
                    "p_value": p_value,
                    "is_significant": p_value < 0.05,
                }
            )

    return pd.DataFrame(records).sort_values(["variable", "lag"]).reset_index(drop=True)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance inflation factors for predictors in X.
    """
    X_df = pd.DataFrame(X).dropna()
    X_with_const = sm.add_constant(X_df, has_constant="add")

    vif_rows = []
    for i, variable in enumerate(X_with_const.columns):
        vif_rows.append({"variable": variable, "vif": variance_inflation_factor(X_with_const.values, i)})

    vif_df = pd.DataFrame(vif_rows)
    return vif_df[vif_df["variable"] != "const"].reset_index(drop=True)


def residual_diagnostics(residuals: pd.Series, lags: int = 10) -> dict:
    """
    Compute residual quality diagnostics for autocorrelation, stationarity, and normality.
    """
    if isinstance(lags, bool) or not isinstance(lags, int) or lags <= 0:
        raise ValueError("lags must be a positive integer")

    resid = pd.Series(residuals).dropna()
    adf_p_value = adfuller(resid)[1]
    ljung_box = acorr_ljungbox(resid, lags=[lags], return_df=True)
    jb_stat, jb_p_value, _, _ = jarque_bera(resid)

    return {
        "mean": resid.mean(),
        "std": resid.std(),
        "adf_p_value": adf_p_value,
        "is_stationary": adf_p_value < 0.05,
        "durbin_watson": durbin_watson(resid),
        "ljung_box_p_value": ljung_box["lb_pvalue"].iloc[0],
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p_value": jb_p_value,
        "is_normal": jb_p_value >= 0.05,
        "n_obs": int(resid.shape[0]),
    }


def build_residual_diagnostics(model) -> dict:
    """
    Build residual diagnostics directly from a fitted statsmodels OLS result.
    """
    return residual_diagnostics(model.resid)


def plot_residual_diagnostics(residuals: pd.Series, lags: int = 10):
    """
    Plot residual time series, distribution, and autocorrelation diagnostics.
    """
    resid = pd.Series(residuals).dropna()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    resid.plot(ax=axes[0], title="Residual Time Series")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Residual")

    axes[1].hist(resid, bins=30, edgecolor="black")
    axes[1].set_title("Residual Histogram")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")

    plot_acf(resid, lags=lags, ax=axes[2])
    axes[2].set_title("Residual Autocorrelation")

    fig.tight_layout()
    return fig, axes

def plot_correlation_heatmap(
    X: pd.DataFrame,
    title: str = "Feature Correlation Heatmap"
) -> pd.DataFrame:
    """
    Plot correlation heatmap for input regressors.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    corr = X.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()

    return corr