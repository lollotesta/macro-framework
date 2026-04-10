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
from typing import Optional
import itertools


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
    plt.show()
    return fig, axes

def plot_correlation_heatmap(
    X: pd.DataFrame,
    title: str = "Feature Correlation Heatmap"
) -> pd.DataFrame:
    """
    Plot correlation heatmap for input regressors.
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

def plot_rolling_coefficients(
    rolling_coefficients: pd.DataFrame,
    title: str = "Rolling Coefficients"
) -> pd.DataFrame:
    """
    Plot rolling regression coefficients through time.

    Parameters
    ----------
    rolling_coefficients : pd.DataFrame
        DataFrame of rolling coefficients, typically from rolling_regression().
    title : str
        Chart title.

    Returns
    -------
    pd.DataFrame
        Cleaned rolling coefficients used for plotting.
    """
    if not isinstance(rolling_coefficients, pd.DataFrame):
        raise ValueError("rolling_coefficients must be a pandas DataFrame")

    coef = rolling_coefficients.dropna(how="all")

    if coef.empty:
        raise ValueError("rolling_coefficients contains no plottable data")

    plt.figure(figsize=(12, 6))
    for column in coef.columns:
        plt.plot(coef.index, coef[column], label=column)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Coefficient Value")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return coef

def compute_rolling_driver_attribution(
    y: pd.Series,
    X: pd.DataFrame,
    window: int
) -> dict:
    """
    Compute rolling model quality and marginal driver attribution via OLS.

    For each rolling window, this function fits a full model and reduced models
    that exclude each regressor one-at-a-time. Marginal contribution is defined
    as:

        contribution_i = R²_full - R²_without_i

    Notes
    -----
    Contributions are marginal and generally not uniquely identifiable when
    regressors are correlated. Multicollinearity can redistribute explanatory
    power across variables.

    Marginal contributions may occasionally be near zero or negative due to
    multicollinearity or estimation noise.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    X : pd.DataFrame
        Regressors.
    window : int
        Positive rolling window length.

    Returns
    -------
    dict
        {
            "rolling_r2": pd.Series,
            "raw_contributions": pd.DataFrame,
            "normalized_contributions": pd.DataFrame,
        }
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if X.shape[1] == 0:
        raise ValueError("X must contain at least one regressor")

    df = pd.concat([y.rename("y"), X], axis=1).dropna()

    if window > len(df):
        raise ValueError("window cannot be larger than the number of valid observations")

    y_aligned = df["y"]
    X_aligned = df.drop(columns="y")
    columns = list(X_aligned.columns)
    index = y_aligned.index

    rolling_r2 = pd.Series(np.nan, index=index, dtype=float)
    raw_contributions = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    normalized_contributions = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)

    for end in range(window - 1, len(df)):
        window_slice = slice(end - window + 1, end + 1)
        y_window = y_aligned.iloc[window_slice]
        X_window = X_aligned.iloc[window_slice]

        full_model = run_ols(y_window, X_window)
        r2_full = full_model.rsquared

        idx = index[end]
        rolling_r2.loc[idx] = r2_full

        for variable in columns:
            if len(columns) == 1:
                raw_contributions.loc[idx, variable] = r2_full
            else:
                reduced_X = X_window.drop(columns=variable)
                reduced_model = run_ols(y_window, reduced_X)
                raw_contributions.loc[idx, variable] = r2_full - reduced_model.rsquared

        row_sum = raw_contributions.loc[idx].sum(skipna=True)
        if np.isclose(row_sum, 0.0):
            normalized_contributions.loc[idx] = np.nan
        else:
            normalized_contributions.loc[idx] = raw_contributions.loc[idx] / row_sum

    return {
        "rolling_r2": rolling_r2,
        "raw_contributions": raw_contributions,
        "normalized_contributions": normalized_contributions,
    }


def plot_rolling_r2(
    rolling_r2: pd.Series,
    title: str = "Rolling R2"
) -> pd.Series:
    """Plot rolling R² through time and return the plotted series."""
    series = pd.Series(rolling_r2).dropna()

    if series.empty:
        raise ValueError("rolling_r2 contains no plottable data")

    plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values, label="Rolling R2")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("R²")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rolling_r2


def plot_driver_attribution(
    normalized_contributions: pd.DataFrame,
    title: str = "Rolling Driver Attribution"
) -> pd.DataFrame:
    """
    Plot normalized rolling driver attribution as a stacked area chart.

    The y-axis represents proportional contribution share (0 to 1).
    """
    if not isinstance(normalized_contributions, pd.DataFrame):
        raise ValueError("normalized_contributions must be a pandas DataFrame")

    df = normalized_contributions.dropna(how="all")

    if df.empty:
        raise ValueError("normalized_contributions contains no plottable data")

    plt.figure(figsize=(12, 6))
    plt.stackplot(df.index, *[df[col].values for col in df.columns], labels=df.columns)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Contribution Share")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    return normalized_contributions

def search_model_specifications(
    y,
    X,
    min_features: int = 1,
    max_features: Optional[int] = None,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Exhaustively evaluate full-sample OLS model specifications and rank by adjusted R².

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X : pd.DataFrame or array-like
        Candidate regressors.
    min_features : int, default 1
        Minimum number of regressors per specification.
    max_features : int or None, default None
        Maximum number of regressors per specification. If None, use all columns in X.
    top_n : int or None, default None
        If provided, return only the top N ranked specifications.

    Returns
    -------
    pd.DataFrame
        Model-specification summary sorted by adjusted R² (descending), including rank.
    """
    y_series = y.rename("y") if isinstance(y, pd.Series) else pd.Series(y, name="y")
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    output_columns = [
        "rank",
        "regressors",
        "n_regressors",
        "r_squared",
        "adj_r_squared",
        "aic",
        "bic",
        "max_vif",
        "avg_vif",
        "residual_adf_p_value",
        "residual_is_stationary",
    ]

    if X_df.empty:
        return pd.DataFrame(columns=output_columns)

    if isinstance(min_features, bool) or not isinstance(min_features, int) or min_features <= 0:
        raise ValueError("min_features must be a positive integer")

    n_candidates = X_df.shape[1]
    upper = n_candidates if max_features is None else max_features

    if isinstance(upper, bool) or not isinstance(upper, int):
        raise ValueError("max_features must be an integer or None")
    if upper <= 0:
        raise ValueError("max_features must be positive")
    if min_features > upper:
        raise ValueError("min_features cannot exceed max_features")

    upper = min(upper, n_candidates)
    if min_features > upper:
        return pd.DataFrame(columns=output_columns)

    records = []

    for k in range(min_features, upper + 1):
        for regressors in itertools.combinations(X_df.columns, k):
            subset = pd.concat([y_series, X_df.loc[:, regressors]], axis=1).dropna()

            if subset.empty or subset.shape[0] <= (k + 1):
                continue

            y_sub = subset["y"]
            X_sub = subset.drop(columns="y")

            try:
                model = run_ols(y_sub, X_sub)
            except Exception:
                continue

            max_vif = np.nan
            avg_vif = np.nan
            try:
                vif_df = compute_vif(X_sub)
                if not vif_df.empty:
                    max_vif = float(vif_df["vif"].max())
                    avg_vif = float(vif_df["vif"].mean())
            except Exception:
                pass

            residual_adf_p_value = np.nan
            residual_is_stationary = np.nan
            try:
                stationarity = test_stationarity(model.resid)
                residual_adf_p_value = float(stationarity["p_value"])
                residual_is_stationary = bool(stationarity["is_stationary"])
            except Exception:
                pass

            records.append(
                {
                    "regressors": tuple(regressors),
                    "n_regressors": int(k),
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "max_vif": max_vif,
                    "avg_vif": avg_vif,
                    "residual_adf_p_value": residual_adf_p_value,
                    "residual_is_stationary": residual_is_stationary,
                }
            )

    result = pd.DataFrame(records)

    if result.empty:
        return pd.DataFrame(columns=output_columns)

    result = result.sort_values(
        by=["adj_r_squared", "n_regressors"],
        ascending=[False, True]
    ).reset_index(drop=True)

    result.insert(0, "rank", np.arange(1, len(result) + 1))

    if top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer or None")
        result = result.head(top_n).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)

    return result