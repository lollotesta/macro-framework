import numpy as np
import pandas as pd

from src.regression import (
    run_ols,
    build_regression_output,
    rolling_regression,
    rolling_zscore,
    compute_fair_value_deviation,
    select_best_rolling_window,
    plot_window_selection,
    test_stationarity,
    granger_causality_test,
    compute_vif,
    residual_diagnostics,
    plot_residual_diagnostics,
    plot_correlation_heatmap,
    plot_fitted_vs_observed,
    plot_zscore,
    plot_actual_fair_value_and_zscore,
    plot_rolling_coefficients,
)


# =========================
# 1. FAKE DATA GENERATION
# =========================

np.random.seed(42)
n = 250
index = pd.date_range("2020-01-01", periods=n, freq="D")

x1 = np.random.normal(0, 1, n)
x2 = 0.5 * x1 + np.random.normal(0, 1, n)
noise = np.random.normal(0, 0.5, n)

y = 1.5 + 2.0 * x1 - 1.0 * x2 + noise

df = pd.DataFrame(
    {
        "y": y,
        "x1": x1,
        "x2": x2,
    },
    index=index,
)

X = df[["x1", "x2"]]
y = df["y"]


# =========================
# 2. BASE OLS REGRESSION
# =========================

print("\n=== BASE OLS REGRESSION ===")
model = run_ols(y, X)
output = build_regression_output(model)

print("\nCoefficients:")
print(output["coefficients"])

print("\nR-squared:")
print(output["r_squared"])

print("\nModel summary:")
print(model.summary())


# =========================
# 3. RESIDUAL DIAGNOSTICS
# =========================

print("\n=== RESIDUAL DIAGNOSTICS ===")
res_diag = residual_diagnostics(output["residuals"])
print(res_diag)

plot_residual_diagnostics(output["residuals"])


# =========================
# 4. STATIONARITY TEST
# =========================

print("\n=== STATIONARITY TEST ON y ===")
stat_test = test_stationarity(y)
print(stat_test)


# =========================
# 5. GRANGER CAUSALITY
# =========================

print("\n=== GRANGER CAUSALITY TEST ===")
granger = granger_causality_test(y=y, X=X, maxlag=5)
print(granger)


# =========================
# 6. VIF
# =========================

print("\n=== VIF ===")
vif = compute_vif(X)
print(vif)


# =========================
# 7. FEATURE CORRELATION HEATMAP
# =========================

print("\n=== FEATURE CORRELATION HEATMAP ===")
corr = plot_correlation_heatmap(X, title="Feature Correlation Heatmap")
print(corr)


# =========================
# 8. ROLLING REGRESSION
# =========================

print("\n=== ROLLING REGRESSION ===")
rolling_out = rolling_regression(y=y, X=X, window=40)

rolling_key = (
    "rolling_coefficients"
    if "rolling_coefficients" in rolling_out
    else "rolling_betas"
)

print("\nLast rolling coefficients:")
print(rolling_out[rolling_key].dropna(how="all").tail())

print("\nLast rolling fitted values:")
print(rolling_out["rolling_fitted_values"].dropna().tail())

print("\nLast rolling residuals:")
print(rolling_out["rolling_residuals"].dropna().tail())


# =========================
# 9. ROLLING BETA STABILITY PLOT
# =========================

print("\n=== PLOTTING ROLLING COEFFICIENTS ===")
plot_rolling_coefficients(
    rolling_out[rolling_key],
    title="Rolling Betas Stability",
)


# =========================
# 10. FAIR VALUE / DEVIATION / Z-SCORE
# =========================

print("\n=== FAIR VALUE / DEVIATION / Z-SCORE ===")
fair_value = rolling_out["rolling_fitted_values"]
deviation = compute_fair_value_deviation(actual=y, fair_value=fair_value)
zscore = rolling_zscore(deviation, window=20)

print("\nLast deviations:")
print(deviation.dropna().tail())

print("\nLast z-scores:")
print(zscore.dropna().tail())


# =========================
# 11. PLOTS: OBSERVED VS FITTED / Z-SCORE
# =========================

print("\n=== PLOTTING OBSERVED VS FITTED ===")
plot_fitted_vs_observed(
    observed=y,
    fitted=fair_value,
    title="Observed vs Rolling Fair Value",
)

print("\n=== PLOTTING Z-SCORE ===")
plot_zscore(
    zscore=zscore,
    title="Residual Z-Score Over Time",
)

print("\n=== PLOTTING COMBINED CHART ===")
plot_actual_fair_value_and_zscore(
    actual=y,
    fair_value=fair_value,
    zscore=zscore,
    title="Rolling Fair Value Signal",
)


# =========================
# 12. WINDOW SELECTION
# =========================

print("\n=== WINDOW SELECTION ===")
best = select_best_rolling_window(
    y=y,
    X=X,
    windows=[20, 40, 60, 80],
    criterion="rmse",
)

print("\nBest window:")
print(best["best_window"])

print("\nBest score:")
print(best["best_score"])

print("\nComparison table:")
print(best["comparison"])

plot_window_selection(
    comparison=best["comparison"],
    criterion="rmse",
    title="RMSE by Rolling Window",
)


print("\n=== TEST COMPLETED SUCCESSFULLY ===")
