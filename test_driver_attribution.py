import numpy as np
import pandas as pd

from src.regression import (
    compute_rolling_driver_attribution,
    plot_rolling_r2,
    plot_driver_attribution,
)

# =========================
# 1. FAKE DATA
# =========================

np.random.seed(42)
n = 250
index = pd.date_range("2020-01-01", periods=n, freq="D")

x1 = np.random.normal(0, 1, n)
x2 = 0.4 * x1 + np.random.normal(0, 1, n)
x3 = np.random.normal(0, 1, n)

# y mostly driven by x1 and x2, weakly by x3
noise = np.random.normal(0, 0.5, n)
y = 1.0 + 1.8 * x1 - 0.9 * x2 + 0.2 * x3 + noise

X = pd.DataFrame(
    {
        "x1": x1,
        "x2": x2,
        "x3": x3,
    },
    index=index,
)

y = pd.Series(y, index=index, name="y")

# =========================
# 2. DRIVER ATTRIBUTION
# =========================

window = 40

attrib = compute_rolling_driver_attribution(
    y=y,
    X=X,
    window=window,
)

print("\n=== LAST ROLLING R2 ===")
print(attrib["rolling_r2"].dropna().tail())

print("\n=== LAST RAW CONTRIBUTIONS ===")
print(attrib["raw_contributions"].dropna(how="all").tail())

print("\n=== LAST NORMALIZED CONTRIBUTIONS ===")
print(attrib["normalized_contributions"].dropna(how="all").tail())

print("\n=== ROW SUM CHECK (should be close to 1) ===")
print(attrib["normalized_contributions"].dropna(how="all").sum(axis=1).tail())

# =========================
# 3. PLOTS
# =========================

plot_rolling_r2(
    attrib["rolling_r2"],
    title="Rolling Model R2",
)

plot_driver_attribution(
    attrib["normalized_contributions"],
    title="Rolling Driver Attribution",
)