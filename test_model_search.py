import numpy as np
import pandas as pd

from src.regression import search_model_specifications

# =========================
# 1. FAKE DATA
# =========================

np.random.seed(42)
n = 250
index = pd.date_range("2020-01-01", periods=n, freq="D")

x1 = np.random.normal(0, 1, n)
x2 = 0.5 * x1 + np.random.normal(0, 1, n)
x3 = np.random.normal(0, 1, n)
x4 = 0.3 * x2 + np.random.normal(0, 1, n)

noise = np.random.normal(0, 0.5, n)

# y is mostly driven by x1 and x2
y = 1.5 + 2.0 * x1 - 1.0 * x2 + 0.2 * x3 + noise

X = pd.DataFrame(
    {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
    },
    index=index,
)

y = pd.Series(y, index=index, name="y")

# =========================
# 2. MODEL SEARCH
# =========================

results = search_model_specifications(
    y=y,
    X=X,
    min_features=1,
    max_features=3,
    top_n=10,
)

# =========================
# 3. OUTPUT
# =========================

print("\n=== TOP MODEL SPECIFICATIONS ===")
print(results)

print("\n=== TOP 5 REGRESSOR SETS ===")
print(results[["rank", "regressors", "adj_r_squared", "max_vif", "residual_is_stationary"]].head())