import numpy as np
import pandas as pd

from src.regression import rolling_zscore

np.random.seed(0)
data = pd.Series(np.random.randn(100))

z = rolling_zscore(data, window=20)

print(z.tail())

import numpy as np
import pandas as pd

from src.regression import rolling_regression

np.random.seed(0)

X = pd.DataFrame({
    "x1": np.random.randn(100),
    "x2": np.random.randn(100),
})

y = 1.0 + 1.5 * X["x1"] - 0.7 * X["x2"] + np.random.randn(100) * 0.5

out = rolling_regression(y, X, window=20)

print(out["rolling_coefficients"].tail())
print(out["rolling_r2"].tail())
print(out["rolling_fitted_values"].tail())
print(out["rolling_residuals"].tail())