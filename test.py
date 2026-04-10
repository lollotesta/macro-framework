import numpy as np
import pandas as pd

from src.regression import rolling_zscore

np.random.seed(0)
data = pd.Series(np.random.randn(100))

z = rolling_zscore(data, window=20)

print(z.tail())