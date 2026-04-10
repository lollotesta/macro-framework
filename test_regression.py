import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path("src").resolve()))

from regression import build_regression_output, summarize_latest

# dati finti
dates = pd.date_range("2020-01-01", periods=200, freq="B")

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path("src").resolve()))

from regression import build_regression_output, summarize_latest

# dati finti
dates = pd.date_range("2020-01-01", periods=200, freq="B")

X = pd.DataFrame({
    "rates": range(200),
    "breakeven": [i * 0.3 for i in range(200)]
}, index=dates)

y = 2.0 * X["rates"] - 1.0 * X["breakeven"] + 10

# run
results = build_regression_output(y, X)

print("\n=== REGRESSION TABLE ===")
print(results["table"])

print("\n=== LATEST ===")
print(summarize_latest(results))

