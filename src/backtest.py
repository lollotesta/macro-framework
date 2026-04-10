import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_signal(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate a simple mean-reversion position from a z-score series.

    Parameters
    ----------
    zscore : pd.Series
        Z-score series.
    entry_threshold : float, default 2.0
        Entry threshold for opening positions.
    exit_threshold : float, default 0.5
        Exit threshold for closing positions.

    Returns
    -------
    pd.Series
        Position series taking values {-1, 0, 1}.
    """
    if not isinstance(zscore, pd.Series):
        raise TypeError("zscore must be a pandas Series")

    if entry_threshold <= 0:
        raise ValueError("entry_threshold must be > 0")

    if exit_threshold < 0:
        raise ValueError("exit_threshold must be >= 0")

    if exit_threshold >= entry_threshold:
        raise ValueError("exit_threshold must be smaller than entry_threshold")

    position = pd.Series(index=zscore.index, dtype=float)
    current_position = 0

    for i, val in enumerate(zscore):
        if pd.isna(val):
            position.iloc[i] = current_position
            continue

        if current_position == 0:
            if val >= entry_threshold:
                current_position = -1
            elif val <= -entry_threshold:
                current_position = 1

        elif current_position == -1:
            if val <= exit_threshold:
                current_position = 0

        elif current_position == 1:
            if val >= -exit_threshold:
                current_position = 0

        position.iloc[i] = current_position

    return position.astype(int)