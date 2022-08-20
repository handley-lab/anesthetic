"""Utilities for reading data from files."""
import numpy as np


def remove_burn_in(data, burn_in):
    """Strip burn in from a dataset."""
    if burn_in:
        if 0 < burn_in < 1:
            burn_in *= len(data)
        return data[np.ceil(burn_in).astype(int):]
    return data
