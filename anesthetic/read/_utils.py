"""Utilities shared by chain readers."""
import numpy as np


def normalise_columns(columns, parameters):
    """Normalise a column selector to parameter positions and names.

    Parameters
    ----------
    columns : list[str], list[int], or slice, optional
        Parameter selector. Integer positions and slices index ``parameters``.
    parameters : list[str]
        Available parameter names in file order.

    Returns
    -------
    indices : list[int] or None
        Selected parameter positions. ``None`` means that all file columns
        should be loaded.
    columns : list[str]
        Names to use for the selected parameters.

    Raises
    ------
    IndexError
        If an integer position is out of range.
    KeyError
        If a requested parameter name is unknown.
    TypeError
        If selector types are mixed or unsupported.

    """
    nparameters = len(parameters)

    if columns is None:
        return None, list(parameters)

    if np.isscalar(columns):
        columns = [columns]

    # slice
    if isinstance(columns, slice):
        indices = list(range(nparameters))[columns]
    # list[str]
    elif all(isinstance(c, str) for c in columns):
        try:
            indices = [parameters.index(c) for c in columns]
        except ValueError as error:
            missing = next(c for c in columns if c not in parameters)
            raise KeyError(f"unknown parameter {missing!r}") from error
    # list[int]
    elif all(type(c) is int or isinstance(c, np.integer) for c in columns):
        indices = [range(nparameters)[c] for c in columns]
    # incorrect type
    else:
        raise TypeError("`columns` must be a slice, a list of parameter "
                        "names, or a list of integer indices.")

    return indices, [parameters[i] for i in indices]
