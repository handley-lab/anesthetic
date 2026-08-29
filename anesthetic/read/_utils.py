"""Utilities shared by chain readers."""
from itertools import compress, islice
import warnings
import numpy as np
from anesthetic.samples import (_compute_burn_in, _thin_weights,
                                _compress_repeats)


def _norm_columns(columns, parameters, renames=None):
    """Normalise a column selector to parameter positions and names.

    Parameters
    ----------
    columns : list[str], list[int], or slice, optional
        Parameter selector. Integer positions and slices index ``parameters``.
    parameters : list[str] or list[int]
        Available parameter names in file order.
        (Integer range as fallback.)
    renames : dict, optional
        Mapping from parameter names to new names.

    Returns
    -------
    indices : list[int] or None
        Selected parameter positions. ``None`` means that all file columns
        should be loaded.
    columns : list[str] or list[int]
        Names to use for the selected parameters.
    renames : dict
        Mapping from parameter names to new names.

    Raises
    ------
    IndexError
        If an integer position is out of range.
    KeyError
        If a requested parameter name is unknown.
    TypeError
        If selector types are mixed or unsupported.

    """
    nparams = len(parameters)
    renames = {} if renames is None else dict(renames)

    if columns is None:
        return None, list(parameters), renames

    if np.isscalar(columns):
        columns = [columns]

    # slice
    if isinstance(columns, slice):
        indices = list(range(nparams))[columns]
    # list[str]
    elif all(isinstance(c, str) for c in columns):
        try:
            indices = [parameters.index(c) for c in columns]
        except ValueError as error:
            if len(columns) == nparams:
                warnings.warn(
                    "Using `columns` to rename parameters is deprecated. "
                    "Use the `renames` keyword argument instead.",
                    FutureWarning,
                    stacklevel=3,
                )
                inferred = dict(zip(parameters, columns))
                inferred.update(renames)
                return None, list(parameters), inferred
            missing = next(c for c in columns if c not in parameters)
            raise KeyError(f"unknown parameter {missing!r}") from error
    # list[int]
    elif all(type(c) is int or isinstance(c, np.integer) for c in columns):
        indices = [range(nparams)[c] for c in columns]
    # incorrect type
    else:
        raise TypeError("`columns` must be a slice, a list of parameter "
                        "names, or a list of integer indices.")

    return indices, [parameters[i] for i in indices], renames


def _infer_weight_dtype(weights):
    """Convert weights to integers if every value is integral."""
    if np.all(weights == np.floor(weights)):
        return weights.astype(int)
    return weights


def _read_mcmc_chains(chain_files, parameters, columns, count_samples,
                      header_rows=0, burn_in=None, thin=None,
                      compress_repeats=False, renames=None):
    """Load selected columns from one or more weighted MCMC chain files."""
    nparams = len(parameters)
    indices, columns, renames = _norm_columns(columns, parameters, renames)

    if indices is None and compress_repeats:
        usecols = [0] + list(range(2, nparams + 2))
    elif indices is None:
        usecols = None
    elif compress_repeats:
        usecols = [0] + [i+2 for i in indices]
    else:
        if 'chi2' in parameters:
            index = parameters.index('chi2')
            if index not in indices:
                indices = indices + [index]
                columns = columns + ['chi2']
        usecols = [0, 1] + [i+2 for i in indices]

    chain_lengths = np.array([count_samples(file) for _, file in chain_files])
    if burn_in is None:
        nskip = np.zeros(len(chain_lengths), dtype=int)
    else:
        nskip = _compute_burn_in(burn_in, chain_lengths)
    selected_lengths = chain_lengths - nskip
    nskip = nskip + header_rows

    # Pre-compute thinned weights to determine the required allocation size.
    selected_weights = []
    if thin is not None:
        for j, ((_, chain_file), skiprows, selected) in enumerate(zip(
                chain_files, nskip, selected_lengths
        )):
            selected_weights.append(_thin_weights(
                np.loadtxt(chain_file, skiprows=skiprows, usecols=0, ndmin=1),
                thin,
            ))
            selected_lengths[j] = np.count_nonzero(selected_weights[-1])

    # Preallocate for all post-thinning rows; compression may use fewer rows.
    nrows = sum(selected_lengths)
    data = np.empty((nrows, len(columns)))
    weights = np.empty(nrows)
    minuslog = np.empty(nrows)  # minuslogP for Cobaya, minuslogL for GetDist
    chains = np.empty(nrows, dtype=int)

    start = 0
    for j, ((i, chain_file), skiprows, selected) in enumerate(zip(
            chain_files, nskip, selected_lengths
    )):
        if selected == 0:
            continue

        stop = start + selected
        if thin is None:
            # Fast path: load each chain in a single pass.
            chain_data = np.loadtxt(chain_file, skiprows=skiprows,
                                    usecols=usecols, ndmin=2)
            weights[start:stop] = chain_data[:, 0]
        else:
            # Load only the rows retained by the preceding weights-only pass.
            mask = selected_weights[j] > 0
            with open(chain_file) as file:
                lines = compress(islice(file, skiprows, None), mask)
                chain_data = np.loadtxt(lines, usecols=usecols, ndmin=2)
            weights[start:stop] = selected_weights[j][mask]

        if compress_repeats:
            indices, compressed_weights = _compress_repeats(
                chain_data[:, 1:], weights[start:stop]
            )
            compressed_data = chain_data[indices, 1:]
            stop = start + len(compressed_data)
            data[start:stop] = compressed_data
            weights[start:stop] = compressed_weights
        else:
            minuslog[start:stop] = chain_data[:, 1]
            data[start:stop] = chain_data[:, 2:]

        chains[start:stop] = int(i or 0)
        start = stop

    # Remove rows left unused by compression.
    if start < nrows:
        # refcheck=False needed for python 3.10
        data.resize((start, len(columns)), refcheck=False)
        weights.resize(start, refcheck=False)
        minuslog.resize(start, refcheck=False)
        chains.resize(start, refcheck=False)

    return (data, columns, _infer_weight_dtype(weights), minuslog, chains,
            renames)
