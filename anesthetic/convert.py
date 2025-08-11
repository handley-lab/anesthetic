"""Tools for converting to other outputs."""
import numpy as np
import pandas as pd

try:
    from importlib.metadata import version, PackageNotFoundError
    cc_version = version('chainconsumer')
    CHAINCONSUMER_V1 = cc_version.startswith('1.')
except PackageNotFoundError:
    CHAINCONSUMER_V1 = False


def to_getdist(samples):
    """Convert from anesthetic to getdist samples.

    Parameters
    ----------
    samples : :class:`anesthetic.samples.Samples`
        anesthetic samples to be converted

    Returns
    -------
    getdist_samples : :class:`getdist.mcsamples.MCSamples`
        getdist equivalent samples
    """
    try:
        import getdist
    except ModuleNotFoundError:
        raise ImportError("You need to install getdist to use to_getdist")
    labels = np.char.strip(samples.get_labels().astype(str), '$')
    samples = samples.drop_labels()
    ranges = samples.agg(['min', 'max']).T.apply(tuple, axis=1).to_dict()
    return getdist.mcsamples.MCSamples(samples=samples.to_numpy(),
                                       weights=samples.get_weights(),
                                       loglikes=-samples.logL.to_numpy(),
                                       names=samples.columns,
                                       ranges=ranges,
                                       labels=labels)


def from_chainconsumer(cc_object, columns=None):
    """Convert a ChainConsumer object or Chain back to anesthetic samples.

    This function automatically detects the ChainConsumer version and converts
    the appropriate object type back to anesthetic format.

    ChainConsumer v0.x (< 1.0.0):
        - Input: ChainConsumer object with .chains list
        - Output: Dict of MCMCSamples (multiple chains) or single MCMCSamples
    ChainConsumer v1.x (>= 1.0.0):
        - Input: Single Chain object with .samples DataFrame
        - Output: Single MCMCSamples object
        - Weights and log_posterior are extracted from Chain structure

    Parameters
    ----------
    cc_object : ChainConsumer or Chain
        - For ChainConsumer v0.x: ChainConsumer object containing one or
          more chains
        - For ChainConsumer v1.x: Single Chain object with samples data
    columns : list, optional
        Parameter names to extract. If None, uses all available parameters:
        - v0.x: Uses chain.parameters for each chain
        - v1.x: Uses chain.data_columns (excluding weight/log_posterior)

    Returns
    -------
    samples : MCMCSamples or dict
        - For single chain: Returns MCMCSamples object directly
        - For multiple chains (v0.x only): Returns dict mapping chain names
          to MCMCSamples objects
        - Includes weights and log-likelihood (logL) when available

    Raises
    ------
    ImportError
        If ChainConsumer is not installed
    """
    try:
        import chainconsumer  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError("You need to install ChainConsumer to use "
                          "from_chainconsumer")

    from .samples import MCMCSamples

    if CHAINCONSUMER_V1:
        chain_data = cc_object.data_samples
        include_logL = columns is None and cc_object.log_posterior is not None
        return MCMCSamples(
            chain_data.values,
            weights=cc_object.weights,
            columns=columns or cc_object.data_columns,
            labels=columns or cc_object.data_columns,
            logL=cc_object.log_posterior if include_logL else None
        )
    else:
        samples_dict = {}
        for ch in cc_object.chains:
            samples_dict[ch.name] = MCMCSamples(
                ch.chain,
                weights=ch.weights,
                columns=columns or ch.parameters,
                labels=ch.parameters
            )
        if len(samples_dict) == 1:
            return list(samples_dict.values())[0]
        return samples_dict


def to_chainconsumer(samples, params=None, name=None, **kwargs):
    """Convert anesthetic samples to a ChainConsumer object or Chain.

    This function automatically detects the installed ChainConsumer version and
    uses the appropriate conversion method.

    ChainConsumer v0.x (< 1.0.0):
        - Accepts single samples or lists of samples
        - Returns a ChainConsumer object with .chains attribute
        - Supports multiple chains via lists
    ChainConsumer v1.x (>= 1.0.0):
        - Accepts only single samples (no lists)
        - Returns a Chain object with .samples DataFrame
        - Zero weights are automatically filtered out
        - Uses pandas DataFrame structure

    Parameters
    ----------
    samples : :class:`anesthetic.samples.Samples` or list of them
        anesthetic samples to be converted. For v1.x, must be a single
        sample.
        For v0.x, can be a single sample or list of samples.
    params : list, optional
        List of parameter names to include. If None, uses all available
        parameters.
    name : str or list of str, optional
        Name(s) for the chain(s). If None, uses sample labels or default names.
        For v0.x with multiple samples, can be a list of names.
    chain_specific_kwargs : list of dict, optional (v0.x only)
        List of dictionaries containing kwargs specific to each chain.
        Must have same length as samples list.
    **kwargs : dict
        Additional keyword arguments passed to ChainConsumer/Chain constructor.

    Returns
    -------
    chain_or_consumer : Chain or ChainConsumer
        - For ChainConsumer v1.x: Returns a Chain object with .samples,
          .weights, .name attributes
        - For ChainConsumer v0.x: Returns a ChainConsumer object with .chains
          list attribute

    Raises
    ------
    TypeError
        If ChainConsumer v1.x is used with a list of samples (not
        supported)
    ImportError
        If ChainConsumer is not installed
    """
    try:
        import chainconsumer  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError("You need to install ChainConsumer to use "
                          "to_chainconsumer")

    if CHAINCONSUMER_V1:
        if isinstance(samples, list):
            raise TypeError("ChainConsumer >= 1.0.0 only supports converting "
                            "a single anesthetic sample object at a time.")
        return _to_chainconsumer_v1(samples, params=params, name=name,
                                    **kwargs)
    else:
        return _to_chainconsumer_v0(samples, params=params, names=name,
                                    **kwargs)


def _to_chainconsumer_v1(samples, params=None, name=None, **kwargs):
    """Convert anesthetic samples to ChainConsumer v1.x Chain object."""
    from chainconsumer.chain import Chain

    if name is None:
        name = (getattr(samples, 'label', 'anesthetic_chain') or
                'anesthetic_chain')

    index = samples.drop_labels().columns
    params_to_use = params or index.tolist()
    positions = [index.get_loc(p) for p in params_to_use]

    weights = samples.get_weights()
    param_data = samples.to_numpy()[:, positions]
    log_posterior = samples.logL.to_numpy() if 'logL' in samples else None

    valid_mask = weights > 0
    if (~valid_mask).any():
        weights = weights[valid_mask]
        param_data = param_data[valid_mask, :]
        if log_posterior is not None:
            log_posterior = log_posterior[valid_mask]

    latex_labels = (samples.get_labels()[positions].tolist()
                    if samples.islabelled() else params_to_use)

    df_dict = {label: param_data[:, j]
               for j, label in enumerate(latex_labels)}
    df_dict['weight'] = weights
    if log_posterior is not None:
        df_dict['log_posterior'] = log_posterior

    return Chain(samples=pd.DataFrame(df_dict), name=name, **kwargs)


def _to_chainconsumer_v0(samples, params=None, names=None,
                         cc=None, chain_kwargs=None, **kwargs):
    """Convert anesthetic samples to ChainConsumer v0.x object."""
    from chainconsumer import ChainConsumer

    if not isinstance(samples, list):
        samples = [samples]

    if names is None:
        names = [getattr(s, 'label', f'chain{i+1}') or f'chain{i+1}'
                 for i, s in enumerate(samples)]
    elif isinstance(names, str):
        if len(samples) > 1:
            raise ValueError("String 'names' is only valid for a single "
                             "sample object.")
        names = [names]
    elif len(names) != len(samples):
        raise ValueError("Length of 'names' must match length of 'samples'.")

    c = cc or ChainConsumer()

    chain_specific_kwargs = kwargs.pop('chain_specific_kwargs', None)
    if chain_specific_kwargs is not None:
        if not isinstance(chain_specific_kwargs, list):
            raise ValueError("chain_specific_kwargs must be a list of "
                             "dictionaries")
        if len(chain_specific_kwargs) != len(samples):
            raise ValueError("chain_specific_kwargs must be a list with "
                             "the same length as samples")

    for i, sample in enumerate(samples):
        index = sample.drop_labels().columns
        params_to_use = params or index.tolist()
        positions = [index.get_loc(p) for p in params_to_use]
        labels = (sample.get_labels()[positions].tolist()
                  if sample.islabelled() else params_to_use)

        if chain_specific_kwargs:
            final_chain_kwargs = kwargs.copy()
            final_chain_kwargs.update(chain_specific_kwargs[i])
        else:
            final_chain_kwargs = kwargs

        c.add_chain(
            sample.to_numpy()[:, positions],
            weights=sample.get_weights(),
            parameters=labels,
            name=names[i],
            **final_chain_kwargs
        )
    return c
