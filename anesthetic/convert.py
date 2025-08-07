"""Tools for converting to other outputs."""
import numpy as np


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


def from_chainconsumer(chain, columns=None):
    """Convert ChainConsumer Chain object to anesthetic samples.

    Parameters
    ----------
    chain : Chain
        ChainConsumer Chain object
    columns : list, optional
        Parameter names to use. If None, uses chain.data_columns

    Returns
    -------
    samples : MCMCSamples
        Anesthetic MCMCSamples object
    """
    try:
        from chainconsumer.chain import Chain  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError("You need to install ChainConsumer to use "
                          "from_chainconsumer")

    from .samples import MCMCSamples

    chain_data = chain.data_samples
    
    # Only include logL if columns is None (i.e., using all data)
    # If specific columns are requested, don't add logL unless explicitly requested
    include_logL = columns is None and chain.log_posterior is not None
    
    return MCMCSamples(
        chain_data.values,
        weights=chain.weights,
        columns=columns or chain.data_columns,
        labels=columns or chain.data_columns,
        logL=chain.log_posterior if include_logL else None
    )


def to_chainconsumer(samples, params=None, name=None, filter_zero_weights=True, **kwargs):
    """Convert anesthetic samples to ChainConsumer Chain object.

    Parameters
    ----------
    samples : :class:`anesthetic.samples.Samples`
        Anesthetic samples object to be converted
    params : list, optional
        List of parameter names to include. If None, uses all parameter
        columns from the samples (excluding labels and weight columns).
    name : str, optional
        Name for the chain. If None, uses sample label or 'chain1'.
    filter_zero_weights : bool, optional
        If True, filter out samples with zero weights (recommended for ChainConsumer).
        Default is True.
    **kwargs : dict
        Additional keyword arguments to pass to Chain constructor.

    Returns
    -------
    chain : Chain
        ChainConsumer Chain object
    """
    try:
        from chainconsumer.chain import Chain
    except ModuleNotFoundError:
        raise ImportError("You need to install ChainConsumer to use "
                          "to_chainconsumer")

    import pandas as pd
    import numpy as np

    if name is None:
        if hasattr(samples, 'label') and samples.label:
            name = samples.label
        else:
            name = 'anesthetic_chain'

    index = samples.drop_labels().columns

    if params is None:
        params_to_use = index.tolist()
        positions = list(range(len(index)))
    else:
        params_to_use = params
        positions = [index.get_loc(p) for p in params]

    weights = samples.get_weights()
    param_data = samples.to_numpy()[:, positions]
    log_posterior = samples.logL.to_numpy() if hasattr(samples, 'logL') and 'logL' in samples.columns else None
    # Chainconsumer requires weights to be > 0.
    if filter_zero_weights:
        valid_mask = weights > 0
        n_filtered = (~valid_mask).sum()
        if n_filtered > 0:
            print(f"Filtering {n_filtered} zero-weight samples out of {len(weights)} for ChainConsumer")
            weights = weights[valid_mask]
            param_data = param_data[valid_mask, :]
            if log_posterior is not None:
                log_posterior = log_posterior[valid_mask]
    
    if samples.islabelled():
        latex_labels = samples.get_labels()[positions].tolist()
    else:
        latex_labels = params_to_use
    
    df_dict = {}
    for j, latex_label in enumerate(latex_labels):
        df_dict[latex_label] = param_data[:, j]
    
    df_dict['weight'] = weights
    
    if log_posterior is not None:
        df_dict['log_posterior'] = log_posterior
    
    df = pd.DataFrame(df_dict)

    return Chain(
        samples=df,
        name=name,
        **kwargs
    )
