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


def from_chainconsumer(cc, columns=None):
    """Convert ChainConsumer object to anesthetic samples.

    Parameters
    ----------
    cc : ChainConsumer
        ChainConsumer object containing one or more chains
    columns : list, optional
        Parameter names to use. If None, uses chain.parameters

    Returns
    -------
    samples : MCMCSamples or dict
        If single chain: returns MCMCSamples object directly
        If multiple chains: returns dictionary mapping chain names to
        MCMCSamples objects
    """
    try:
        from chainconsumer import ChainConsumer  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError("You need to install ChainConsumer to use "
                          "from_chainconsumer")

    from .samples import MCMCSamples

    samples_dict = {}
    for chain in cc.chains:
        samples_dict[chain.name] = MCMCSamples(
            chain.chain,
            weights=chain.weights,
            columns=columns or chain.parameters,
            labels=chain.parameters
        )

    # If only one chain, return the samples directly instead of dictionary
    if len(samples_dict) == 1:
        return list(samples_dict.values())[0]

    return samples_dict


def to_chainconsumer(samples, params=None, names=None, colors=None,
                     chainconsumer=None, **kwargs):
    """Convert anesthetic samples to ChainConsumer object.

    Parameters
    ----------
    samples : :class:`anesthetic.samples.Samples` or list
        Single anesthetic samples object or list of anesthetic samples to be
        converted
    params : list, optional
        List of parameter names to include. If None, uses all parameter
        columns from the samples (excluding labels and weight columns).
    names : str or list, optional
        Name(s) for the chain(s) in ChainConsumer. If single samples and str
        provided, uses that name. If list of samples, should be list of names
        with same length. If None, uses sample labels or generates names like
        'chain1', 'chain2', etc.
    colors : str or list, optional
        Color(s) for the chain(s) in ChainConsumer. If single samples and str
        provided, uses that color. If list of samples, should be list of
        colors with same length. If None, ChainConsumer will use default
        colors.
    chainconsumer : ChainConsumer, optional
        Existing ChainConsumer object to add chains to. If None, creates a
        new one.
    **kwargs : dict
        Additional keyword arguments to pass to ChainConsumer.add_chain().
        Can be a single dict (applied to all chains) or list of dicts (one
        per chain).

    Returns
    -------
    chainconsumer : ChainConsumer
        ChainConsumer object with the samples added
    """
    try:
        from chainconsumer import ChainConsumer
    except ModuleNotFoundError:
        raise ImportError("You need to install ChainConsumer to use "
                          "to_chainconsumer")

    # Handle single sample vs list of samples
    if not isinstance(samples, list):
        samples = [samples]

    # Handle names
    if names is None:
        names = []
        for i, sample in enumerate(samples):
            # Use the sample's label if available, otherwise default naming
            if hasattr(sample, 'label') and sample.label:
                names.append(sample.label)
            else:
                names.append(f"chain{i+1}")
    elif isinstance(names, str):
        if len(samples) == 1:
            names = [names]
        else:
            raise ValueError("If providing string name, samples must be a "
                             "single object, not a list")
    elif len(names) != len(samples):
        raise ValueError("Length of names must match length of samples list")

    # Handle colors
    if colors is not None:
        if isinstance(colors, str):
            if len(samples) == 1:
                colors = [colors]
            else:
                raise ValueError("If providing string color, samples must "
                                 "be a single object, not a list")
        elif len(colors) != len(samples):
            raise ValueError("Length of colors must match length of "
                             "samples list")

    # Use existing ChainConsumer object or create new one
    c = chainconsumer if chainconsumer is not None else ChainConsumer()

    # Add each chain
    for i, sample in enumerate(samples):
        # Get parameter columns and positions
        index = sample.drop_labels().columns

        # If params not specified, use all parameter columns
        if params is None:
            params_to_use = index.tolist()
            positions = list(range(len(index)))
        else:
            params_to_use = params
            positions = [index.get_loc(p) for p in params]

        # Get labels for the selected parameters
        if sample.islabelled():
            labels = sample.get_labels()[positions].tolist()
        else:
            labels = params_to_use

        # Handle kwargs - can be single dict or list of dicts
        if isinstance(kwargs, list):
            if len(kwargs) != len(samples):
                raise ValueError("If kwargs is a list, it must have same "
                                 "length as samples")
            chain_kwargs = kwargs[i] if i < len(kwargs) else {}
        else:
            chain_kwargs = kwargs

        # Add color to kwargs if provided
        if colors is not None:
            chain_kwargs = chain_kwargs.copy()  # Don't modify original
            chain_kwargs['color'] = colors[i]

        # Add the chain
        c.add_chain(
            sample.to_numpy()[:, positions],
            weights=sample.get_weights(),
            parameters=labels,
            name=names[i],
            **chain_kwargs
        )

    return c
