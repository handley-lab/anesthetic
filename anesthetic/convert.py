"""Tools for converting to other outputs."""


def to_getdist(nested_samples):
    """Convert from anesthetic to getdist samples.

    Parameters
    ----------
    nested_samples: MCMCSamples or NestedSamples
        anesthetic samples to be converted

    Returns
    -------
    getdist_samples: getdist.mcsamples.MCSamples
        getdist equivalent samples
    """
    import getdist
    samples = nested_samples.values
    weights = nested_samples.weights
    loglikes = -2*nested_samples.logL.values
    names = nested_samples.columns
    ranges = {name: nested_samples._limits(name) for name in names}
    return getdist.mcsamples.MCSamples(samples=samples,
                                       weights=weights,
                                       loglikes=loglikes,
                                       ranges=ranges,
                                       names=names)
