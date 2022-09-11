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
    nested_samples = nested_samples.drop_labels()
    samples = nested_samples.to_numpy()
    weights = nested_samples.get_weights()
    loglikes = -nested_samples.logL.to_numpy()
    names = nested_samples.columns
    ranges = {name: (nested_samples[name].min(), nested_samples[name].max())
              for name in names}
    return getdist.mcsamples.MCSamples(samples=samples,
                                       weights=weights,
                                       loglikes=loglikes,
                                       ranges=ranges,
                                       names=names)
