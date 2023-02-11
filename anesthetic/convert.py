"""Tools for converting to other outputs."""


def to_getdist(samples):
    """Convert from anesthetic to getdist samples.

    Parameters
    ----------
    samples: :class:`anesthetic.samples.Samples`
        anesthetic samples to be converted

    Returns
    -------
    getdist_samples: :class:`getdist.mcsamples.MCSamples`
        getdist equivalent samples
    """
    import getdist
    samples = samples.drop_labels()
    samples = samples.to_numpy()
    weights = samples.get_weights()
    loglikes = -samples.logL.to_numpy()
    names = samples.columns
    ranges = {name: (samples[name].min(), samples[name].max())
              for name in names}
    return getdist.mcsamples.MCSamples(samples=samples,
                                       weights=weights,
                                       loglikes=loglikes,
                                       ranges=ranges,
                                       names=names)
