"""Tools for converting to other outputs."""
from anesthetic.samples import NestedSamples, MCMCSamples


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
    samples = nested_samples.to_numpy()
    weights = nested_samples.weights
    loglikes = -2*nested_samples.logL.to_numpy()
    names = nested_samples.columns
    ranges = {name: nested_samples._limits(name) for name in names}
    return getdist.mcsamples.MCSamples(samples=samples,
                                       weights=weights,
                                       loglikes=loglikes,
                                       ranges=ranges,
                                       names=names)

def from_anesthetic(dynesty_sampler, columns=None, tex=None, limits=None):
    """Convert from dynesty to anesthetic samples.

    Parameters
    ----------
    dynesty_sampler: dynesty.dynesty.NestedSampler or dynesty.results.Results
        dynesty NestedSampler instance to copy results from
        (must have been run already), or results of a dynesty run.

    columns: list, optional
        List of (python) parameter names
        default: None

    tex: list, optional
        List of LaTeX parameter names
        default: None

    limits: list, optional
        List of parameter limits
        default: None

    Returns
    -------
    nested_samples: NestedSamples
        anesthetic nested samples
    """
    if isinstance(dynesty_sampler, dynesty.dynesty.NestedSampler):
        dynesty_results = dynesty_sampler.results
    elif isinstance(dynesty_sampler, dynesty.results.Results):
        dynesty_results = dynesty_sampler
    else:
        raise ValueError, "Unknown dynesty input type"

    data = dynesty_results['samples']
    weights = np.exp(dynesty_results['logwt'])
    logl = dynesty_results['logl']
    #dynesty_results['logz']
    #logL_birth
    #label
    #beta
    #logzero
    return NestedSamples(data=data,
                         weights=weights,
                         logL=logl,
                         columns=coolumns,
                         tex=tex,
                         limits=limits)

def from_emcee(emcee_sampler, columns=None, tex=None, limits=None):
    """Convert from emcee to anesthetic samples.

    Parameters
    ----------
    emcee_sampler: emcee.ensemble.EnsembleSampler
        emcee sampler to copy results from

    columns: list, optional
        List of (python) parameter names
        default: None

    tex: list, optional
        List of LaTeX parameter names
        default: None

    limits: list, optional
        List of parameter limits
        default: None

    Returns
    -------
    mcmc_samples: MCMCSamples
        anesthetic MCMC samples
    """
    if not isinstance(emcee_sampler, emcee.ensemble.EnsembleSampler):
        raise ValueError, "Wrong input type, please pass\
                            emcee.ensemble.EnsembleSampler"

    data = emcee_sampler.flatchain
    #weights = 1
    #logL
    #label
    #logzero
    #burn_in
    return MCMCSamples(data=data,
                       columns=columns,
                       tex=tex,
                       limits=limits)
