"""Tools for converting to other outputs."""
from anesthetic.samples import NestedSamples, MCMCSamples
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
    import getdist
    labels = np.char.strip(samples.get_labels().astype(str), '$')
    samples = samples.drop_labels()
    ranges = samples.agg(['min', 'max']).T.apply(tuple, axis=1).to_dict()
    return getdist.mcsamples.MCSamples(samples=samples.to_numpy(),
                                       weights=samples.get_weights(),
                                       loglikes=-samples.logL.to_numpy(),
                                       names=samples.columns,
                                       ranges=ranges,
                                       labels=labels)


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
    import dynesty.results
    import dynesty.dynesty

    if isinstance(dynesty_sampler, dynesty.dynesty.NestedSampler):
        dynesty_results = dynesty_sampler.results
    elif isinstance(dynesty_sampler, dynesty.results.Results):
        dynesty_results = dynesty_sampler
    else:
        raise ValueError("Unknown dynesty input type")

    data = dynesty_results['samples']
    weights = np.exp(dynesty_results['logwt'])
    logl = dynesty_results['logl']
    # dynesty_results['logz']
    # logL_birth
    # label
    # beta
    # logzero
    return NestedSamples(data=data,
                         weights=weights,
                         logL=logl,
                         columns=columns,
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
    import emcee.ensemble
    if not isinstance(emcee_sampler, emcee.ensemble.EnsembleSampler):
        raise ValueError("Wrong input type, please pass\
                          emcee.ensemble.EnsembleSampler")

    data = emcee_sampler.flatchain
    # weights = 1
    # logL
    # label
    # logzero
    # burn_in
    return MCMCSamples(data=data,
                       columns=columns,
                       tex=tex,
                       limits=limits)
