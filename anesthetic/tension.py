"""Tension statistics between two datasets."""
from anesthetic.samples import Samples
from scipy.stats import chi2


def tension_stats(A, B, AB, nsamples=None, beta=None):
    r"""Compute tension statistics between two samples.

    Using nested sampling we can compute:

    - ``logZ``: Bayesian evidence

        .. math::
            \log Z = \log \int L \pi d\theta

    - ``D_KL``: Kullback--Leibler divergence

        .. math::
            D_\mathrm{KL} = \int \mathcal{P} \log(\mathcal{P} / \pi) d\theta

    - ``logL_P``: posterior averaged log-likelihood

        .. math::
            \langle\log L\rangle_\mathcal{P} = \int \mathcal{P} \log L d\theta

    - ``d_G``: Gaussian model dimensionality
        (or posterior variance of the log-likelihood)

        .. math::
            d_\mathrm{G}/2 = \mathrm{var}(\log L)_\mathcal{P}

    - ``p``: p-value for the tension between two samples

        .. math::
            p = \int_{d_\mathrm{G} - 2 \log S}^{\infty} \chi^2 (x)dx

    Parameters
    ----------
    A : :class:`Samples`/:class:`NestedSamples`
        Dataset A
        List or array-like of one or more nested sampling runs

    B : :class:`Samples`/:class:`NestedSamples`
        Dataset B
        List or array-like of one or more nested sampling runs

    AB : :class:`Samples`/:class:`NestedSamples`
        Dataset AB
        List or array-like of one or more nested sampling runs

    nsamples : int, optional
        - If nsamples is not supplied, calculate mean value
        - If nsamples is integer, draw nsamples from the distribution of
            values inferred by nested sampling

    beta : float, array-like, optional
        inverse temperature(s) beta=1/kT. Default self.beta

    Returns
    -------
    samples_stats : :class:`pandas.DataFrame`
        DataFrame containing the following tension statistics:
        logZ, D_KL, logL_P, d_G, p
    """
    statsA = A.stats(nsamples, beta)
    statsB = B.stats(nsamples, beta)
    statsAB = AB.stats(nsamples, beta)

    samples_stats = Samples(index=statsA.index)

    logR = statsAB.logZ-statsA.logZ-statsB.logZ
    samples_stats['logR'] = logR
    samples_stats.set_label('logR', r'$\log{R}$')

    logI = statsA.D_KL + statsB.D_KL - statsAB.D_KL
    samples_stats['logI'] = logI
    samples_stats.set_label('logI', r'$\log{I}$')

    logS = logR-logI
    samples_stats['logS'] = logS
    samples_stats.set_label('logS', r'$\log{S}$')

    d_G = statsA.d_G + statsB.d_G - statsAB.d_G
    samples_stats['d_G'] = d_G
    samples_stats.set_label('d_G', r'$d_\mathrm{G}$')

    p = chi2.sf(d_G-2*logS, d_G)
    samples_stats['p'] = p
    samples_stats.set_label('p', r'$p$')
    return samples_stats
