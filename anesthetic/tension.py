"""Tension statistics between two datasets."""
from anesthetic.samples import Samples
from scipy.stats import chi2


def stats(A, B, AB, nsamples=None, beta=None):  # noqa: D301
    r"""Compute tension statistics between two samples.

    Using nested sampling we can compute:

    - ``logR``: R statistic for dataset consistency

      .. math::
        \log R = \log Z_{AB} - \log Z_{A} - \log Z_{B}

    - ``logI``: information ratio

      .. math::
        \log I = D_{KL}^{A} + D_{KL}^{B} - D_{KL}^{AB}

    - ``logS``: suspiciousness

      .. math::
        \log S = \log L_{AB} - \log L_{A} - \log L_{B}

    - ``d_G``: Gaussian model dimensionality of shared constrained parameters

      .. math::
        d = d_{A} + d_{B} - d_{AB}

    - ``p``: p-value for the tension between two samples

      .. math::
        p = \int_{d-2\log{S}}^{\infty} \chi^2_d(x) dx

    Parameters
    ----------
    A : :class:`anesthetic.samples.NestedSamples`
        NestedSamples object from a sampling run using only dataset A.
        Alternatively, you can pass the precomputed stats object returned from
        :meth:`anesthetic.samples.NestedSamples.stats`.

    B : :class:`anesthetic.samples.NestedSamples`
        NestedSamples object from a sampling run using only dataset B.
        Alternatively, you can pass the precomputed stats object returned from
        :meth:`anesthetic.samples.NestedSamples.stats`.

    AB : :class:`anesthetic.samples.NestedSamples`
        NestedSamples object from a sampling run using both datasets A and B
        jointly.

    nsamples : int, optional
        - If nsamples is not supplied, calculate mean value
        - If nsamples is integer, draw nsamples from the distribution of
          values inferred by nested sampling

    beta : float, array-like, default=1
        Inverse temperature(s) beta=1/kT.

    Returns
    -------
    samples : :class:`anesthetic.samples.Samples`
        DataFrame containing the following tension statistics in columns:
        ['logR', 'logI', 'logS', 'd_G', 'p']
    """
    statsA = A.stats(nsamples=nsamples, beta=beta)
    statsB = B.stats(nsamples=nsamples, beta=beta)
    statsAB = AB.stats(nsamples=nsamples, beta=beta)

    samples = Samples(index=statsA.index)

    samples['logR'] = statsAB['logZ'] - statsA['logZ'] - statsB['logZ']
    samples.set_label('logR', r'$\ln\mathcal{R}$')

    samples['logI'] = statsA['D_KL'] + statsB['D_KL'] - statsAB['D_KL']
    samples.set_label('logI', r'$\ln\mathcal{I}$')

    samples['logS'] = statsAB['logL_P'] - statsA['logL_P'] - statsB['logL_P']
    samples.set_label('logS', r'$\ln\mathcal{S}$')

    samples['d_G'] = statsA['d_G'] + statsB['d_G'] - statsAB['d_G']
    samples.set_label('d_G', r'$d_\mathrm{G}$')

    p = chi2.sf(samples['d_G'] - 2 * samples['logS'], df=samples['d_G'])
    samples['p'] = p
    samples.set_label('p', '$p$')
    return samples
