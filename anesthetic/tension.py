"""Tension statistics between two datasets."""
from anesthetic.samples import Samples
from scipy.stats import chi2


def tension_stats(A, B, AB, nsamples=None, beta=None):  # noqa: D301
    """Compute tension statistics between two samples.

    Using nested sampling we can compute:

    - ``logR``: Logarithmic of R statistic

      .. math::
        \\log R = \\log Z_\\mathrm{AB} - \\log Z_\\mathrm{A}
                     - \\log Z_\\mathrm{B}

    - ``logI``: Logarithmic of information ratio

      .. math::
        \\log I = D_\\mathrm{KL}^A + D_\\mathrm{KL}^B - D_\\mathrm{KL}^{AB}

    - ``logS``: Logarithmic of suspiciousness

      .. math::
        \\log S = \\log L_\\mathrm{AB} - \\log
        L_\\mathrm{A} - \\log L_\\mathrm{B}

    - ``d_G``: Gaussian model dimensionality
      (or posterior variance of the log-likelihood)

      .. math::
        d_\\mathrm{G}/2 = \\mathrm{var}(\\log L)_P

    - ``p``: p-value for the tension between two samples

      .. math::
        p = \\int_{d_\\mathrm{G} - 2 \\log S}^{\\infty} \\chi^2 (x)dx

    Parameters
    ----------
    A : :class:`anesthetic.samples.Samples` or \
        :class:`anesthetic.samples.NestedSamples`
        (Nested) Samples from a sampling run using only dataset A.

    B : :class:`anesthetic.samples.Samples` or \
        :class:`anesthetic.samples.NestedSamples`
        (Nested) Samples from a sampling run using only dataset B.

    AB : :class:`anesthetic.samples.Samples` or \
         :class:`anesthetic.samples.NestedSamples`
        (Nested) Samples from a sampling run using datasets A and B jointly.

    nsamples : int, optional
        - If nsamples is not supplied, calculate mean value
        - If nsamples is integer, draw nsamples from the distribution of
          values inferred by nested sampling

    beta : float, array-like, optional
        inverse temperature(s) beta=1/kT. Default 1

    Returns
    -------
    samples_stats : :class:`anesthetic.samples.Samples`
        DataFrame containing the following tension statistics:
        logR, logI, logS, d_G, p
    """
    statsA = A.stats(nsamples=nsamples, beta=beta)
    statsB = B.stats(nsamples=nsamples, beta=beta)
    statsAB = AB.stats(nsamples=nsamples, beta=beta)

    samples_stats = Samples(index=statsA.index)

    logR = statsAB.logZ-statsA.logZ-statsB.logZ
    samples_stats['logR'] = logR
    samples_stats.set_label('logR', r'$\log{R}$')

    logI = statsA.D_KL + statsB.D_KL - statsAB.D_KL
    samples_stats['logI'] = logI
    samples_stats.set_label('logI', r'$\log{I}$')

    logS = statsAB.logL_P - statsA.logL_P - statsB.logL_P
    samples_stats['logS'] = logS
    samples_stats.set_label('logS', r'$\log{S}$')

    d_G = statsA.d_G + statsB.d_G - statsAB.d_G
    samples_stats['d_G'] = d_G
    samples_stats.set_label('d_G', r'$d_\mathrm{G}$')

    p = chi2.sf(d_G-2*logS, d_G)
    samples_stats['p'] = p
    samples_stats.set_label('p', r'$p$')
    return samples_stats
