"""Tension statistics between two or more datasets."""

import numpy as np
from scipy.stats import chi2
from scipy.special import erfcinv
from anesthetic.samples import Samples


def tension_stats(joint, *separate):
    r"""Compute tension statistics between two or more samples.

    With the Bayesian (log-)evidence ``logZ``, Kullback--Leibler divergence
    ``D_KL``, posterior average of the log-likelihood ``logL_P``, Gaussian
    model dimensionality ``d_G``, we can compute tension statistics between
    two or more samples (example here for simplicity just with two datasets
    A and B):

    - ``logR``: R statistic for dataset consistency

      .. math::
        \log R = \log Z_{AB} - \log Z_{A} - \log Z_{B}

    - ``I``: information ratio

      .. math::
        I = D_{KL}^{A} + D_{KL}^{B} - D_{KL}^{AB}

    - ``logS``: suspiciousness

      .. math::
        \log S = \log L_{AB} - \log L_{A} - \log L_{B}

    - ``d_G``: Gaussian model dimensionality of shared constrained parameters

      .. math::
        d = d_{A} + d_{B} - d_{AB}

    - ``p``: p-value for the tension between two samples

      .. math::
        p = \int_{d-2\log{S}}^{\infty} \chi^2_d(x) dx

    - ``tension``: tension quantification in terms of numbers of sigma
      calculated from p

      .. math::
        \sqrt{2} \rm{erfc}^{-1}(p)

    Parameters
    ----------
    joint : :class:`anesthetic.samples.Samples`
        Bayesian stats from a nested sampling run using all the datasets from
        the list in ``separate`` jointly. This should be a ``stats`` object
        with columns ['logZ', 'D_KL', 'logL_P', 'd_G'] as returned by
        :meth:`anesthetic.samples.NestedSamples.stats`.

    *separate
        A variable number of Bayesian stats from independent nested sampling
        runs using various datasets (A, B, ...) separately. Each should be a
        ``stats`` object with the columns ['logZ', 'D_KL', 'logL_P', 'd_G']
        as returned by :meth:`anesthetic.samples.NestedSamples.stats`.

    Returns
    -------
    samples : :class:`anesthetic.samples.Samples`
        DataFrame containing the following tension statistics in columns:
        ['logR', 'I', 'logS', 'd_G', 'p', 'tension']
    """
    columns = ["logZ", "D_KL", "logL_P", "d_G"]
    if not set(columns).issubset(joint.drop_labels().columns):
        raise ValueError(
            "The DataFrame passed to `joint` needs to contain"
            "the columns 'logZ', 'D_KL', 'logL_P', and 'd_G'."
        )
    for s in separate:
        if not set(columns).issubset(s.drop_labels().columns):
            raise ValueError(
                "The DataFrames passed to `separate` need to "
                "contain the columns 'logZ', 'D_KL', 'logL_P', "
                "and 'd_G'."
            )
    separate_stats = separate[0][columns].copy()
    for s in separate[1:]:
        separate_stats += s
    joint_stats = joint[columns].copy()

    samples = Samples(index=joint_stats.index)

    samples["logR"] = joint_stats["logZ"] - separate_stats["logZ"]
    samples.set_label("logR", r"$\ln\mathcal{R}$")

    samples["I"] = separate_stats["D_KL"] - joint_stats["D_KL"]
    samples.set_label("I", r"$\mathcal{I}$")

    samples["logS"] = joint_stats["logL_P"] - separate_stats["logL_P"]
    samples.set_label("logS", r"$\ln\mathcal{S}$")

    samples["d_G"] = separate_stats["d_G"] - joint_stats["d_G"]
    samples.set_label("d_G", r"$d_\mathrm{G}$")

    p = chi2.sf(samples["d_G"] - 2 * samples["logS"], df=samples["d_G"])
    samples["p"] = p
    samples.set_label("p", "$p$")

    samples["tension"] = erfcinv(p) * np.sqrt(2)
    samples.set_label("tension", r"tension~[$\sigma$]")

    return samples
