from anesthetic.samples import Samples
from scipy.stats import chi2


def tension_stats(A, B, AB, nsamples, beta):
    # Compute Nested Sampling statistics
    statsA = A.stats(nsamples, beta)
    statsB = B.stats(nsamples, beta)
    statsAB = AB.stats(nsamples, beta)

    samples = Samples(index=statsA.index)
    print(samples)
    logR = statsAB.logZ-statsA.logZ-statsB.logZ
    samples['logR'] = logR

    logI = statsA.D_KL + statsB.D_KL - statsAB.D_KL
    samples['logI'] = logI

    logS = logR-logI
    samples['logS'] = logS

    d_G = statsA.d_G + statsB.d_G - statsAB.d_G
    samples['d_G'] = d_G

    p = chi2.sf(d_G-2*logS, d_G)
    samples['p'] = p
    return samples
