from anesthetic.samples import Samples


def tension_stats(A,B,AB,nsamples,beta):
    # Compute Nested Sampling statistics
    statsA = A.stats(nsamples,beta)
    statsB = B.stats(nsamples,beta)
    statsAB = AB.stats(nsamples,beta)

    samples = Samples(index=statsA.index)
    logR = statsAB.logZ-statsA.logZ-statsB.logZ
    samples['logR']=logR
    logI  = statsA.D_KL + statsB.D_KL - statsAB.D_KL
    samples['logI']=logI
    samples['logS']=logR-logI
    samples['d_G']=statsA.d_G + statsB.d_G - statsAB.d_G
    return samples