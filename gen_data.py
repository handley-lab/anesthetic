import numpy 
from anesthetic.anesthetic import MCMCSamples, NestedSamples

def loglikelihood(x):
    sigma = 0.1
    return -(x-0.5) @ (x-0.5) / 2 / sigma**2


# MCMC
# ----
def mcmc_sim(ndims=4):
    """ Simple Metropolis Hastings algorithm. """
    numpy.random.seed(0)
    x = [numpy.random.normal(0.5, 0.1, ndims)]
    l = [loglikelihood(x[-1])]
    w = [1]

    for _ in range(10000):
        x1 = x[-1] + numpy.random.randn(ndims)*0.1
        l1 = loglikelihood(x1)
        if numpy.random.rand() < numpy.exp(l1-l[-1]):
            x.append(x1)
            l.append(l1)
            w.append(1)
        else:
            w[-1]+=1
    return x, l, w


params, logL, w = mcmc_sim()
mcmc = MCMCSamples.build(params=params, logL=logL, w=w)

# Dead file
root = './tests/example_data/mcmc/mcmc'
mcmc[['w', 'logL'] + mcmc.paramnames].to_csv(root + '_1.txt', sep=' ', index=False, header=False)

# paramnames file
with open(root + '.paramnames', 'w') as f:
    for i in range(len(mcmc.paramnames)):
        f.write('x%i\tx_%i\n' % (i, i))

# NS
# --

def ns_sim(ndims=4, nlive=50):
    """Brute force Nested Sampling run"""
    numpy.random.seed(0)
    live_points = numpy.random.rand(nlive, ndims)
    live_likes = numpy.array([loglikelihood(x) for x in live_points])
    live_birth_likes = numpy.ones(nlive) * -numpy.inf

    dead_points = []
    dead_likes = []
    birth_likes = []
    for _ in range(nlive*9):
        i = numpy.argmin(live_likes)
        Lmin = live_likes[i]
        dead_points.append(live_points[i].copy())
        dead_likes.append(live_likes[i])
        birth_likes.append(live_birth_likes[i])
        live_birth_likes[i] = Lmin
        while live_likes[i] <= Lmin:
            live_points[i, :] = numpy.random.rand(ndims)
            live_likes[i] = loglikelihood(live_points[i])
    return dead_points, dead_likes, birth_likes
params, logL, logL_birth = ns_sim()
ns = NestedSamples.build(params=params, logL=logL, logL_birth=logL_birth)

# Dead file
root = './tests/example_data/ns/ns'
ns[ns.paramnames + ['logL', 'logL_birth']].to_csv(root + '_dead-birth.txt', sep=' ', index=False, header=False)

# paramnames file
with open(root + '.paramnames', 'w') as f:
    for i in range(len(ns.paramnames)):
        f.write('x%i\tx_%i\n' % (i, i))

