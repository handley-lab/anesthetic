import numpy 
from anesthetic import MCMCSamples, NestedSamples

def loglikelihood(x):
    """Example non-trivial loglikelihood

    - Constrained zero-centered correlated parameters x0 and x1,
    - half-constrained x2 (exponential).
    - unconstrained x3 between 0 and 1
    - x4 is a slanted top-hat distribution between 2 and 4

    """
   
    x0, x1, x2, x3, x4 = x[:]
    if x2 < 0 or x3 > 1 or x3 < 0 or x4 < 2 or x4 > 4:
        return -numpy.inf
    eps = 0.9
    sigma0, sigma1, sigma2 = 0.1, 0.1, 0.1
    mu0, mu1 = 0, 0
    x0 /= sigma0
    x1 /= sigma1

    logl = 0
    logl -= numpy.log(2*numpy.pi*sigma0*sigma1*(1-eps**2)**0.5)
    logl -= (x0**2 - 2*eps*x0*x1 + x1**2)/(1-eps**2)/2

    logl -= numpy.log(sigma2) 
    logl -= x2/sigma2

    logl -= numpy.log(3)
    logl += numpy.log(x4-1)
    return logl


ndims = 5
columns = ['x%i' % i for i in range(ndims)]
tex = {p: '$x_%i$' % i  for i, p in enumerate(columns)}
roots = []

# MCMC
# ----
def mcmc_sim(ndims=5):
    """ Simple Metropolis Hastings algorithm. """
    numpy.random.seed(0)
    x = [numpy.array([0, 0, 0.1, 0.5, 3])]
    l = [loglikelihood(x[-1])]
    w = [1]

    while len(x) < 10000:
        x1 = x[-1] + numpy.random.randn(ndims)*0.1
        l1 = loglikelihood(x1)
        if numpy.random.rand() < numpy.exp(l1-l[-1]):
            x.append(x1)
            l.append(l1)
            w.append(1)
        else:
            w[-1]+=1
    return numpy.array(x), numpy.array(l), numpy.array(w)


data, logL, w = mcmc_sim()
mcmc = MCMCSamples(data=data, columns=columns, logL=logL, w=w, tex=tex)


# MCMC multiple files
root = './tests/example_data/gd'
roots.append(root)
mcmc[['weight', 'logL'] + columns][:len(mcmc)//2].to_csv(root + '_1.txt', sep=' ', index=False, header=False)
mcmc[['weight', 'logL'] + columns][len(mcmc)//2:].to_csv(root + '_2.txt', sep=' ', index=False, header=False)

# MCMC single file
root = './tests/example_data/gd_single'
roots.append(root)
mcmc[['weight', 'logL'] + columns].to_csv(root + '.txt', sep=' ', index=False, header=False)



# NS
# --

def ns_sim(ndims=5, nlive=125):
    """Brute force Nested Sampling run"""
    numpy.random.seed(0)
    low=(-1,-1,0,0,2)
    high=(1,1,1,1,4)
    live_points = numpy.random.uniform(low=low, high=high, size=(nlive, ndims))
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
            live_points[i, :] = numpy.random.uniform(low=low, high=high, size=ndims) 
            live_likes[i] = loglikelihood(live_points[i])
    return dead_points, dead_likes, birth_likes, live_points, live_likes, live_birth_likes

data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim()

ns = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_ns = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)

# Dead file for polychord
root = './tests/example_data/pc'
roots.append(root)

ns[columns + ['logL', 'logL_birth']].to_csv(root + '_dead-birth.txt', sep=' ', index=False, header=False)

# Dead file for multinest
ns['cluster'] = 1
live_ns['cluster']=1
root = './tests/example_data/mn'
roots.append(root)
ns['dlogX'] = ns._dlogX()
ns[columns + ['logL', 'logL_birth', 'dlogX', 'cluster']].to_csv(root + 'dead-birth.txt', sep=' ', index=False, header=False)
live_ns[columns + ['logL', 'logL_birth', 'cluster']].to_csv(root + 'phys_live-birth.txt', sep=' ', index=False, header=False)
ns[columns + ['logL', 'dlogX', 'cluster']].to_csv(root + 'ev.dat', sep=' ', index=False, header=False)
live_ns[columns + ['logL', 'cluster']].to_csv(root + 'phys_live.points', sep=' ', index=False, header=False)


# Dead file for old multinest
root = './tests/example_data/mn_old'
roots.append(root)

ns[columns + ['logL', 'dlogX', 'cluster']].to_csv(root + 'ev.dat', sep=' ', index=False, header=False)
ns[columns + ['logL', 'cluster']].to_csv(root + 'phys_live.points', sep=' ', index=False, header=False)

for root in roots:
    # paramnames file
    with open(root + '.paramnames', 'w') as f:
        for p in columns:
            f.write('%s\t%s\n' % (p, tex[p]))

    # ranges file
    with open(root + '.ranges', 'w') as f:
        f.write('%s\tNone\tNone\n' % columns[0])
        f.write('%s\tNone\tNone\n' % columns[1])
        f.write('%s\t0\tNone\n' % columns[2])
        f.write('%s\t0\t1\n' % columns[3])

