import numpy as np
from ultranest import ReactiveNestedSampler

ndim = 4
sigma = np.logspace(-1, -6, ndim)
width = 1 - 5 * sigma
width[width < 1e-20] = 1e-20
centers = (np.sin(np.arange(ndim) / 2.) * width + 1.) / 2.


def loglike(theta):
    """compute log-likelihood."""
    like = - 0.5 * (((theta - centers) / sigma)**2).sum(axis=1) \
           - 0.5 * np.log(2 * np.pi * sigma**2).sum()
    return like


def transform(x):
    """transform according to prior."""
    return x * 20 - 10


paramnames = ['a', 'b', 'c', 'd']

sampler = ReactiveNestedSampler(
    paramnames, loglike, transform=transform,
    log_dir='un', resume=True, vectorized=True)

sampler.run(frac_remain=0.5, min_num_live_points=400)
sampler.print_results()
sampler.plot()
