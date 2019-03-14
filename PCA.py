%load_ext autoreload
%autoreload 2

import numpy
import matplotlib.pyplot as plt
from nskde.kde import load_nested_samples
import pandas
import scipy.linalg

samples = load_nested_samples('/data/will/data/pablo/runs/chains/planck')
samples = load_nested_samples('/data/will/data/pablo/runs/chains/DES')

samples['h'] = samples['H0']/100
samples.tex['h'] = 'h'
samples['omegab'] = samples['omegabh2']/samples['h']**2
samples.tex['omegab'] = '\Omega_b'
samples['omegac'] = samples['omegach2']/samples['h']**2
samples.tex['omegac'] = '\Omega_c'

samples['log_omegac'] = numpy.log(samples['omegac'])
samples.tex['log_omegac'] = '\log \Omega_c'
samples['log_omegab'] = numpy.log(samples['omegab'])
samples.tex['log_omegab'] = '\log \Omega_b'

samples['log_omegam'] = numpy.log(samples['omegam'])
samples.tex['log_omegam'] = '\log \Omega_m'
samples['log_sigma8'] = numpy.log(samples['sigma8'])
samples.tex['log_sigma8'] = '\log \sigma_8'
samples['log_H0'] = numpy.log(samples['H0'])
samples.tex['log_H0'] = '\log H_0'
samples['log_ns'] = numpy.log(samples['ns'])
samples.tex['log_ns'] = '\log n_s'


kwargs={}
def make_axes(paramnames, **kwargs):
    paramnames_x = paramnames
    paramnames_y = kwargs.pop('paramnames_y', paramnames)
    tex = kwargs.pop('tex', {p:p for p in paramnames})

    n_x = len(paramnames_x)
    n_y = len(paramnames_y)
    fig, axes = plt.subplots(n_x, n_y, sharex='col', sharey='row', gridspec_kw={'wspace':0, 'hspace':0})

    for ax in axes[0,0].get_shared_x_axes():
        print(ax)

    print(axes[0,0])

    for p_y, ax in zip(paramnames_y, axes[:,0]):
        ax.set_ylabel('$%s$' % tex[p_y])

    for p_x, ax in zip(paramnames_x, axes[-1,:]):
        ax.set_xlabel('$%s$' % tex[p_x])

    # Unshare any 1D axes
    for j, (p_j, row) in enumerate(zip(paramnames, axes)):
        for i, (p_i, ax) in enumerate(zip(paramnames, row)):
            if p_i == p_j:
                axes[i,j] = ax.twinx()
                axes[i,j].set_yticks([])

    return fig, axes


def compute_eigenvectors(cov):
    eigenvalues, eigenvectors = scipy.linalg.eig(cov)
    eigenvalues = numpy.real(eigenvalues)
    i = numpy.argsort(eigenvalues)
    eigenvalues = eigenvalues[i]
    eigenvectors = eigenvectors[:,i]
    eigenvectors = [pandas.Series(eigenvectors[:,i], index=cov.index) for i, _ in enumerate(eigenvalues)]
    return eigenvalues, eigenvectors



paramnames = ['log_omegab', 'log_omegac', 'log_H0', 'tau', 'logA', 'ns']
#paramnames = ['tau','logA']
fig, axes = make_axes(paramnames, tex=samples.tex)

for j, (p_j, row) in enumerate(zip(paramnames, axes)):
    for i, (p_i, ax) in enumerate(zip(paramnames, row)):
        print(p_i, p_j)
        if i > j:
            kind='scatter'
        else:
            kind='contour'
        #for prior, color in [(True,'r'), (False, 'b')]:
        for prior, color in [(False, 'b')]:
            samples.plot(p_i, p_j, ax, prior=prior, kind=kind, colorscheme=color)

paramnames = ['omegab', 'omegac', 'H0', 'tau', 'logA', 'ns']
trans = pandas.DataFrame(numpy.diag(numpy.diagonal(samples.cov(paramnames, prior=True))**-1),index=paramnames,columns=paramnames)
cov = trans @ samples.cov(paramnames)

ds = []
for _ in range(1000):
    weights = samples.w()
    weights /= weights.sum()
    entropy = weights*numpy.log(weights) 
    N = numpy.exp(-entropy[~numpy.isnan(entropy)].sum())
    n = 1000
    ds.append(N**2/(numpy.pi*numpy.e*n**2))
weights = samples.posterior_weights
plt.plot(numpy.cumsum(sorted(weights)))
numpy.random.choice(len(weights), size=1000, replace=False, p=weights)
plt.yscale('log')

plt.hist(ds)
weights
N**2/(numpy.pi*numpy.e*1000**2)


eigenvalues, eigenvectors = compute_eigenvectors(cov)
eigenvectors[-2]
eigenvalues

M = numpy.diag(eigenvalues**-0.5) @ numpy.array(eigenvectors)
Minv = numpy.array(eigenvectors).T  @ numpy.diag(eigenvalues**0.5)
M @ cov @ M.T

Minv @ Minv.T - cov

cov = samples.cov(paramnames, prior=False)
cov = M @ cov @ M.T

eigenvalues, eigenvectors = compute_eigenvectors(cov)
eigenvalues
pandas.Series(Minv.T.dot(eigenvectors[-1]),index=paramnames)



eigenvalues, eigenvectors = compute_eigenvectors(samples, ['tau', 'logA'])
eigenvalues
eigenvectors[0]

eigenvalues, eigenvectors = compute_eigenvectors(samples, paramnames)
eigenvalues
eigenvectors[0]
eigenvectors[1]
eigenvectors[2]
eigenvectors[3]
eigenvectors[5]


eigenvalues
eigenvectors[3] / eigenvectors[3]['log_omegac']
eigenvectors[5]
emax = eigenvectors[numpy.argmin(eigenvalues)]
emax/emax['log_omegac']

#(cov @ eigenvectors[1]) / eigenvectors[1] / eigenvalues[1]
