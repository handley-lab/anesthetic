import numpy
import pandas
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def load_nested_samples(root):
    return NestedSamplingKDE.read(root)


def plot_1d(data, weights, ax=None, colorscheme=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    npoints = 1000
    i = numpy.random.choice(len(weights), size=npoints, replace=True, p=weights)
    kde = gaussian_kde(data[i], weights=weights[i])
    mean = numpy.average(data, weights=weights)
    x = numpy.linspace(data[i].min(), data[i].max(), 100)
    kde_max = numpy.exp(-minimize(lambda x: -kde.logpdf(x),mean,method='Nelder-Mead')['fun'])
    return ax.plot(x, kde(x)/kde_max, color=colorscheme, *args, **kwargs)

convert={'r':'Reds', 'b':'Blues', 'y':'Yellows', 'g':'Greens', 'k':'Greys'}

def contour_plot_2d(data_x, data_y, weights, ax=None, colorscheme='b', *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    npoints = 1000
    i = numpy.random.choice(len(weights), size=npoints, replace=True, p=weights)
    data = numpy.array([data_x, data_y])
    kde = gaussian_kde(data[:,i])

    p = sorted(kde(data[:,i]))
    m = numpy.arange(len(p))/len(p)

    interp = interp1d([0]+list(p)+[numpy.inf],[0]+list(m)+[1])

    x = numpy.linspace(data_x[i].min(), data_x[i].max(), 100)
    y = numpy.linspace(data_y[i].min(), data_y[i].max(), 100)
    xx, yy = numpy.meshgrid(x, y)
    positions = numpy.vstack([xx.ravel(), yy.ravel()])
    f = numpy.reshape(kde(positions).T, xx.shape)
    f = interp(f)
    cbar = ax.contour(x, y, f, [0.05, 0.33, 1], vmin=0,vmax=1, linewidths=0.5, colors='k', *args, **kwargs)  
    cbar = ax.contourf(x, y, f, [0.05, 0.33, 1], vmin=0,vmax=1, cmap=plt.cm.get_cmap(convert[colorscheme]), *args, **kwargs)  
    return cbar


def scatter_plot_2d(data_x, data_y, weights, ax=None, colorscheme=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    w = weights / weights.max()
    if w.sum() > 100:
        w *= 100/w.sum()
    i = w > numpy.random.rand(len(w))
    return ax.scatter(data_x[i], data_y[i], c=colorscheme, *args, **kwargs)


class NestedSamplingKDE(pandas.DataFrame):
    _metadata = ['paramnames', 'tex']

    @classmethod
    def read(cls, root):
        # Get paramnames
        paramnames = []
        tex = {}
        paramnames_file = root + '.paramnames'
        for line in open(paramnames_file, 'r'):
            line = line.strip().split()
            paramname = line[0].replace('*', '')
            paramnames.append(paramname)
            tex[paramname] = ''.join(line[1:])

        # Get data
        birth_file = root + '_dead-birth.txt'
        data = numpy.loadtxt(birth_file)
        columns = paramnames + ['logL', 'logL_birth']
        data = cls(data=data, columns=columns)
        data.tex = tex
        data.paramnames = paramnames

        # Compute nlive
        index = data.logL.searchsorted(data.logL_birth)-1
        births = pandas.Series(+1, index=index).sort_index()
        deaths = pandas.Series(-1, index=data.index)
        nlive = pandas.concat([births, deaths]).sort_index().cumsum()
        nlive = (nlive[~nlive.index.duplicated(keep='first')]+1)[1:]
        data['nlive'] = nlive

        # Compute weights
        dlogX = data.dlogX(-1)
        data['posterior_weights'] = data.logL+dlogX
        data['posterior_weights'] -= logsumexp(data.posterior_weights)
        data['posterior_weights'] = numpy.exp(data.posterior_weights)
        data['prior_weights'] = dlogX
        data['prior_weights'] -= logsumexp(data.prior_weights)
        data['prior_weights'] = numpy.exp(data.prior_weights)
        return data

    def dlogX(self, nsamples=100):
        if nsamples < 0:
            t = numpy.atleast_2d(numpy.log(self.nlive/(self.nlive+1)))
            nsamples=1
        else:
            t = numpy.log(numpy.random.rand(nsamples, len(self))
                          )/self.nlive.values
        logX = numpy.concatenate((numpy.zeros((nsamples, 1)),
                                  t.cumsum(axis=1),
                                  -numpy.inf*numpy.ones((nsamples, 1))
                                  ), axis=1)
        dlogX = logsumexp([logX[:, :-2], logX[:, 2:]],
                          b=[numpy.ones_like(t), -numpy.ones_like(t)], axis=0)
        dlogX -= numpy.log(2)
        return numpy.squeeze(dlogX)

    def infer(self, nsamples=100):
        columns = ['logZ', 'D', 'd']
        data = pandas.DataFrame(numpy.zeros((nsamples, len(columns))),
                                columns=columns)
        dlogX = self.dlogX(nsamples)

        data.logZ = logsumexp(self.logL.values + dlogX, axis=1)
        logw = ((self.logL.values + dlogX).T - data.logZ.values).T
        Info = ((self.logL.values + numpy.zeros_like(dlogX)).T
                - data.logZ.values).T

        data.D = numpy.exp(logsumexp(logw, b=Info, axis=1))
        data.d = numpy.exp(logsumexp(logw, b=(Info.T-data.D.values).T**2,
                                     axis=1))*2
        return data

    def w(self):
        dlogX = self.dlogX(1)
        logZ = logsumexp(self.logL + dlogX)
        return numpy.exp(self.logL + dlogX - logZ)

    def weights(self, prior=False):
        if prior:
            return self.prior_weights
        else:
            return self.posterior_weights

    def plot(self, paramname_x, paramname_y=None, ax=None, colorscheme='b',
             prior=False, kind='contour', *args, **kwargs):
        if paramname_y is None or paramname_x == paramname_y:
            return plot_1d(self[paramname_x], self.weights(prior), ax, colorscheme,
                           *args, **kwargs)

        if kind == 'contour':
            plot = contour_plot_2d
        elif kind == 'scatter':
            plot = scatter_plot_2d

        return plot(self[paramname_x], self[paramname_y], self.weights(prior),
                    ax, colorscheme, *args, **kwargs)

    def cov(self, paramnames, prior=False):
        return pandas.DataFrame(numpy.cov(self[paramnames].T,
                                          aweights=self.weights(prior)),
                                columns=paramnames, index=paramnames)
