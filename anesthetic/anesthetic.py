import numpy
import pandas
from scipy.special import logsumexp
from anesthetic.plot import make_1D_axes, make_2D_axes, plot_1d, scatter_plot_2d, contour_plot_2d

def load_nested_samples(root):
    return NestedSamplingKDE.read(root)


class NestedSamplingKDE(pandas.DataFrame):
    _metadata = ['paramnames', 'tex', 'prior_range']

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

        weights = data['posterior_weights']
        with numpy.errstate(divide='ignore'):
            cc = int(numpy.exp((weights * -numpy.log(weights)).sum()))
        frac, iw = numpy.modf(weights*cc)
        data['posterior_iweights'] = (iw + (numpy.random.rand(len(frac))<frac)).astype(int)

        weights = data['prior_weights']
        with numpy.errstate(divide='ignore'):
            cc = int(numpy.exp((weights * -numpy.log(weights)).sum()))
        frac, iw = numpy.modf(weights*cc)
        data['prior_iweights'] = (iw + (numpy.random.rand(len(frac))<frac)).astype(int)

        data.prior_range = {}

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

    def weights(self, integer=True, prior=False):
        if prior:
            if integer:
                return self.prior_iweights
            else:
                return self.prior_weights
        else:
            if integer:
                return self.posterior_iweights
            else:
                return self.posterior_weights

    def plot_1d(self, paramnames, axes=None, prior=None, color='b'):
        if isinstance(paramnames, str):
            paramnames = [paramnames]

        if axes is None:
            fig, axes = make_1D_axes(paramnames, self.tex)
        else:
            fig = axes[0,0].figure

        for p, ax in zip(paramnames, axes.flatten()):
            self.plot(p, ax=ax, prior=prior, colorscheme=color)

        return fig, axes

    def plot_2d(self, paramnames, paramnames_y=None, axes=None, prior=None, color='b'):
        if isinstance(paramnames, str):
            paramnames = [paramnames]
        if isinstance(paramnames_y, str):
            paramnames_y = [paramnames_y]

        if axes is None:
            fig, axes = make_2D_axes(paramnames, paramnames_y, self.tex)
        else:
            fig = axes[0,0].figure

        paramnames_x = paramnames
        if paramnames_y is None:
            paramnames_y = paramnames

        for y, (p_y, row) in enumerate(zip(paramnames_y, axes)):
            for x, (p_x, ax) in enumerate(zip(paramnames_x, row)):
                if paramnames_x == paramnames_y and x > y:
                    kind='scatter'
                else:
                    kind='contour'
                self.plot(p_x, p_y, ax, prior=prior, kind=kind, colorscheme=color)
        return fig, axes

    def plot(self, paramname_x, paramname_y=None, ax=None, colorscheme='b',
             prior=False, kind='contour', *args, **kwargs):
        if paramname_y is None or paramname_x == paramname_y:
            xmin, xmax = self.prior_range.get(paramname_x, (None, None))
            return plot_1d(self[paramname_x], self.weights(prior=prior),
                           ax=ax, colorscheme=colorscheme,
                           xmin=xmin, xmax=xmax, *args, **kwargs)

        if kind == 'contour':
            plot = contour_plot_2d
        elif kind == 'scatter':
            plot = scatter_plot_2d
        xmin, xmax = self.prior_range.get(paramname_x, (None, None))
        ymin, ymax = self.prior_range.get(paramname_y, (None, None))

        return plot(self[paramname_x], self[paramname_y], self.weights(prior=prior),
                    ax=ax, colorscheme=colorscheme,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
                    *args, **kwargs)

    def cov(self, paramnames, prior=False):
        return pandas.DataFrame(numpy.cov(self[paramnames].T,
                                          aweights=self.weights(prior)),
                                columns=paramnames, index=paramnames)
