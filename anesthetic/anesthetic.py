import numpy
import pandas
from scipy.special import logsumexp
from anesthetic.plot import make_1D_axes, make_2D_axes, plot_1d, scatter_plot_2d, contour_plot_2d
from anesthetic.read import read_chains, read_birth, read_limits, read_paramnames

def load_nested_samples(root):
    return NestedSamplingKDE.read(root)

#def build_samples(samples, paramnames=None, tex=None, prior_range=None):
    #return NestedSamplingKDE.build(samples, paramnames=paramnames, tex=tex, prior_range=prior_range)

class MCMCSamples(pandas.DataFrame):
    """Extension to pandas DataFrame for storing and plotting MCMC samples.

    We extend the DataFrame by providing plotting methods and standardising
    sample storage.
    """
    _metadata = ['paramnames', 'tex', 'limits']

    @classmethod
    def read(cls, root):
        # Read in data
        weights, logL, params = read_chains(root)
        paramnames, tex = read_paramnames(root)
        limits = read_limits(root)

        # Build class
        return cls.build(params=params, weights=weights, logL=logL, paramnames=paramnames, tex=tex, limits=limits)

    @classmethod
    def build(cls, **kwargs):
        params = kwargs.pop('params', None)
        logL = kwargs.pop('logL', None) 
        if params is None and logL is None:
            raise ValueError("You must provide either params or logL")
        elif params is None:
            params = numpy.empty(logL.shape[0],0)
        elif logL is None:
            logL = numpy.empty(params.shape[0],0)

        nsamps, nparams = params.shape

        weights = kwargs.pop('weights', numpy.ones((nsamps,1)))
        paramnames = kwargs.pop('paramnames', ['x%i' % i for i in range(nparams)])
        tex = kwargs.pop('tex', paramnames)
        limits = kwargs.pop('limits', {})
        columns = paramnames + ['weights', 'logL']
        data = numpy.concatenate((params, weights, logL), axis=1)

        data = cls(data=data, columns=columns)
        data.tex = tex
        data.paramnames = paramnames
        data.limits = limits

        return data

    def plot(self, paramname_x, paramname_y=None, ax=None, colorscheme='b',
             kind='contour', *args, **kwargs):
        """Generic plotting interface. 
        
        Produces a single 1D or 2D plot on an axis.

        Parameters
        ----------
        paramname_x: str
            Choice of parameter to plot on x-coordinate from self.columns.

        paramname_y: str
            Choice of parameter to plot on y-coordinate from self.columns.
            If not provided, or the same as paramname_x, then 1D plot produced.

        ax: matplotlib.axes.Axes
            Axes to plot on. 
            If not provided, the last axis is used
        """

        if paramname_y is None or paramname_x == paramname_y:
            xmin, xmax = self._limits(paramname_x)
            return plot_1d(self[paramname_x], self.weights,
                           ax=ax, colorscheme=colorscheme,
                           xmin=xmin, xmax=xmax, *args, **kwargs)

        xmin, xmax = self._limits(paramname_x)
        ymin, ymax = self._limits(paramname_y)

        if kind == 'contour':
            plot = contour_plot_2d
        elif kind == 'scatter':
            plot = scatter_plot_2d

        return plot(self[paramname_x], self[paramname_y], self.weights,
                    ax=ax, colorscheme=colorscheme,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
                    *args, **kwargs)

    def plot_1d(self, paramnames, axes=None, colorscheme='b', *args, **kwargs):
        """Create an array of 1D plots

        Parameters
        ----------
        paramnames: list(str) or str
            list of parameter names, or single parameter name to plot from
            self.columns

        axes: numpy.array(matplotlib.axes.Axes)
            Existing array of axes to plot on. If not provided, one is created.
        """
        paramnames = numpy.atleast_1d(paramnames)

        if axes is None:
            fig, axes = make_1D_axes(paramnames, self.tex)
        else:
            fig = numpy.atleast_2d(axes)[0,0].figure

        for p, ax in zip(paramnames, axes.flatten()):
            self.plot(p, ax=ax, colorscheme=colorscheme, *args, **kwargs)

        return fig, axes

    def plot_2d(self, paramnames, paramnames_y=None, axes=None, colorscheme='b'):
        paramnames_x = numpy.atleast_1d(paramnames)
        """Create an array of 2D plots

        Parameters
        ----------
        paramnames: list(str) or str
            list of parameter names, or single parameter name to plot from
            self.columns. If paramnames_y is not provided, produce triangle plot

        paramnames_y: list(str) or str
            list of parameter names, or single parameter name to plot on y
            coordinate from self.columns. If not provided, then a triangle plot
            is produced from paramnames

        axes: numpy.array(matplotlib.axes.Axes)
            Existing array of axes to plot on. If not provided, one is created.
        """
        if paramnames_y is None:
            paramnames_y = paramnames_x
        else:
            paramnames_y = numpy.atleast_1d(paramnames_y)

        if axes is None:
            fig, axes = make_2D_axes(paramnames, paramnames_y, self.tex)
        else:
            fig = numpy.atleast_2d(axes)[0,0].figure


        for y, (p_y, row) in enumerate(zip(paramnames_y, axes)):
            for x, (p_x, ax) in enumerate(zip(paramnames_x, row)):
                if (paramnames_x == paramnames_y).all() and x > y:
                    kind='scatter'
                else:
                    kind='contour'
                self.plot(p_x, p_y, ax, kind=kind, colorscheme=colorscheme)
        return fig, axes

    def cov(self, paramnames, prior=False):
        return pandas.DataFrame(numpy.cov(self[paramnames].T,
                                          aweights=self.weights(prior)),
                                columns=paramnames, index=paramnames)

    def _limits(self, paramname):
        return self.limits.get(paramname, (None, None))


class NestedSamples(MCMCSamples):
    _prior = False
    _integer = True

    @classmethod
    def read(cls, root):
        # Read in data
        paramnames, tex = read_paramnames(root)
        limits = read_limits(root)
        params, logL, logL_birth = read_birth(root)

        # Build class
        return cls.build(params=params, logL=logL, paramnames=paramnames, tex=tex, limits=limits, logL_birth=logL_birth)

    @classmethod
    def build(cls, **kwargs):
        # Build pandas DataFrame
        logL_birth = kwargs.pop('logL_birth', None)
        data = super(NestedSamples, cls).build(**kwargs)
        data.drop('weights', inplace=True, axis=1)
        data['logL_birth'] = logL_birth

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

    @property
    def weights(self):
        if self._prior:
            if self._integer:
                return self.prior_iweights
            else:
                return self.prior_weights
        else:
            if self._integer:
                return self.posterior_iweights
            else:
                return self.posterior_weights

