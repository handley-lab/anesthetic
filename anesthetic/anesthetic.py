import numpy
import pandas
from scipy.special import logsumexp
from anesthetic.plot import make_1D_axes, make_2D_axes, plot_1d, scatter_plot_2d, contour_plot_2d
from anesthetic.read import read_chains, read_birth, read_limits, read_paramnames
from anesthetic.information_theory import compress_weights


class MCMCSamples(pandas.DataFrame):
    """Storage and plotting tools for MCMC samples

    We extend the pandas.DataFrame by providing plotting methods and standardising
    sample storage.

    Note that because of the design of pandas this does not override the
    __init__ constructor. You should build the samples with either:

    * `mcmc = MCMCSamples.read('your/file/root')`
    * `mcmc = MCMCSamples.build(params=params, other_keyword_arguments)`

    Example plotting commands include

    * `mcmc.plot_1d()`
    * `mcmc.plot_2d(['paramA', 'paramB'])`
    * `mcmc.plot_2d(['paramA', 'paramB'],['paramC', 'paramD'])`

    """
    _metadata = pandas.DataFrame._metadata + ['paramnames', 'tex', 'limits', 'root']

    @classmethod
    def read(cls, root):
        # Read in data
        w, logL, params = read_chains(root)
        paramnames, tex = read_paramnames(root)
        limits = read_limits(root)

        # Build class
        data = cls.build(params=params, w=w, logL=logL, paramnames=paramnames,
                         tex=tex, limits=limits)

        # Record root
        data.root = root
        return data

    @classmethod
    def build(cls, **kwargs):
        params = kwargs.pop('params', None)
        logL = kwargs.pop('logL', None) 
        if params is None and logL is None:
            raise ValueError("You must provide either params or logL")
        elif params is None:
            params = numpy.empty(logL.shape[0],0)

        nsamps, nparams = params.shape

        w = kwargs.pop('w', None)
        paramnames = kwargs.pop('paramnames', ['x%i' % i for i in range(nparams)])

        tex = kwargs.pop('tex', {})
        if isinstance(tex, list):
            tex = {p:t for p, t in zip(paramnames, tex)}

        limits = kwargs.pop('limits', {})

        data = cls(data=params, columns=paramnames)
        if w is not None:
            data['w'] = w 
            tex['w'] = r'MCMC weight'
        if logL is not None:
            data['logL'] = logL
            tex['logL'] = r'$\log\mathcal{L}$'

        data['u'] = numpy.random.rand(len(data))

        data.tex = tex
        data.paramnames = paramnames
        data.limits = limits
        data.root = None
        return data

    def plot(self, paramname_x, paramname_y=None, ax=None, colorscheme='b',
             kind='contour', beta=1, *args, **kwargs):
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
            return plot_1d(self[paramname_x], self.weights(beta),
                           ax=ax, colorscheme=colorscheme,
                           xmin=xmin, xmax=xmax, *args, **kwargs)

        xmin, xmax = self._limits(paramname_x)
        ymin, ymax = self._limits(paramname_y)

        if kind == 'contour':
            return contour_plot_2d(self[paramname_x], self[paramname_y],
                                   self.weights(beta),
                                   ax=ax, colorscheme=colorscheme,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
                                   *args, **kwargs)
        elif kind == 'scatter':
            return scatter_plot_2d(self[paramname_x], self[paramname_y],
                                   self.weights(beta, nsamples=500),
                                   ax=ax, colorscheme=colorscheme,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
                                   *args, **kwargs)


    def plot_1d(self, paramnames=None, axes=None, colorscheme='b', beta=1, *args, **kwargs):
        """Create an array of 1D plots

        Parameters
        ----------
        paramnames: list(str) or str
            list of parameter names, or single parameter name to plot from
            self.columns

        axes: numpy.array(matplotlib.axes.Axes)
            Existing array of axes to plot on. If not provided, one is created.
        """
        if paramnames is None:
            paramnames = self.paramnames
        else:
            paramnames = numpy.atleast_1d(paramnames)

        if axes is None:
            fig, axes = make_1D_axes(paramnames, self.full_tex())
        else:
            fig = numpy.atleast_2d(axes)[0,0].figure

        for p, ax in zip(paramnames, axes.flatten()):
            self.plot(p, ax=ax, colorscheme=colorscheme, beta=beta, *args, **kwargs)

        return fig, axes

    def plot_2d(self, paramnames, paramnames_y=None, axes=None, colorscheme='b', beta=1, *args, **kwargs):
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
        paramnames_x = numpy.atleast_1d(paramnames)
        if paramnames_y is None:
            paramnames_y = paramnames_x
        else:
            paramnames_y = numpy.atleast_1d(paramnames_y)
        all_paramnames = list(paramnames_y) +list(paramnames_x)

        if axes is None:
            fig, axes = make_2D_axes(paramnames_x, paramnames_y, self.full_tex())
        else:
            fig = numpy.atleast_2d(axes)[0,0].figure

        for p_y, row in zip(paramnames_y, axes):
            for p_x, ax in zip(paramnames_x, row):
                if p_x in paramnames_y and p_y in paramnames_x and all_paramnames.index(p_x) > all_paramnames.index(p_y):
                    kind='scatter'
                else:
                    kind='contour'
                self.plot(p_x, p_y, ax, kind=kind, colorscheme=colorscheme, beta=beta, *args, **kwargs)
        return fig, axes

    def weights(self, beta, nsamples=None):
        """ Return the posterior weights for plotting. """
        try:
            return compress_weights(self.w, self.u, nsamples=nsamples)
        except AttributeError:
            return numpy.ones(len(self), dtype=int, unit_weights=unit_weights)

    def _limits(self, paramname):
        return self.limits.get(paramname, (None, None))

    def full_tex(self):
        return {p:self.tex.get(p,p) for p in self.paramnames}

    def reload_data(self):
        self = type(self).read(self.root)


class NestedSamples(MCMCSamples):
    """Storage and plotting tools for Nested Sampling samples

    We extend the MCMCSamples class with the additional methods:
    
    * ns_output

    Note that because of the design of pandas this does not override the
    __init__ constructor. You should build the samples with either:

    * NestedSamples.read('your/file/root')
    * NestedSamples.build(params=params, other_keyword_arguments)
    """

    @classmethod
    def read(cls, root):
        # Read in data
        paramnames, tex = read_paramnames(root)
        limits = read_limits(root)
        params, logL, logL_birth = read_birth(root)

        # Build class
        data = cls.build(params=params, logL=logL, paramnames=paramnames, tex=tex, limits=limits, logL_birth=logL_birth)

        # Record root
        data.root = root
        return data

    @classmethod
    def build(cls, **kwargs):
        # Build pandas DataFrame
        logL_birth = kwargs.pop('logL_birth', None)
        data = super(NestedSamples, cls).build(**kwargs)
        data['logL_birth'] = logL_birth

        # Compute nlive
        index = data.logL.searchsorted(data.logL_birth)-1
        births = pandas.Series(+1, index=index).sort_index()
        deaths = pandas.Series(-1, index=data.index)
        nlive = pandas.concat([births, deaths]).sort_index().cumsum()
        nlive = (nlive[~nlive.index.duplicated(keep='first')]+1)[1:]
        data['nlive'] = nlive
        data['logw'] = data.dlogX()
        return data

    def dlogX(self, nsamples=None):
        if nsamples is None:
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

    def ns_output(self, nsamples=100):
        columns = ['logZ', 'D', 'd']
        dlogX = self.dlogX(nsamples)

        logZ = logsumexp(self.logL.values + dlogX, axis=1)
        logw = ((self.logL.values + dlogX).T - logZ).T
        S = ((self.logL.values + numpy.zeros_like(dlogX)).T
             - logZ).T

        D = numpy.exp(logsumexp(logw, b=S, axis=1))
        d = numpy.exp(logsumexp(logw, b=(S.T-D).T**2, axis=1))*2

        params = numpy.vstack((logZ, D, d)).T
        paramnames = ['logZ', 'D', 'd']
        tex = [r'$\log\mathcal{Z}$', r'$\mathcal{D}$', r'$d$']
        return MCMCSamples.build(params=params, paramnames=paramnames, tex=tex)

    def live_points(self, logL):
        """ Get the live points within logL """
        return self[(self.logL > logL) & (self.logL_birth <= logL)]

    def posterior_points(self, beta):
        """ Get the posterior points at temperature beta """
        return self[self.weights(beta, nsamples=-1)>0]

    def weights(self, beta, nsamples=None):
        logw = self.logw + beta*self.logL
        w = numpy.exp(logw - logw.max())
        return compress_weights(w, self.u, nsamples=nsamples)
