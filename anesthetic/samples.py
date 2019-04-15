"""Main classes for the anesthetic module.

- ``MCMCSamples``
- ``NestedSamples``
"""
import numpy
import pandas
from scipy.special import logsumexp
from anesthetic.plot import (make_1D_axes, make_2D_axes, plot_1d,
                             scatter_plot_2d, contour_plot_2d)
from anesthetic.read import (read_chains, read_birth, read_limits,
                             read_paramnames)
from anesthetic.utils import compress_weights


class MCMCSamples(pandas.DataFrame):
    """Storage and plotting tools for MCMC samples.

    We extend the pandas.DataFrame by providing plotting methods and
    standardising sample storage.

    Note that because of the design of pandas this does not override the
    __init__ constructor. You should build the samples with either:

    - ``mcmc = MCMCSamples.read('your/file/root')``
    - ``mcmc = MCMCSamples.build(samples=samples, other_keyword_arguments)``

    Example plotting commands include

    - ``mcmc.plot_1d()``
    - ``mcmc.plot_2d(['paramA', 'paramB'])``
    - ``mcmc.plot_2d(['paramA', 'paramB'], ['paramC', 'paramD'])``

    """

    _metadata = (pandas.DataFrame._metadata +
                 ['params', 'tex', 'limits', 'root'])

    @classmethod
    def read(cls, root):
        """Read in data from file root."""
        # Read in data
        w, logL, samples = read_chains(root)
        params, tex = read_paramnames(root)
        limits = read_limits(root)

        # Build class
        data = cls.build(samples=samples, w=w, logL=logL, params=params,
                         tex=tex, limits=limits)

        # Record root
        data.root = root
        return data

    @classmethod
    def build(cls, **kwargs):
        """Build an augmented pandas array for MCMC samples.

        Parameters
        ----------
        samples: numpy.array
            Coordinates of samples. shape = (nsamples, ndims).

        logL: numpy.array
            loglikelihoods of samples.

        w: numpy.array
            weights of samples.

        params: list(str)
            reference names of parameters

        tex: dict
            mapping from params to tex labels for plotting

        limits: dict
            mapping from params to prior limits

        """
        samples = kwargs.pop('samples', None)
        logL = kwargs.pop('logL', None)
        if samples is None and logL is None:
            raise ValueError("You must provide either samples or logL")
        elif samples is None:
            samples = numpy.empty((len(logL), 0))

        nsamples, nparams = numpy.atleast_2d(samples).shape

        w = kwargs.pop('w', None)
        params = kwargs.pop('params', ['x%i' % i for i in range(nparams)])

        tex = kwargs.pop('tex', {})
        limits = kwargs.pop('limits', {})

        data = cls(data=samples, columns=params)
        if w is not None:
            data['w'] = w
            tex['w'] = r'MCMC weight'
        if logL is not None:
            data['logL'] = logL
            tex['logL'] = r'$\log\mathcal{L}$'

        data['u'] = numpy.random.rand(len(data))

        data.tex = tex
        data.params = params
        data.limits = limits
        data.root = None
        return data

    def plot(self, ax, paramname_x, paramname_y=None, *args, **kwargs):
        """Interface for 2D and 1D plotting routines.

        Produces a single 1D or 2D plot on an axis.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            Axes to plot on

        paramname_x: str
            Choice of parameter to plot on x-coordinate from self.columns.

        paramname_y: str
            Choice of parameter to plot on y-coordinate from self.columns.
            If not provided, or the same as paramname_x, then 1D plot produced.

        plot_type: str, optional
            Must be in {'kde','scatter'}. (Default: 'kde')

        beta: float, optional
            Temperature to plot at. beta=0 corresponds to the prior, beta=1
            corresponds to the posterior. (Default: 1)

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.DataFrame or pandas.Series of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        plot_type = kwargs.pop('plot_type', 'kde')
        beta = kwargs.pop('beta', 1)

        if beta != 1 and not isinstance(self, NestedSamples):
            raise ValueError("You cannot adjust temperature for MCMCSamples")

        if paramname_y is None or paramname_x == paramname_y:
            xmin, xmax = self._limits(paramname_x)
            return plot_1d(ax, numpy.repeat(self[paramname_x],
                           self._weights(beta)), xmin=xmin, xmax=xmax,
                           *args, **kwargs)

        xmin, xmax = self._limits(paramname_x)
        ymin, ymax = self._limits(paramname_y)

        if plot_type == 'kde':
            weights = self._weights(beta)
            plot = contour_plot_2d
        elif plot_type == 'scatter':
            weights = self._weights(beta, nsamples=500)
            plot = scatter_plot_2d
        else:
            raise ValueError("plot_type must be in {'kde', 'scatter'}")

        return plot(ax, numpy.repeat(self[paramname_x], weights),
                    numpy.repeat(self[paramname_y], weights),
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    *args, **kwargs)

    def plot_1d(self, axes=None, *args, **kwargs):
        """Create an array of 1D plots.

        Parameters
        ----------
        axes: plotting axes, optional
            Can be either:
                * list(str) or str
                * pandas.Series(matplotlib.axes.Axes)
            If a pandas.Series is provided as an existing set of axes, then
            this is used for creating the plot. Otherwise a new set of axes are
            created using the list or lists of strings.
            Default: self.params

        beta: float, optional
            Temperature to plot at. beta=0 corresponds to the prior, beta=1
            corresponds to the posterior. (Default: 1)

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.Series of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        beta = kwargs.pop('beta', 1)
        if axes is None:
            axes = self.params

        if not isinstance(axes, pandas.Series):
            fig, axes = make_1D_axes(axes, tex=self.tex)
        else:
            fig = axes.values[~axes.isna()][0].figure

        for x, ax in axes.iteritems():
            if ax is not None and x in self:
                self.plot(ax, x, beta=beta, *args, **kwargs)

        return fig, axes

    def plot_2d(self, axes=None, *args, **kwargs):
        """Create an array of 2D plots.

        To avoid intefering with y-axis sharing, one-dimensional plots are
        created on a separate axis, which is monkey-patched onto the argument
        ax as the attribute ax.twin.

        Parameters
        ----------
        axes: plotting axes, optional
            Can be either:
                * list(str) if the x and y axes are the same
                * [list(str),list(str)] if the x and y axes are different
                * pandas.DataFrame(matplotlib.axes.Axes)
            If a pandas.DataFrame is provided as an existing set of axes, then
            this is used for creating the plot. Otherwise a new set of axes are
            created using the list or lists of strings.
            Default: self.params

        types: list(str) or str, optional
            What type (or types) of plots to produce. If two types are provided
            then pairs of parameters 'above the diagonal' have the second type.
            each string must be one of {'kde', 'scatter'}.
            Default: ['kde', 'scatter']

        beta: float, optional
            Temperature to plot at. beta=0 corresponds to the prior, beta=1
            corresponds to the posterior. (Default: 1)

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.DataFrame of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        types = kwargs.pop('types', ['kde', 'scatter'])
        beta = kwargs.pop('beta', 1)
        if axes is None:
            axes = self.params

        if not isinstance(axes, pandas.DataFrame):
            upper = None if len(types) > 1 else False
            fig, axes = make_2D_axes(axes, tex=self.tex, upper=upper)
        else:
            fig = axes.values[~axes.isna()][0].figure

        types = numpy.atleast_1d(types)

        for y, row in axes.iterrows():
            for x, ax in row.iteritems():
                if ax is not None and x in self and y in self:
                    plot_type = types[-1] if ax._upper else types[0]
                    ax_ = ax.twin if x == y else ax
                    self.plot(ax_, x, y, plot_type=plot_type, beta=beta,
                              *args, **kwargs)

        return fig, axes

    def _weights(self, beta, nsamples=None):
        """Return the posterior weights for plotting."""
        try:
            return compress_weights(self.w, self.u, nsamples=nsamples)
        except AttributeError:
            return numpy.ones(len(self), dtype=int)

    def _limits(self, paramname):
        return self.limits.get(paramname, (None, None))

    def _reload_data(self):
        self = type(self).read(self.root)


class NestedSamples(MCMCSamples):
    """Storage and plotting tools for Nested Sampling samples.

    We extend the MCMCSamples class with the additional methods:

    * ``self.ns_output()``
    * ``self.live_points(logL)``
    * ``self.posterior_points(beta)``

    Note that because of the design of pandas this does not override the
    __init__ constructor. You should build the samples with either:

    * ``NestedSamples.read('your/file/root')``
    * ``NestedSamples.build(keyword_arguments)``
    """

    @classmethod
    def read(cls, root):
        """Read in data from file root."""
        # Read in data
        params, tex = read_paramnames(root)
        limits = read_limits(root)
        samples, logL, logL_birth = read_birth(root)

        # Build class
        data = cls.build(samples=samples, logL=logL, params=params,
                         tex=tex, limits=limits, logL_birth=logL_birth)

        # Record root
        data.root = root
        return data

    @classmethod
    def build(cls, **kwargs):
        """Build an augmented pandas array for Nested samples.

        Parameters
        ----------
        params: numpy.array
            Coordinates of samples. shape = (nsamples, ndims).

        logL: numpy.array
            loglikelihoods of samples.

        logL_birth: numpy.array
            birth loglikelihoods of samples.

        w: numpy.array
            weights of samples.

        params: list(str)
            reference names of parameters

        tex: dict
            mapping from params to tex labels for plotting

        limits: dict
            mapping from params to prior limits

        """
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
        data['logw'] = data._dlogX()
        return data

    def ns_output(self, nsamples=100):
        """Compute Bayesian global quantities.

        Using nested sampling we can compute the evidence (logZ),
        Kullback-Leibler divergence (D) and Bayesian model dimensionality (d).
        More precisely, we can infer these quantities via their probability
        distribution.

        Parameters
        ----------
        nsamples: int, optional
            number of samples to generate (Default: 100)

        Returns
        -------
        pandas.DataFrame
            Samples from the P(logZ, D, d) distribution

        """
        dlogX = self._dlogX(nsamples)

        logZ = logsumexp(self.logL.values + dlogX, axis=1)
        logw = ((self.logL.values + dlogX).T - logZ).T
        S = ((self.logL.values + numpy.zeros_like(dlogX)).T
             - logZ).T

        D = numpy.exp(logsumexp(logw, b=S, axis=1))
        d = numpy.exp(logsumexp(logw, b=(S.T-D).T**2, axis=1))*2

        samples = numpy.vstack((logZ, D, d)).T
        params = ['logZ', 'D', 'd']
        tex = {'logZ': r'$\log\mathcal{Z}$',
               'D': r'$\mathcal{D}$',
               'd': r'$d$'}
        return MCMCSamples.build(samples=samples, params=params, tex=tex)

    def live_points(self, logL):
        """Get the live points within logL."""
        return self[(self.logL > logL) & (self.logL_birth <= logL)]

    def posterior_points(self, beta):
        """Get the posterior points at temperature beta."""
        return self[self._weights(beta, nsamples=-1) > 0]

    def _weights(self, beta, nsamples=None):
        """Return the posterior weights for plotting."""
        logw = self.logw + beta*self.logL
        w = numpy.exp(logw - logw.max())
        return compress_weights(w, self.u, nsamples=nsamples)

    def _dlogX(self, nsamples=None):
        """Compute volume of shell of loglikelihood.

        Parameters
        ----------
        nsamples: int, optional
            Number of samples to generate. optional. If None, then compute the
            statistical average. If integer, generate samples from the
            distribution. (Default: None)

        """
        if nsamples is None:
            t = numpy.atleast_2d(numpy.log(self.nlive/(self.nlive+1)))
            nsamples = 1
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
