"""Main classes for the anesthetic module.

- ``MCMCSamples``
- ``NestedSamples``
"""
import os
import numpy
import pandas
from scipy.special import logsumexp
from anesthetic.plot import (make_1d_axes, make_2d_axes, fastkde_plot_1d,
                             kde_plot_1d, hist_plot_1d, scatter_plot_2d,
                             fastkde_contour_plot_2d,
                             kde_contour_plot_2d, hist_plot_2d)
from anesthetic.read.samplereader import SampleReader
from anesthetic.utils import compute_nlive
from anesthetic.gui.plot import RunPlotter
from anesthetic.weighted_pandas import WeightedDataFrame


class MCMCSamples(WeightedDataFrame):
    """Storage and plotting tools for MCMC samples.

    Extends the pandas.DataFrame by providing plotting methods and
    standardising sample storage.

    Example plotting commands include
        - ``mcmc.plot_1d(['paramA', 'paramB'])``
        - ``mcmc.plot_2d(['paramA', 'paramB'])``
        - ``mcmc.plot_2d([['paramA', 'paramB'], ['paramC', 'paramD']])``

    Parameters
    ----------
    root: str, optional
        root for reading chains from file. Overrides all other arguments.

    data: numpy.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns: list(str)
        reference names of parameters

    w: numpy.array
        weights of samples.

    logL: numpy.array
        loglikelihoods of samples.

    tex: dict
        mapping from columns to tex labels for plotting

    limits: dict
        mapping from columns to prior limits

    label: str
        Legend label

    """

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', None)
        if root is not None:
            reader = SampleReader(root)
            w, logL, samples = reader.samples()
            params, tex = reader.paramnames()
            limits = reader.limits()
            kwargs['label'] = kwargs.get('label', os.path.basename(root))
            self.__init__(data=samples, columns=params, w=w, logL=logL,
                          tex=tex, limits=limits, *args, **kwargs)
            self.root = root
        else:
            logL = kwargs.pop('logL', None)
            self.tex = kwargs.pop('tex', {})
            self.limits = kwargs.pop('limits', {})
            self.label = kwargs.pop('label', None)
            self.root = None
            super(MCMCSamples, self).__init__(*args, **kwargs)

            if logL is not None:
                self['logL'] = logL
                self.tex['logL'] = r'$\log\mathcal{L}$'

            self['weight'] = self.weight
            self.tex['weight'] = r'MCMC weight'

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
            optional, if not provided, or the same as paramname_x, then 1D plot
            produced.

        plot_type: str
            Must be in {'kde', 'scatter', 'hist', 'fastkde'} for 2D plots and
            in {'kde', 'hist', 'fastkde', 'astropyhist'} for 1D plots.
            optional, (Default: 'kde')

        ncompress: int
            Number of samples to use in plotting routines.
            optional, Default dynamically chosen

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.DataFrame or pandas.Series of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        plot_type = kwargs.pop('plot_type', 'kde')
        do_1d_plot = paramname_y is None or paramname_x == paramname_y
        kwargs['label'] = kwargs.get('label', self.label)
        ncompress = kwargs.pop('ncompress', None)

        if do_1d_plot:
            if paramname_x in self and plot_type is not None:
                xmin, xmax = self._limits(paramname_x)
                if plot_type == 'kde':
                    if ncompress is None:
                        ncompress = 1000
                    return kde_plot_1d(ax, self[paramname_x],
                                       weights=self.weight,
                                       ncompress=ncompress,
                                       xmin=xmin, xmax=xmax,
                                       *args, **kwargs)
                elif plot_type == 'fastkde':
                    x = self[paramname_x].compress(ncompress)
                    return fastkde_plot_1d(ax, x, xmin=xmin, xmax=xmax,
                                           *args, **kwargs)
                elif plot_type == 'hist':
                    return hist_plot_1d(ax, self[paramname_x],
                                        weights=self.weight,
                                        xmin=xmin, xmax=xmax, *args, **kwargs)
                elif plot_type == 'astropyhist':
                    x = self[paramname_x].compress(ncompress)
                    return hist_plot_1d(ax, x, plotter='astropyhist',
                                        xmin=xmin, xmax=xmax, *args, **kwargs)
                else:
                    raise NotImplementedError("plot_type is '%s', but must be"
                                              " one of {'kde', 'fastkde', "
                                              "'hist', 'astropyhist'}."
                                              % plot_type)
            else:
                ax.plot([], [])

        else:
            if (paramname_x in self and paramname_y in self
                    and plot_type is not None):
                xmin, xmax = self._limits(paramname_x)
                ymin, ymax = self._limits(paramname_y)
                if plot_type == 'kde':
                    if ncompress is None:
                        ncompress = 1000
                    x = self[paramname_x]
                    y = self[paramname_y]
                    return kde_contour_plot_2d(ax, x, y, weights=self.weight,
                                               xmin=xmin, xmax=xmax,
                                               ymin=ymin, ymax=ymax,
                                               ncompress=ncompress,
                                               *args, **kwargs)
                elif plot_type == 'fastkde':
                    x = self[paramname_x].compress(ncompress)
                    y = self[paramname_y].compress(ncompress)
                    return fastkde_contour_plot_2d(ax, x, y,
                                                   xmin=xmin, xmax=xmax,
                                                   ymin=ymin, ymax=ymax,
                                                   *args, **kwargs)
                elif plot_type == 'scatter':
                    if ncompress is None:
                        ncompress = 500
                    x = self[paramname_x].compress(ncompress)
                    y = self[paramname_y].compress(ncompress)
                    return scatter_plot_2d(ax, x, y, xmin=xmin, xmax=xmax,
                                           ymin=ymin, ymax=ymax,
                                           *args, **kwargs)
                elif plot_type == 'hist':
                    x = self[paramname_x]
                    y = self[paramname_y]
                    return hist_plot_2d(ax, x, y, weights=self.weight,
                                        xmin=xmin, xmax=xmax,
                                        ymin=ymin, ymax=ymax,
                                        *args, **kwargs)
                else:
                    raise NotImplementedError("plot_type is '%s', but must be"
                                              "in {'kde', 'fastkde',"
                                              "'scatter', 'hist'}."
                                              % plot_type)

            else:
                ax.plot([], [])

    def plot_1d(self, axes, *args, **kwargs):
        """Create an array of 1D plots.

        Parameters
        ----------
        axes: plotting axes
            Can be:
                - list(str) or str
                - pandas.Series(matplotlib.axes.Axes)
            If a pandas.Series is provided as an existing set of axes, then
            this is used for creating the plot. Otherwise a new set of axes are
            created using the list or lists of strings.

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.Series of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        if not isinstance(axes, pandas.Series):
            fig, axes = make_1d_axes(axes, tex=self.tex)
        else:
            fig = axes.values[~axes.isna()][0].figure

        for x, ax in axes.iteritems():
            self.plot(ax, x, *args, **kwargs)

        return fig, axes

    def plot_2d(self, axes, *args, **kwargs):
        """Create an array of 2D plots.

        To avoid intefering with y-axis sharing, one-dimensional plots are
        created on a separate axis, which is monkey-patched onto the argument
        ax as the attribute ax.twin.

        Parameters
        ----------
        axes: plotting axes
            Can be:
                - list(str) if the x and y axes are the same
                - [list(str),list(str)] if the x and y axes are different
                - pandas.DataFrame(matplotlib.axes.Axes)
            If a pandas.DataFrame is provided as an existing set of axes, then
            this is used for creating the plot. Otherwise a new set of axes are
            created using the list or lists of strings.

        types: dict, optional
            What type (or types) of plots to produce. Takes the keys 'diagonal'
            for the 1D plots and 'lower' and 'upper' for the 2D plots.
            The options for 'diagonal are:
                - 'kde'
                - 'hist'
                - 'astropyhist'
            The options for 'lower' and 'upper' are:
                - 'kde'
                - 'scatter'
                - 'hist'
                - 'fastkde'
            Default: {'diagonal': 'kde', 'lower': 'kde', 'upper':'scatter'}

        diagonal_kwargs, lower_kwargs, upper_kwargs: dict, optional
            kwargs for the diagonal (1D)/lower or upper (2D) plots. This is
            useful when there is a conflict of kwargs for different types of
            plots.  Note that any kwargs directly passed to plot_2d will
            overwrite any kwarg with the same key passed to <sub>_kwargs.
            Default: {}

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.DataFrame of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        default_types = {'diagonal': 'kde', 'lower': 'kde', 'upper': 'scatter'}
        types = kwargs.pop('types', default_types)
        diagonal = kwargs.pop('diagonal', True)
        if isinstance(types, list) or isinstance(types, str):
            from warnings import warn
            warn("MCMCSamples.plot_2d's argument 'types' might stop accepting "
                 "str or list(str) as input in the future. It takes a "
                 "dictionary as input, now, with keys 'diagonal' for the 1D "
                 "plots and 'lower' and 'upper' for the 2D plots. 'diagonal' "
                 "accepts the values 'kde' or 'hist' and both 'lower' and "
                 "'upper' accept the values 'kde' or 'scatter'. "
                 "Default: {'diagonal': 'kde', 'lower': 'kde'}.",
                 FutureWarning)

            if isinstance(types, str):
                types = {'lower': types}
                if diagonal:
                    types['diagonal'] = types['lower']
            elif isinstance(types, list):
                types = {'lower': types[0], 'upper': types[-1]}
                if diagonal:
                    types['diagonal'] = types['lower']

        local_kwargs = {pos: kwargs.pop('%s_kwargs' % pos, {})
                        for pos in default_types}

        for pos in local_kwargs:
            local_kwargs[pos].update(kwargs)

        if not isinstance(axes, pandas.DataFrame):
            fig, axes = make_2d_axes(axes, tex=self.tex,
                                     upper=('upper' in types),
                                     lower=('lower' in types),
                                     diagonal=('diagonal' in types))
        else:
            fig = axes.values[~axes.isna()][0].figure

        for y, row in axes.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    pos = ax.position
                    ax_ = ax.twin if x == y else ax
                    plot_type = types.get(pos, None)
                    lkwargs = local_kwargs.get(pos, {})
                    self.plot(ax_, x, y, plot_type=plot_type, *args, **lkwargs)

        return fig, axes

    def _limits(self, paramname):
        return self.limits.get(paramname, (None, None))

    def _reload_data(self):
        self.__init__(root=self.root)
        return self

    _metadata = WeightedDataFrame._metadata + ['tex', 'limits',
                                               'root', 'label']

    @property
    def _constructor(self):
        return MCMCSamples


class NestedSamples(MCMCSamples):
    """Storage and plotting tools for Nested Sampling samples.

    We extend the MCMCSamples class with the additional methods:

    * ``self.ns_output()``
    * ``self.live_points(logL)``
    * ``self.posterior_points(beta)``

    Parameters
    ----------
    root: str, optional
        root for reading chains from file. Overrides all other arguments.

    data: numpy.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns: list(str)
        reference names of parameters

    logL: numpy.array
        loglikelihoods of samples.

    logL_birth: numpy.array or int
        birth loglikelihoods, or number of live points.

    tex: dict
        mapping from columns to tex labels for plotting

    limits: dict
        mapping from columns to prior limits

    label: str
        Legend label

    beta: float
        thermodynamic temperature

    """

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', None)
        if root is not None:
            reader = SampleReader(root)
            samples, logL, logL_birth = reader.samples()
            params, tex = reader.paramnames()
            limits = reader.limits()
            kwargs['label'] = kwargs.get('label', os.path.basename(root))
            self.__init__(data=samples, columns=params,
                          logL=logL, logL_birth=logL_birth,
                          tex=tex, limits=limits, *args, **kwargs)
            self.root = root
        else:
            self._beta = kwargs.pop('beta', 1.)
            logL_birth = kwargs.pop('logL_birth', None)
            super(NestedSamples, self).__init__(*args, **kwargs)

            # Compute nlive
            if logL_birth is not None:
                if isinstance(logL_birth, int):
                    nlive = logL_birth
                    self['nlive'] = nlive
                    descending = numpy.arange(nlive, 0, -1)
                    self.loc[len(self)-nlive:, 'nlive'] = descending
                else:
                    self['logL_birth'] = logL_birth
                    self.tex['logL_birth'] = r'$\log\mathcal{L}_{\rm birth}$'
                    self['nlive'] = compute_nlive(self.logL, self.logL_birth)

                self.tex['nlive'] = r'$n_{\rm live}$'

            if 'nlive' in self:
                self.beta = self._beta

    @property
    def beta(self):
        """Thermodynamic inverse temperature."""
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        logw = self._dlogX() + self.beta*self.logL
        self._weight = numpy.exp(logw - logw.max())

    def set_beta(self, beta, inplace=False):
        """Change the inverse temperature.

        Parameters
        ----------
        beta: float
            Temperature to set

        inplace: bool, optional
            Indicates whether to modify the existing array, or return a copy
            with the temperature changed. Default: False

        """
        if inplace:
            self.beta = beta
        else:
            data = self.copy()
            data.beta = beta
            return data

    def ns_output(self, nsamples=200):
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
        return MCMCSamples(data=samples, columns=params, tex=tex)

    def live_points(self, logL):
        """Get the live points within logL."""
        return self[(self.logL > logL) & (self.logL_birth <= logL)]

    def posterior_points(self, beta):
        """Get the posterior points at temperature beta."""
        return self.set_beta(beta).compress(0)

    def gui(self, params=None):
        """Construct a graphical user interface for viewing samples."""
        return RunPlotter(self, params)

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

    _metadata = MCMCSamples._metadata + ['_beta']

    @property
    def _constructor(self):
        return NestedSamples
