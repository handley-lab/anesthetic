"""Main classes for the anesthetic module.

- ``MCMCSamples``
- ``NestedSamples``
"""
import os
import numpy as np
import pandas
import copy
import warnings
from pandas import MultiIndex, DataFrame, Series
from collections.abc import Sequence
from anesthetic.plot import (make_1d_axes, make_2d_axes, fastkde_plot_1d,
                             kde_plot_1d, hist_plot_1d, scatter_plot_2d,
                             fastkde_contour_plot_2d,
                             kde_contour_plot_2d, hist_plot_2d)
from anesthetic.read.samplereader import SampleReader
from anesthetic.utils import (compute_nlive, compute_insertion_indexes,
                              is_int, logsumexp, modify_inplace)
from anesthetic.gui.plot import RunPlotter
from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries


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

    data: np.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns: list(str)
        reference names of parameters

    weights: np.array
        weights of samples.

    logL: np.array
        loglikelihoods of samples.

    tex: dict
        mapping from columns to tex labels for plotting

    limits: dict
        mapping from columns to prior limits

    label: str
        Legend label

    logzero: float
        The threshold for `log(0)` values assigned to rejected sample points.
        Anything equal or below this value is set to `-np.inf`.
        default: -1e30

    burn_in: int or float
        Discards the first integer number of nsamples if int
        or the first fraction of nsamples if float.
        Only works if `root` provided and if chains are GetDist compatible.
        default: False

    """

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', None)
        if root is not None:
            reader = SampleReader(root)
            if hasattr(reader, 'birth_file') or hasattr(reader, 'ev_file'):
                raise ValueError("The file root %s seems to point to a Nested "
                                 "Sampling chain. Please use NestedSamples "
                                 "instead which has the same features as "
                                 "MCMCSamples and more. MCMCSamples should be "
                                 "used for MCMC chains only." % root)
            burn_in = kwargs.pop('burn_in', False)
            weights, logL, samples = reader.samples(burn_in=burn_in)
            params, tex = reader.paramnames()
            columns = kwargs.pop('columns', params)
            limits = reader.limits()
            kwargs['label'] = kwargs.get('label', os.path.basename(root))
            self.__init__(data=samples, columns=columns, weights=weights,
                          logL=logL, tex=tex, limits=limits, *args, **kwargs)
            self.root = root
        else:
            logzero = kwargs.pop('logzero', -1e30)
            logL = kwargs.pop('logL', None)
            if logL is not None:
                logL = np.array(logL)
                logL = np.where(logL <= logzero, -np.inf, logL)
            self.tex = kwargs.pop('tex', {})
            self.limits = kwargs.pop('limits', {})
            self.label = kwargs.pop('label', None)
            self.root = None
            super().__init__(*args, **kwargs)

            if logL is not None:
                self['logL'] = logL
                self.tex['logL'] = r'$\ln\mathcal{L}$'

            self._set_automatic_limits()

    def _set_automatic_limits(self):
        """Set all unassigned limits to min and max of sample."""
        for param in self.columns:
            if param not in self.limits:
                self.limits[param] = (self[param].min(), self[param].max())

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

        q: str, float, (float, float)
            Plot the `q` inner posterior quantiles in 1d 'kde' plots. To plot
            the full range, set `q=0` or `q=1`.
            * if str: any of {'1sigma', '2sigma', '3sigma', '4sigma', '5sigma'}
                Plot within mean +/- #sigma of posterior.
            * if float: Plot within the symmetric confidence interval
                `(1-q, q)`  or `(q, 1-q)`.
            * if tuple:  Plot within the (possibly asymmetric) confidence
                interval `q`.
            optional, (Default: '5sigma')

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.DataFrame or pandas.Series of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        self._set_automatic_limits()
        plot_type = kwargs.pop('plot_type', 'kde')
        do_1d_plot = paramname_y is None or paramname_x == paramname_y
        kwargs['label'] = kwargs.get('label', self.label)
        ncompress = kwargs.pop('ncompress', None)

        if do_1d_plot:
            if paramname_x in self and plot_type is not None:
                xmin, xmax = self._limits(paramname_x)
                kwargs['xmin'] = kwargs.get('xmin', xmin)
                kwargs['xmax'] = kwargs.get('xmax', xmax)
                if plot_type == 'kde':
                    if ncompress is None:
                        ncompress = 1000
                    return kde_plot_1d(ax, self[paramname_x],
                                       weights=self.weights,
                                       ncompress=ncompress,
                                       *args, **kwargs)
                elif plot_type == 'fastkde':
                    x = self[paramname_x].compress(ncompress)
                    return fastkde_plot_1d(ax, x, *args, **kwargs)
                elif plot_type == 'hist':
                    return hist_plot_1d(ax, self[paramname_x],
                                        weights=self.weights,
                                        *args, **kwargs)
                elif plot_type == 'astropyhist':
                    x = self[paramname_x].compress(ncompress)
                    return hist_plot_1d(ax, x, plotter='astropyhist',
                                        *args, **kwargs)
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
                kwargs['xmin'] = kwargs.get('xmin', xmin)
                kwargs['xmax'] = kwargs.get('xmax', xmax)
                ymin, ymax = self._limits(paramname_y)
                kwargs['ymin'] = kwargs.get('ymin', ymin)
                kwargs['ymax'] = kwargs.get('ymax', ymax)
                if plot_type == 'kde':
                    if ncompress is None:
                        ncompress = 1000
                    x = self[paramname_x]
                    y = self[paramname_y]
                    return kde_contour_plot_2d(ax, x, y, weights=self.weights,
                                               ncompress=ncompress,
                                               *args, **kwargs)
                elif plot_type == 'fastkde':
                    x = self[paramname_x].compress(ncompress)
                    y = self[paramname_y].compress(ncompress)
                    return fastkde_contour_plot_2d(ax, x, y,
                                                   *args, **kwargs)
                elif plot_type == 'scatter':
                    if ncompress is None:
                        ncompress = 500
                    x = self[paramname_x].compress(ncompress)
                    y = self[paramname_y].compress(ncompress)
                    return scatter_plot_2d(ax, x, y, *args, **kwargs)
                elif plot_type == 'hist':
                    x = self[paramname_x]
                    y = self[paramname_y]
                    return hist_plot_2d(ax, x, y, weights=self.weights,
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
        if not isinstance(axes, Series):
            fig, axes = make_1d_axes(axes, tex=self.tex)
        else:
            fig = axes.bfill().to_numpy().flatten()[0].figure

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
        local_kwargs = {pos: kwargs.pop('%s_kwargs' % pos, {})
                        for pos in default_types}

        for pos in local_kwargs:
            local_kwargs[pos].update(kwargs)

        if not isinstance(axes, DataFrame):
            fig, axes = make_2d_axes(axes, tex=self.tex,
                                     upper=('upper' in types),
                                     lower=('lower' in types),
                                     diagonal=('diagonal' in types))
        else:
            fig = axes.bfill().to_numpy().flatten()[0].figure

        for y, row in axes.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    pos = ax.position
                    ax_ = ax.twin if x == y else ax
                    plot_type = types.get(pos, None)
                    lkwargs = local_kwargs.get(pos, {})
                    self.plot(ax_, x, y, plot_type=plot_type, *args, **lkwargs)

        return fig, axes

    def importance_sample(self, logL_new, action='add', inplace=False):
        """Perform importance re-weighting on the log-likelihood.

        Parameters
        ----------
        logL_new: np.array
            New log-likelihood values. Should have the same shape as `logL`.

        action: str, optional
            Can be any of {'add', 'replace', 'mask'}.
                * add: Add the new `logL_new` to the current `logL`.
                * replace: Replace the current `logL` with the new `logL_new`.
                * mask: treat `logL_new` as a boolean mask and only keep the
                        corresponding (True) samples.
            default: 'add'

        inplace: bool, optional
            Indicates whether to modify the existing array, or return a new
            frame with importance sampling applied.
            default: False

        Returns
        -------
        samples: MCMCSamples
            Importance re-weighted samples.
        """
        samples = self.copy()
        if action == 'add':
            samples.weights *= np.exp(logL_new - logL_new.max())
            samples.logL += logL_new
        elif action == 'replace':
            logL_new2 = logL_new - samples.logL
            samples.weights *= np.exp(logL_new2 - logL_new2.max())
            samples.logL = logL_new
        elif action == 'mask':
            samples = samples[logL_new]
        else:
            raise NotImplementedError("`action` needs to be one of "
                                      "{'add', 'replace', 'mask'}, but '%s' "
                                      "was requested." % action)

        return modify_inplace(self, samples, inplace)

    def _limits(self, paramname):
        limits = self.limits.get(paramname, (None, None))
        if limits[0] == limits[1]:
            limits = (None, None)
        return limits

    def _reload_data(self):
        self.__init__(root=self.root)
        return self

    def copy(self, deep=True):
        """Copy which also includes mutable metadata."""
        new = super().copy(deep)
        if deep:
            new.tex = copy.deepcopy(self.tex)
            new.limits = copy.deepcopy(self.limits)
        return new

    _metadata = WeightedDataFrame._metadata + ['tex', 'limits',
                                               'root', 'label']

    @property
    def _constructor(self):
        return MCMCSamples


class NestedSamples(MCMCSamples):
    """Storage and plotting tools for Nested Sampling samples.

    We extend the MCMCSamples class with the additional methods:

    * ``self.live_points(logL)``
    * ``self.posterior_points(beta)``
    * ``self.ns_output()``
    * ``self.logZ()``
    * ``self.D()``
    * ``self.d()``
    * ``self.recompute()``
    * ``self.gui()``
    * ``self.importance_sample()``

    Parameters
    ----------
    root: str, optional
        root for reading chains from file. Overrides all other arguments.

    data: np.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns: list(str)
        reference names of parameters

    logL: np.array
        loglikelihoods of samples.

    logL_birth: np.array or int
        birth loglikelihoods, or number of live points.

    tex: dict
        optional mapping from column names to tex labels for plotting

    limits: dict
        mapping from columns to prior limits.
        Defaults defined by .ranges file (if it exists)
        otherwise defined by minimum and maximum of the nested sampling data

    label: str
        Legend label
        default: basename of root

    beta: float
        thermodynamic temperature
        default: 1.

    logzero: float
        The threshold for `log(0)` values assigned to rejected sample points.
        Anything equal or below this value is set to `-np.inf`.
        default: -1e30

    """

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', None)
        if root is not None:
            reader = SampleReader(root)
            samples, logL, logL_birth = reader.samples()
            params, tex = reader.paramnames()
            columns = kwargs.pop('columns', params)
            limits = reader.limits()
            kwargs['label'] = kwargs.get('label', os.path.basename(root))
            self.__init__(data=samples, columns=columns,
                          logL=logL, logL_birth=logL_birth,
                          tex=tex, limits=limits, *args, **kwargs)
            self.root = root
        else:
            logzero = kwargs.pop('logzero', -1e30)
            self._beta = kwargs.pop('beta', 1.)
            logL_birth = kwargs.pop('logL_birth', None)
            if not isinstance(logL_birth, int) and logL_birth is not None:
                logL_birth = np.array(logL_birth)
                logL_birth = np.where(logL_birth <= logzero, -np.inf,
                                      logL_birth)

            super().__init__(logzero=logzero, *args, **kwargs)
            if logL_birth is not None:
                self.recompute(logL_birth, inplace=True)

            self._set_automatic_limits()

    @property
    def beta(self):
        """Thermodynamic inverse temperature."""
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        logw = self.dlogX() + np.where(self.logL == -np.inf, -np.inf,
                                       self.beta * self.logL)
        self.weights = np.exp(logw - logw.max())

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

    def prior(self, inplace=False):
        """Re-weight samples at infinite temperature to get prior samples."""
        return self.set_beta(beta=0, inplace=inplace)

    def ns_output(self, nsamples=None, beta=None):
        """Compute Bayesian global quantities.

        Using nested sampling we can compute the evidence (logZ),
        Kullback-Leibler divergence (D) and Bayesian model dimensionality (d).
        More precisely, we can infer these quantities via their probability
        distribution.

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, return mean value
            - If nsamples is integer, return nsamples from the distribution of
              values inferred by nested sampling

        beta: float, array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if beta is scalar and nsamples is None:
            Series, index ['logZ', 'd', 'D', 'logL']
        elif beta is scalar and nsamples is int:
            DataFrame, index range(nsamples),
            columns ['logZ', 'd', 'D', 'logL'] 
        elif beta is array-like and nsamples is None:
            DataFrame, index beta,
            columns ['logZ', 'd', 'D', 'logL'] 
        elif beta is array-like and nsamples is int:
            DataFrame, index MultiIndex the product of beta and range(nsamples)
            columns ['logZ', 'd', 'D', 'logL'] 
        """
        logw = self.logw(nsamples, beta)
        if nsamples is None and beta is None:
            samples = Series(dtype=float)
        else:
            samples = DataFrame(index=logw.columns)
        samples['logZ'] = self.logZ(logw)

        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - samples.logZ

        samples['D'] = np.exp(logsumexp(logw-samples.logZ, b=S, axis=0))
        samples['d'] = np.exp(logsumexp(logw-samples.logZ, b=(S-samples.D)**2,
                              axis=0))*2
        samples['logL'] = samples['logZ'] + samples['D']

        samples.tex = {'logZ': r'$\ln\mathcal{Z}$',
                       'D': r'$\mathcal{D}$',
                       'd': r'$d$',
                       'logL': r'$\ln L$'}
        samples.label = self.label
        return samples

    def logX(self, nsamples=None):
        """Log-Volume.

        The log of the prior volume contained within each iso-likelihood
        contour.

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, return mean value
            - If nsamples is integer, return nsamples from the distribution of
              values inferred by nested sampling

        Returns
        -------
        if nsamples is None:
            WeightedSeries like self
        elif nsamples is int
            WeightedDataFrame like self, columns range(nsamples)
        """
        if nsamples is None:
            t = np.log(self.nlive/(self.nlive+1))
            t.name = 'logX'
        else:
            r = np.log(np.random.rand(len(self), nsamples))
            r = WeightedDataFrame(r, self.index, weights=self.weights)
            t = r.divide(self.nlive, axis=0)
            t.columns.name = 'samples'
        return t.cumsum()

    def _betalogL(self, beta=None):
        """log (L**beta) convenience function.

        Parameters
        ----------
        beta, scalar or array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if beta is scalar:
            WeightedSeries like self
        elif beta is array-like:
            WeightedDataFrame like self, columns of beta
        """
        if beta is None:
            beta = self.beta
        if np.isscalar(beta):
            betalogL = WeightedSeries(beta*self.logL, index=self.index)
            betalogL.name = 'betalogL'
        else:
            betalogL = WeightedDataFrame(np.outer(self.logL, beta),
                                         index=self.index, columns=beta)
            betalogL.columns.name = 'beta'
        return betalogL

    def logw(self, nsamples=None, beta=None):
        """Log-nested sampling weight.

        The logarithm of the (unnormalised) sampling weight log(L**beta*dX).

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, return mean value
            - If nsamples is integer, return nsamples from the distribution of
              values inferred by nested sampling
            - If nsamples is array, nsamples is assumed to be logw and returned
              (implementation convenience functionality)

        beta: float, array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if nsamples is array-like
            WeightedDataFrame equal to nsamples
        elif beta is scalar and nsamples is None:
            WeightedSeries like self
        elif beta is array-like and nsamples is None:
            WeightedDataFrame like self, columns of beta
        elif beta is scalar and nsamples is int:
            WeightedDataFrame like self, columns of range(nsamples)
        elif beta is array-like and nsamples is int:
            WeightedDataFrame like self, MultiIndex columns the product of beta
            and range(nsamples)
        """
        if np.ndim(nsamples) > 0:
            return nsamples

        dlogX = self.dlogX(nsamples)
        betalogL = self._betalogL(beta)
        if dlogX.ndim == 1 and betalogL.ndim == 1:
            logw = dlogX + betalogL
        elif dlogX.ndim > 1 and betalogL.ndim == 1:
            logw = dlogX.add(betalogL, axis=0)
        elif dlogX.ndim == 1 and betalogL.ndim > 1:
            logw = (dlogX + betalogL.T).T
        else:
            cols = MultiIndex.from_product([betalogL.columns, dlogX.columns])
            logw = DataFrame(dlogX).reindex(columns=cols, level='samples')
            logw = logw + betalogL
            logw = WeightedDataFrame(logw)
        return logw

    def logZ(self, nsamples=None, beta=None):
        """Log-Evidence.

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, return mean value
            - If nsamples is integer, return nsamples from the distribution of
              values inferred by nested sampling
            - If nsamples is array, nsamples is assumed to be logw

        beta: float, array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if nsamples is array-like:
            Series, index nsamples.columns
        elif beta is scalar and nsamples is None:
            float
        elif beta is array-like and nsamples is None:
            Series, index beta
        elif beta is scalar and nsamples is int:
            Series, index range(nsamples)
        elif beta is array-like and nsamples is int:
            Series, MultiIndex columns the product of beta and range(nsamples)
        """
        logw = self.logw(nsamples, beta)
        logZ = logsumexp(logw, axis=0)
        if np.isscalar(logZ):
            return logZ
        else:
            return Series(logZ, name='logZ', index=logw.columns).squeeze()

    _logZ_function_shape ='\n' + '\n'.join(logZ.__doc__.split('\n')[1:])

    def D(self, nsamples=None, beta=None):
        """Kullback-Leibler divergence."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - logZ
        D = np.exp(logsumexp(logw-logZ, b=S, axis=0))
        if np.isscalar(D):
            return D
        else:
            return Series(D, name='D', index=logw.columns).squeeze()

    D.__doc__ += _logZ_function_shape

    def d(self, nsamples=None, beta=None):
        """Bayesian model dimensionality."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - logZ
        D = self.D(logw, beta)
        d = np.exp(logsumexp(logw-logZ, b=(S-D)**2, axis=0))*2
        if np.isscalar(d):
            return d
        else:
            return Series(d, name='d', index=logw.columns).squeeze()

    d.__doc__ += _logZ_function_shape

    def live_points(self, logL=None):
        """Get the live points within logL.

        Parameters
        ----------
        logL: float or int, optional
            Loglikelihood or iteration number to return live points.
            If not provided, return the last set of active live points.

        Returns
        -------
        live_points: MCMCSamples
            Live points at either:
                - contour logL (if input is float)
                - ith iteration (if input is integer)
                - last set of live points if no argument provided
        """
        if logL is None:
            logL = self.logL_birth.max()
        else:
            try:
                logL = float(self.logL[logL])
            except KeyError:
                pass
        i = (self.logL >= logL) & (self.logL_birth < logL)
        return MCMCSamples(self[i], weights=np.ones(i.sum()))

    def posterior_points(self, beta=1):
        """Get equally weighted posterior points at temperature beta."""
        return self.set_beta(beta).compress(-1)

    def prior_points(self, params=None):
        """Get equally weighted prior points."""
        return self.posterior_points(beta=0)

    def gui(self, params=None):
        """Construct a graphical user interface for viewing samples."""
        return RunPlotter(self, params)

    def dlogX(self, nsamples=None):
        """Compute volume of shell of loglikelihood.
        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, return mean value
            - If nsamples is integer, return nsamples from the distribution of
              values inferred by nested sampling

        Returns
        -------
        if nsamples is None:
            WeightedSeries like self
        elif nsamples is int
            WeightedDataFrame like self, columns range(nsamples)
        """
        logX = self.logX(nsamples)
        logXp = logX.shift(1, fill_value=0).to_numpy()
        logXm = logX.shift(-1, fill_value=-np.inf).to_numpy()
        dlogX = logsumexp([logXp, logXm], axis=0,
                          b=[np.ones_like(logXp), -np.ones_like(logXm)])
        dlogX -= np.log(2)
        if nsamples is None:
            dlogX = WeightedSeries(dlogX, self.index, weights=self.weights)
            dlogX.name = 'dlogX'
        else:
            dlogX = WeightedDataFrame(dlogX, self.index, weights=self.weights)
            dlogX.columns.name = logX.columns.name
        return dlogX

    def importance_sample(self, logL_new, action='add', inplace=False):
        """Perform importance re-weighting on the log-likelihood.

        Parameters
        ----------
        logL_new: np.array
            New log-likelihood values. Should have the same shape as `logL`.

        action: str, optional
            Can be any of {'add', 'replace', 'mask'}.
                * add: Add the new `logL_new` to the current `logL`.
                * replace: Replace the current `logL` with the new `logL_new`.
                * mask: treat `logL_new` as a boolean mask and only keep the
                        corresponding (True) samples.
            default: 'add'

        inplace: bool, optional
            Indicates whether to modify the existing array, or return a new
            frame with importance sampling applied.
            default: False

        Returns
        -------
        samples: NestedSamples
            Importance re-weighted samples.
        """
        samples = super().importance_sample(logL_new, action=action)
        samples = samples[samples.logL > samples.logL_birth].recompute()
        return modify_inplace(self, samples, inplace)

    def recompute(self, logL_birth=None, inplace=False):
        """Re-calculate the nested sampling contours and live points.

        Parameters
        ----------
        logL_birth: array-like or int, optional
            array-like: the birth contours.
            int: the number of live points.
            default: use the existing birth contours to compute nlive

        inplace: bool, optional
            Indicates whether to modify the existing array, or return a new
            frame with contours resorted and nlive recomputed
            default: False
        """
        samples = self.copy()

        if is_int(logL_birth):
            nlive = logL_birth
            samples.sort_values('logL', inplace=True)
            samples.reset_index(drop=True, inplace=True)
            samples['nlive'] = nlive
            descending = np.arange(nlive, 0, -1)
            samples.loc[len(samples)-nlive:, 'nlive'] = descending
        else:
            if logL_birth is not None:
                samples['logL_birth'] = logL_birth
                samples.tex['logL_birth'] = r'$\ln\mathcal{L}_{\rm birth}$'

            if 'logL_birth' not in samples:
                raise RuntimeError("Cannot recompute run without "
                                   "birth contours logL_birth.")

            invalid = samples.logL <= samples.logL_birth
            n_bad = invalid.sum()
            n_equal = (samples.logL == samples.logL_birth).sum()
            if n_bad:
                warnings.warn("%i out of %i samples have logL <= logL_birth,"
                              "\n%i of which have logL == logL_birth."
                              "\nThis may just indicate numerical rounding "
                              "errors at the peak of the likelihood, but "
                              "further investigation of the chains files is "
                              "recommended."
                              "\nDropping the invalid samples." %
                              (n_bad, len(samples), n_equal),
                              RuntimeWarning)
                samples = samples[~invalid].reset_index(drop=True)

            samples.sort_values('logL', inplace=True)
            samples.reset_index(drop=True, inplace=True)
            samples['nlive'] = compute_nlive(samples.logL, samples.logL_birth)

        samples.tex['nlive'] = r'$n_{\rm live}$'
        samples.beta = samples._beta

        if np.any(pandas.isna(samples.logL)):
            warnings.warn("NaN encountered in logL. If this is unexpected, you"
                          " should investigate why your likelihood is throwing"
                          " NaNs. Dropping these samples at prior level",
                          RuntimeWarning)
            samples = samples[samples.logL.notna()].recompute()

        return modify_inplace(self, samples, inplace)

    def _compute_insertion_indexes(self):
        logL = self.logL.to_numpy()
        logL_birth = self.logL_birth.to_numpy()
        self['insertion'] = compute_insertion_indexes(logL, logL_birth)

    _metadata = MCMCSamples._metadata + ['_beta']

    @property
    def _constructor(self):
        return NestedSamples


def merge_nested_samples(runs):
    """Merge one or more nested sampling runs.

    Parameters
    ----------
    runs: list(NestedSamples)
        List or array-like of one or more nested sampling runs.
        If only a single run is provided, this recalculates the live points and
        as such can be used for masked runs.

    Returns
    -------
    samples: NestedSamples
        Merged run.
    """
    merge = pandas.concat(runs, ignore_index=True)
    merge.tex = {key: val for r in runs for key, val in r.tex.items()}
    return merge.recompute()


def merge_samples_weighted(samples, weights=None, label=None):
    r"""Merge sets of samples with weights.

    Combine two (or more) samples so the new PDF is
    P(x|new) = weight_A P(x|A) + weight_B P(x|B).
    The number of samples and internal weights do not affect the result.

    Parameters
    ----------
    samples: list(NestedSamples) or list(MCMCSamples)
        List or array-like of one or more MCMC or nested sampling runs.

    weights: list(double) or None
        Weight for each run in samples (normalized internally).
        Can be omitted if samples are NestedSamples,
        then exp(logZ) is used as weight.

    label: str or None
        Label for the new samples. Default: None

    Returns
    -------
    new_samples: MCMCSamples
        Merged (weighted) run.
    """
    if not (isinstance(samples, Sequence) or
            isinstance(samples, Series)):
        raise TypeError("samples must be a list of samples "
                        "(Sequence or pandas.Series)")

    mcmc_samples = copy.deepcopy([MCMCSamples(s) for s in samples])
    if weights is None:
        try:
            logZs = np.array(copy.deepcopy([s.logZ() for s in samples]))
        except AttributeError:
            raise ValueError("If samples includes MCMCSamples "
                             "then weights must be given.")
        # Subtract logsumexp to avoid numerical issues (similar to max(logZs))
        logZs -= logsumexp(logZs)
        weights = np.exp(logZs)
    else:
        if len(weights) != len(samples):
            raise ValueError("samples and weights must have the same length,"
                             "each weight is for a whole sample. Currently",
                             len(samples), len(weights))

    new_samples = MCMCSamples()
    for s, w in zip(mcmc_samples, weights):
        # Normalize the given weights
        new_weights = s.weights / s.weights.sum() * w/np.sum(weights)
        s = MCMCSamples(s, weights=new_weights)
        new_samples = new_samples.append(s)

    new_samples.weights /= new_samples.weights.max()

    new_samples.label = label
    # Copy tex, if different values for same key exist, the last one is used.
    new_samples.tex = {key: val for s in samples for key, val in s.tex.items()}

    return new_samples
