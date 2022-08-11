"""Main classes for the anesthetic module.

- ``Samples``
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
from anesthetic.read.samplereader import SampleReader
from anesthetic.utils import (compute_nlive, compute_insertion_indexes,
                              is_int, logsumexp, modify_inplace)
from anesthetic.gui.plot import RunPlotter
from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from pandas.core.accessor import CachedAccessor
from anesthetic.plot import make_1d_axes, make_2d_axes
import anesthetic.weighted_pandas
from anesthetic.plotting import PlotAccessor
anesthetic.weighted_pandas._WeightedObject.plot =\
    CachedAccessor("plot", PlotAccessor)
anesthetic.weighted_pandas._WeightedObject.plot =\
    CachedAccessor("plot", PlotAccessor)


class Samples(WeightedDataFrame):
    """Storage and plotting tools for general samples.

    Extends the pandas.DataFrame by providing plotting methods and
    standardising sample storage.

    Example plotting commands include
        - ``samples.plot_1d(['paramA', 'paramB'])``
        - ``samples.plot_2d(['paramA', 'paramB'])``
        - ``samples.plot_2d([['paramA', 'paramB'], ['paramC', 'paramD']])``

    Parameters
    ----------
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

    """

    _metadata = WeightedDataFrame._metadata + ['tex', 'limits', 'label']

    def __init__(self, *args, **kwargs):
        logzero = kwargs.pop('logzero', -1e30)
        logL = kwargs.pop('logL', None)
        if logL is not None:
            logL = np.array(logL)
            logL = np.where(logL <= logzero, -np.inf, logL)
        self.tex = kwargs.pop('tex', {})
        self.limits = kwargs.pop('limits', {})
        self.label = kwargs.pop('label', None)
        super().__init__(*args, **kwargs)

        if logL is not None:
            self['logL'] = logL
            self.tex['logL'] = r'$\ln\mathcal{L}$'

        self._set_automatic_limits()

    @property
    def _constructor(self):
        return Samples

    def _reload_data(self):
        self.__init__(root=self.root)
        return self

    def _limits(self, paramname):
        limits = self.limits.get(paramname, (None, None))
        if limits[0] == limits[1]:
            limits = (None, None)
        return limits

    def _set_automatic_limits(self):
        """Set all unassigned limits to min and max of sample."""
        for param in self.columns:
            if param not in self.limits:
                self.limits[param] = (self[param].min(), self[param].max())

    def copy(self, deep=True):
        """Copy which also includes mutable metadata."""
        new = super().copy(deep)
        if deep:
            new.tex = copy.deepcopy(self.tex)
            new.limits = copy.deepcopy(self.limits)
        return new

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

        kind: str, optional
            What kind of plots to produce. Alongside the usual pandas options
            {'hist', 'box', 'kde', 'density'}, anesthetic also provides
            {'hist_1d', 'kde_1d', 'fastkde_1d'}.
            Warning -- while the other pandas plotting options
            {'line', 'bar', 'barh', 'area', 'pie'} are also accessible, these
            can be hard to interpret/expensive for Samples, MCMCSamples, or
            NestedSamples.
            Default kde_1d

        Returns
        -------
        fig: matplotlib.figure.Figure
            New or original (if supplied) figure object

        axes: pandas.Series of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        self._set_automatic_limits()

        if not isinstance(axes, Series):
            fig, axes = make_1d_axes(axes, tex=self.tex)
        else:
            fig = axes.bfill().to_numpy().flatten()[0].figure

        kwargs['kind'] = kwargs.get('kind', 'kde_1d')
        kwargs['label'] = kwargs.get('label', self.label)

        for x, ax in axes.iteritems():
            if x in self and kwargs['kind'] is not None:
                self[x].plot(ax=ax, *args, **kwargs)
            else:
                ax.plot([], [])

        return fig, axes

    def plot_2d(self, axes, *args, **kwargs):
        """Create an array of 2D plots.

        To avoid interfering with y-axis sharing, one-dimensional plots are
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
            this is used for creating the plot. Otherwise, a new set of axes
            are created using the list or lists of strings.

        kind/kinds: dict, optional
            What kinds of plots to produce. Dictionary takes the keys
            'diagonal' for the 1D plots and 'lower' and 'upper' for the 2D
            plots. The options for 'diagonal' are:
                - 'kde_1d'
                - 'hist_1d'
                - 'fastkde_1d'
                - 'kde'
                - 'hist'
                - 'box'
                - 'kde'
                - 'density'
            The options for 'lower' and 'upper' are:
                - 'kde_2d'
                - 'hist_2d'
                - 'scatter_2d'
                - 'fastkde_2d'
                - 'kde'
                - 'scatter'
                - 'hexbin'
            There are also a set of shortcuts provided in
            Samples.plot_2d_default_kinds:
                - 'kde_1d': 1d kde plots down the diagonal
                - 'kde_2d': 2d kde plots in lower triangle
                - 'kde': 1d & 2d kde plots in lower & diagonal
            Feel free to add your own to this list!
            Default: {'diagonal': 'kde_1d',
                      'lower': 'kde_2d',
                      'upper':'scatter_2d'}

        diagonal_kwargs, lower_kwargs, upper_kwargs: dict, optional
            kwargs for the diagonal (1D)/lower or upper (2D) plots. This is
            useful when there is a conflict of kwargs for different kinds of
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
        kind = kwargs.pop('kind', 'default')
        kind = kwargs.pop('kinds', kind)
        if isinstance(kind, str) and kind in self.plot_2d_default_kinds:
            kind = self.plot_2d_default_kinds.get(kind)
        if (not isinstance(kind, dict) or
                not set(kind.keys()) <= {'lower', 'upper', 'diagonal'}):
            raise ValueError(f"{kind} is not a valid input. `kind`/`kinds` "
                             "must be a dict mapping "
                             "{'lower','diagonal','upper'} to an allowed plot "
                             "(see `help(NestedSamples.plot2d)`), or one of "
                             "the following string shortcuts: "
                             f"{list(self.plot_2d_default_kinds.keys())}")

        local_kwargs = {pos: kwargs.pop('%s_kwargs' % pos, {})
                        for pos in ['upper', 'lower', 'diagonal']}
        kwargs['label'] = kwargs.get('label', self.label)

        self._set_automatic_limits()

        for pos in local_kwargs:
            local_kwargs[pos].update(kwargs)

        if not isinstance(axes, DataFrame):
            fig, axes = make_2d_axes(axes, tex=self.tex,
                                     upper=('upper' in kind),
                                     lower=('lower' in kind),
                                     diagonal=('diagonal' in kind))
        else:
            fig = axes.bfill().to_numpy().flatten()[0].figure

        for y, row in axes.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    pos = ax.position
                    lkwargs = local_kwargs.get(pos, {})
                    lkwargs['kind'] = kind.get(pos, None)
                    if x in self and y in self and lkwargs['kind'] is not None:
                        if x == y:
                            self[x].plot(ax=ax.twin, *args, **lkwargs)
                        else:
                            self.plot(x, y, ax=ax, *args, **lkwargs)
                    else:
                        if x == y:
                            ax.twin.plot([], [])
                        else:
                            ax.plot([], [])

        return fig, axes

    plot_2d_default_kinds = {
        'default': {'diagonal': 'kde_1d',
                    'lower': 'kde_2d',
                    'upper': 'scatter_2d'},
        'kde': {'diagonal': 'kde_1d', 'lower': 'kde_2d'},
        'kde_1d': {'diagonal': 'kde_1d'},
        'kde_2d': {'lower': 'kde_2d'},
        'hist': {'diagonal': 'hist_1d', 'lower': 'hist_2d'},
        'hist_1d': {'diagonal': 'hist_1d'},
        'hist_2d': {'lower': 'hist_2d'},
    }

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
        samples: Samples/MCMCSamples/NestedSamples
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


class MCMCSamples(Samples):
    """Storage and plotting tools for MCMC samples.

    We extend the Samples class with the additional methods:

    * ``burn_in`` parameter

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

    _metadata = Samples._metadata + ['root']

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', None)
        if root is not None:
            reader = SampleReader(root)
            if hasattr(reader, 'birth_file') or hasattr(reader, 'ev_file'):
                raise ValueError("The file root %s seems to point to a Nested "
                                 "Sampling chain. Please use NestedSamples "
                                 "instead which has the same features as "
                                 "Samples and more. MCMCSamples should be "
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
            self.root = None
            super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return MCMCSamples


class NestedSamples(Samples):
    """Storage and plotting tools for Nested Sampling samples.

    We extend the Samples class with the additional methods:

    * ``self.live_points(logL)``
    * ``self.set_beta(beta)``
    * ``self.prior()``
    * ``self.posterior_points(beta)``
    * ``self.prior_points()``
    * ``self.stats()``
    * ``self.logZ()``
    * ``self.D_KL()``
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

    _metadata = Samples._metadata + ['root', '_beta']

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
            self.root = None
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
    def _constructor(self):
        return NestedSamples

    def _compute_insertion_indexes(self):
        logL = self.logL.to_numpy()
        logL_birth = self.logL_birth.to_numpy()
        self['insertion'] = compute_insertion_indexes(logL, logL_birth)

    @property
    def beta(self):
        """Thermodynamic inverse temperature."""
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        logw = self.logw(beta=beta)
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

    def stats(self, nsamples=None, beta=None):
        """Compute Nested Sampling statistics.

        Using nested sampling we can compute:
            - logZ: the Bayesian evidence
            - D_KL: the Kullback-Leibler divergence
            - d_G: the Gaussian model dimensionality
            - logL_P: the posterior averaged loglikelihood

        (Note that all of these are available as individual functions with the
        same signature). See https://arxiv.org/abs/1903.06682 for more detail.

        In addition to point estimates nested sampling provides an error bar
        or more generally samples from a (correlated) distribution over the
        variables. Samples from this distribution can be computed by providing
        an integer nsamples.

        Nested sampling as an athermal algorithm is also capable of producing
        these as a function of inverse thermodynamic temperature beta. This is
        provided as a vectorised function. If nsamples is also provided a
        MultiIndex dataframe is generated.

        These obey Occam's razor equation: logZ = logL_P - D_KL, which splits
        a model's quality (logZ) into a goodness-of-fit (logL_P) and an
        complexity penalty (D_KL) https://arxiv.org/abs/2102.11511

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
              values inferred by nested sampling

        beta: float, array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if beta is scalar and nsamples is None:
            Series, index ['logZ', 'd_G', 'DK_L', 'logL_P']
        elif beta is scalar and nsamples is int:
            Samples, index range(nsamples),
            columns ['logZ', 'd_G', 'DK_L', 'logL_P']
        elif beta is array-like and nsamples is None:
            Samples, index beta,
            columns ['logZ', 'd_G', 'DK_L', 'logL_P']
        elif beta is array-like and nsamples is int:
            Samples, index MultiIndex the product of beta and range(nsamples)
            columns ['logZ', 'd_G', 'DK_L', 'logL_P']
        """
        logw = self.logw(nsamples, beta)
        if nsamples is None and beta is None:
            samples = Series(dtype=float)
        else:
            samples = Samples(index=logw.columns)
        samples['logZ'] = self.logZ(logw)

        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - samples.logZ

        logD_KL, s = logsumexp(logw-samples.logZ, b=S, axis=0,
                               return_sign=True)
        samples['D_KL'] = s * np.exp(logD_KL)

        logd_G_o2, s = logsumexp(logw-samples.logZ, b=(S-samples.D_KL)**2,
                                 axis=0, return_sign=True)
        samples['d_G'] = s * np.exp(logd_G_o2*2)
        samples['logL_P'] = samples['logZ'] + samples['D_KL']

        samples.tex = {'logZ': r'$\ln\mathcal{Z}$',
                       'D_KL': r'$\mathcal{D}_\mathrm{KL}$',
                       'd_G': r'$d_\mathrm{G}$',
                       'logL_P': r'$\langle\ln L\rangle_\mathcal{P}$'}
        samples.label = self.label
        return samples

    def logX(self, nsamples=None):
        """Log-Volume.

        The log of the prior volume contained within each iso-likelihood
        contour.

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
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

    def dlogX(self, nsamples=None):
        """Compute volume of shell of loglikelihood.

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
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
            dlogX = WeightedSeries(dlogX, self.index)
            dlogX.name = 'dlogX'
        else:
            dlogX = WeightedDataFrame(dlogX, self.index)
            dlogX.columns.name = logX.columns.name
        return dlogX

    def _betalogL(self, beta=None):
        """Log(L**beta) convenience function.

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
            betalogL = WeightedSeries(beta*self.logL, self.index)
            betalogL.name = 'betalogL'
        else:
            betalogL = WeightedDataFrame(np.outer(self.logL, beta),
                                         self.index, columns=beta)
            betalogL.columns.name = 'beta'
        return betalogL

    def logw(self, nsamples=None, beta=None):
        """Log-nested sampling weight.

        The logarithm of the (unnormalised) sampling weight log(L**beta*dX).

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
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
            logw = betalogL.add(dlogX, axis=0)
        else:
            cols = MultiIndex.from_product([betalogL.columns, dlogX.columns])
            dlogX = DataFrame(dlogX).reindex(columns=cols, level='samples')
            betalogL = DataFrame(betalogL).reindex(columns=cols, level='beta')
            logw = WeightedDataFrame(betalogL+dlogX, self.index)
        return logw

    def logZ(self, nsamples=None, beta=None):
        """Log-Evidence.

        Parameters
        ----------
        nsamples: int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
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

    _logZ_function_shape = '\n' + '\n'.join(logZ.__doc__.split('\n')[1:])

    def D_KL(self, nsamples=None, beta=None):
        """Kullback-Leibler divergence."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - logZ
        logD_KL, s = logsumexp(logw-logZ, b=S, axis=0, return_sign=True)
        D_KL = s * np.exp(logD_KL)
        if np.isscalar(D_KL):
            return D_KL
        else:
            return Series(D_KL, name='D_KL', index=logw.columns).squeeze()

    D_KL.__doc__ += _logZ_function_shape

    def d_G(self, nsamples=None, beta=None):
        """Bayesian model dimensionality."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - logZ
        D_KL = self.D_KL(logw, beta)
        logd_G_o2, s = logsumexp(logw-logZ, b=(S-D_KL)**2, axis=0,
                                 return_sign=True)
        d_G = s * np.exp(logd_G_o2*2)
        if np.isscalar(d_G):
            return d_G
        else:
            return Series(d_G, name='d_G', index=logw.columns).squeeze()

    d_G.__doc__ += _logZ_function_shape

    def logL_P(self, nsamples=None, beta=None):
        """Posterior averaged loglikelihood."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        betalogL = (logw*0).add(betalogL, axis=0)
        loglogL_P, s = logsumexp(logw-logZ, b=betalogL, axis=0,
                                 return_sign=True)
        logL_P = s * np.exp(loglogL_P)
        if np.isscalar(logL_P):
            return logL_P
        else:
            return Series(logL_P, name='logL_P', index=logw.columns).squeeze()

    logL_P.__doc__ += _logZ_function_shape

    def live_points(self, logL=None):
        """Get the live points within logL.

        Parameters
        ----------
        logL: float or int, optional
            Loglikelihood or iteration number to return live points.
            If not provided, return the last set of active live points.

        Returns
        -------
        live_points: Samples
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
        return Samples(self[i], weights=np.ones(i.sum()))

    def posterior_points(self, beta=1):
        """Get equally weighted posterior points at temperature beta."""
        return self.set_beta(beta).compress(-1)

    def prior_points(self, params=None):
        """Get equally weighted prior points."""
        return self.posterior_points(beta=0)

    def gui(self, params=None):
        """Construct a graphical user interface for viewing samples."""
        return RunPlotter(self, params)

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
                samples.tex['logL_birth'] = r'$\ln\mathcal{L}_\mathrm{birth}$'

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

        samples.tex['nlive'] = r'$n_\mathrm{live}$'
        samples.beta = samples._beta

        if np.any(pandas.isna(samples.logL)):
            warnings.warn("NaN encountered in logL. If this is unexpected, you"
                          " should investigate why your likelihood is throwing"
                          " NaNs. Dropping these samples at prior level",
                          RuntimeWarning)
            samples = samples[samples.logL.notna()].recompute()

        return modify_inplace(self, samples, inplace)


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
    new_samples: Samples
        Merged (weighted) run.
    """
    if not (isinstance(samples, Sequence) or
            isinstance(samples, Series)):
        raise TypeError("samples must be a list of samples "
                        "(Sequence or pandas.Series)")

    mcmc_samples = copy.deepcopy([Samples(s) for s in samples])
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

    new_samples = Samples()
    for s, w in zip(mcmc_samples, weights):
        # Normalize the given weights
        new_weights = s.weights / s.weights.sum() * w/np.sum(weights)
        s = Samples(s, weights=new_weights)
        new_samples = new_samples.append(s)

    new_samples.weights /= new_samples.weights.max()

    new_samples.label = label
    # Copy tex, if different values for same key exist, the last one is used.
    new_samples.tex = {key: val for s in samples for key, val in s.tex.items()}

    return new_samples
