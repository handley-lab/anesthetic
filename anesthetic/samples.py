"""Main classes for the anesthetic module.

- ``Samples``
- ``MCMCSamples``
- ``NestedSamples``
"""
import numpy as np
import pandas
import copy
import warnings
from pandas import MultiIndex, Series
from collections.abc import Sequence
from anesthetic.utils import (compute_nlive, compute_insertion_indexes,
                              is_int, logsumexp)
from anesthetic.gui.plot import RunPlotter
from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from anesthetic.labelled_pandas import LabelledDataFrame, LabelledSeries
from pandas.core.accessor import CachedAccessor
from anesthetic.plot import (make_1d_axes, make_2d_axes,
                             AxesSeries, AxesDataFrame)
import anesthetic.weighted_pandas
from anesthetic.plotting import PlotAccessor
anesthetic.weighted_pandas._WeightedObject.plot =\
    CachedAccessor("plot", PlotAccessor)


class WeightedLabelledDataFrame(WeightedDataFrame, LabelledDataFrame):
    """DataFrame with weights and labels."""

    _metadata = WeightedDataFrame._metadata + LabelledDataFrame._metadata

    def __init__(self, *args, **kwargs):
        labels = kwargs.pop('labels', None)
        if not hasattr(self, '_labels'):
            self._labels = ('weights', 'labels')
        super().__init__(*args, **kwargs)
        if labels is not None:
            if isinstance(labels, dict):
                labels = [labels.get(p, '') for p in self]
            self.set_labels(labels, inplace=True)

    def islabelled(self, axis=1):
        """Search for existence of labels."""
        return super().islabelled(axis=axis)

    def get_labels(self, axis=1):
        """Retrieve labels from an axis."""
        return super().get_labels(axis=axis)

    def get_labels_map(self, axis=1):
        """Retrieve mapping from paramnames to labels from an axis."""
        return super().get_labels_map(axis=axis)

    def get_label(self, param, axis=1):
        """Retrieve mapping from paramnames to labels from an axis."""
        return super().get_label(param, axis=axis)

    def set_label(self, param, value, axis=1):
        """Set a specific label to a specific value on an axis."""
        return super().set_label(param, value, axis=axis, inplace=True)

    def drop_labels(self, axis=1):
        """Drop the labels from an axis if present."""
        return super().drop_labels(axis)

    def set_labels(self, labels, axis=1, inplace=False, level=None):
        """Set labels along an axis."""
        return super().set_labels(labels, axis=axis,
                                  inplace=inplace, level=level)

    @property
    def _constructor(self):
        return WeightedLabelledDataFrame

    @property
    def _constructor_sliced(self):
        return WeightedLabelledSeries


class WeightedLabelledSeries(WeightedSeries, LabelledSeries):
    """Series with weights and labels."""

    _metadata = WeightedSeries._metadata + LabelledSeries._metadata

    def __init__(self, *args, **kwargs):
        if not hasattr(self, '_labels'):
            self._labels = ('weights', 'labels')
        super().__init__(*args, **kwargs)

    def set_label(self, param, value, axis=0):
        """Set a specific label to a specific value."""
        return super().set_label(param, value, axis=axis, inplace=True)

    @property
    def _constructor(self):
        return WeightedLabelledSeries

    @property
    def _constructor_expanddim(self):
        return WeightedLabelledDataFrame


class Samples(WeightedLabelledDataFrame):
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

    labels: dict or array-like
        mapping from columns to plotting labels

    label: str
        Legend label

    logzero: float
        The threshold for `log(0)` values assigned to rejected sample points.
        Anything equal or below this value is set to `-np.inf`.
        default: -1e30

    """

    _metadata = WeightedLabelledDataFrame._metadata + ['label']

    def __init__(self, *args, **kwargs):
        logzero = kwargs.pop('logzero', -1e30)
        logL = kwargs.pop('logL', None)
        if logL is not None:
            logL = np.array(logL)
            logL = np.where(logL <= logzero, -np.inf, logL)
        self.label = kwargs.pop('label', None)

        super().__init__(*args, **kwargs)

        if logL is not None:
            self['logL'] = logL
            if self.islabelled(axis=1):
                self.set_label('logL', r'$\ln\mathcal{L}$')

    @property
    def _constructor(self):
        return Samples

    plot = CachedAccessor("plot", PlotAccessor)

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

        kind: str, default='kde_1d'
            What kind of plots to produce. Alongside the usual pandas options
            {'hist', 'box', 'kde', 'density'}, anesthetic also provides
            {'hist_1d', 'kde_1d', 'fastkde_1d'}.
            Warning -- while the other pandas plotting options
            {'line', 'bar', 'barh', 'area', 'pie'} are also accessible, these
            can be hard to interpret/expensive for Samples, MCMCSamples, or
            NestedSamples.

        Returns
        -------
        axes: pandas.Series of matplotlib.axes.Axes
            Pandas array of axes objects

        """
        if not isinstance(axes, AxesSeries):
            _, axes = make_1d_axes(axes, labels=self.get_labels_map())

        kwargs['kind'] = kwargs.get('kind', 'kde_1d')
        kwargs['label'] = kwargs.get('label', self.label)

        for x, ax in axes.iteritems():
            if x in self and kwargs['kind'] is not None:
                xlabel = self.get_label(x)
                self[x].plot(ax=ax, xlabel=xlabel,
                             *args, **kwargs)
                ax.set_xlabel(xlabel)
            else:
                ax.plot([], [])

        return axes

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
                - 'hist_1d': 1d histograms down the diagonal
                - 'hist_2d': 2d histograms in lower triangle
                - 'hist': 1d & 2d histograms in lower & diagonal

            Feel free to add your own to this list!
            Default:
            {'diagonal': 'kde_1d', 'lower': 'kde_2d', 'upper':'scatter_2d'}

        diagonal_kwargs, lower_kwargs, upper_kwargs: dict, optional
            kwargs for the diagonal (1D)/lower or upper (2D) plots. This is
            useful when there is a conflict of kwargs for different kinds of
            plots.  Note that any kwargs directly passed to plot_2d will
            overwrite any kwarg with the same key passed to <sub>_kwargs.
            Default: {}

        Returns
        -------
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

        for pos in local_kwargs:
            local_kwargs[pos].update(kwargs)

        if not isinstance(axes, AxesDataFrame):
            _, axes = make_2d_axes(axes, labels=self.get_labels(),
                                   upper=('upper' in kind),
                                   lower=('lower' in kind),
                                   diagonal=('diagonal' in kind))

        for y, row in axes.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    pos = ax.position
                    lkwargs = local_kwargs.get(pos, {})
                    lkwargs['kind'] = kind.get(pos, None)
                    if x in self and y in self and lkwargs['kind'] is not None:
                        xlabel = self.get_label(x)
                        ylabel = self.get_label(y)
                        if x == y:
                            self[x].plot(ax=ax.twin, xlabel=xlabel,
                                         *args, **lkwargs)
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel(ylabel)
                        else:
                            self.plot(x, y, ax=ax, xlabel=xlabel,
                                      ylabel=ylabel, *args, **lkwargs)
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel(ylabel)
                    else:
                        if x == y:
                            ax.twin.plot([], [])
                        else:
                            ax.plot([], [])

        return axes

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

        Spam, ham and eggs.

        Parameters
        ----------
        logL_new: np.array
            New log-likelihood values. Should have the same shape as `logL`.

        action: str, default='add'
            Can be any of {'add', 'replace', 'mask'}.

            * add: Add the new `logL_new` to the current `logL`.
            * replace: Replace the current `logL` with the new `logL_new`.
            * mask: treat `logL_new` as a boolean mask and only keep the
              corresponding (True) samples.

        inplace: bool, default=False
            Indicates whether to modify the existing array, or return a new
            frame with importance sampling applied.

        Returns
        -------
        samples: Samples/MCMCSamples/NestedSamples
            Importance re-weighted samples.

        """
        if inplace:
            samples = self
        else:
            samples = self.copy()

        if action == 'add':
            new_weights = samples.get_weights()
            new_weights *= np.exp(logL_new - logL_new.max())
            samples.set_weights(new_weights, inplace=True)
            samples.logL += logL_new
        elif action == 'replace':
            logL_new2 = logL_new - samples.logL
            new_weights = samples.get_weights()
            new_weights *= np.exp(logL_new2 - logL_new2.max())
            samples.set_weights(new_weights, inplace=True)
            samples.logL = logL_new
        elif action == 'mask':
            samples = samples[logL_new]
        else:
            raise NotImplementedError("`action` needs to be one of "
                                      "{'add', 'replace', 'mask'}, but '%s' "
                                      "was requested." % action)

        if inplace:
            self._update_inplace(samples)
        else:
            return samples.__finalize__(self, "importance_sample")


class MCMCSamples(Samples):
    """Storage and plotting tools for MCMC samples.

    Any new functionality specific to MCMC (e.g. convergence criteria etc)
    should be put here.

    Parameters
    ----------
    root: str, optional
        root for reading chains from file. Overrides all other arguments.

    data: np.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns: array-like
        reference names of parameters

    weights: np.array
        weights of samples.

    logL: np.array
        loglikelihoods of samples.

    labels: dict or array-like
        mapping from columns to plotting labels

    label: str
        Legend label

    logzero: float
        The threshold for `log(0)` values assigned to rejected sample points.
        Anything equal or below this value is set to `-np.inf`.
        default: -1e30
    """

    _metadata = Samples._metadata + ['root']

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', None)
        super().__init__(*args, **kwargs)
        self.root = root

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

    labels: dict
        optional mapping from column names to plot labels

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
        self.root = kwargs.pop('root', None)
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
        self.set_weights(np.exp(logw - logw.max()), inplace=True)

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
            samples = self._constructor_sliced(index=self.columns[:0],
                                               dtype=float)
        else:
            samples = WeightedLabelledDataFrame(index=logw.columns,
                                                columns=self.columns[:0])
        samples['logZ'] = self.logZ(logw)
        samples.set_label('logZ', r'$\ln\mathcal{Z}$')
        w = np.exp(logw-samples['logZ'])

        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - samples.logZ

        samples['D_KL'] = (S*w).sum()
        samples.set_label('D_KL', r'$\mathcal{D}_\mathrm{KL}$')

        samples['d_G'] = ((S-samples.D_KL)**2*w).sum()
        samples.set_label('d_G', r'$d_\mathrm{G}$')

        samples['logL_P'] = samples['logZ'] + samples['D_KL']
        samples.set_label('logL_P',
                          r'$\langle\ln\mathcal{L}\rangle_\mathcal{P}$')
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
        else:
            r = np.log(np.random.rand(len(self), nsamples))
            w = self.get_weights()
            r = self.nlive._constructor_expanddim(r, self.index, weights=w)
            t = r.divide(self.nlive, axis=0)
            t.columns.name = 'samples'
        logX = t.cumsum()
        logX.name = 'logX'
        return logX

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
        logXp = logX.shift(1, fill_value=0)
        logXm = logX.shift(-1, fill_value=-np.inf)
        dlogX = np.log(1 - np.exp(logXm-logXp)) + logXp - np.log(2)
        dlogX.name = 'dlogX'

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
        logL = self.logL
        if np.isscalar(beta):
            betalogL = beta * logL
            betalogL.name = 'betalogL'
        else:
            betalogL = logL._constructor_expanddim(np.outer(self.logL, beta),
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
            dlogX = dlogX.reindex(columns=cols, level='samples')
            betalogL = betalogL.reindex(columns=cols, level='beta')
            logw = betalogL+dlogX
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
            return logw._constructor_sliced(logZ, name='logZ',
                                            index=logw.columns).squeeze()

    _logZ_function_shape = '\n' + '\n'.join(logZ.__doc__.split('\n')[1:])

    def D_KL(self, nsamples=None, beta=None):
        """Kullback-Leibler divergence."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - logZ
        w = np.exp(logw-logZ)
        D_KL = (S*w).sum()
        if np.isscalar(D_KL):
            return D_KL
        else:
            return self._constructor_sliced(D_KL, name='D_KL',
                                            index=logw.columns).squeeze()

    D_KL.__doc__ += _logZ_function_shape

    def d_G(self, nsamples=None, beta=None):
        """Bayesian model dimensionality."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - logZ
        w = np.exp(logw-logZ)
        D_KL = (S*w).sum()
        d_G = ((S-D_KL)**2*w).sum()
        if np.isscalar(d_G):
            return d_G
        else:
            return self._constructor_sliced(d_G, name='d_G',
                                            index=logw.columns).squeeze()

    d_G.__doc__ += _logZ_function_shape

    def logL_P(self, nsamples=None, beta=None):
        """Posterior averaged loglikelihood."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        betalogL = (logw*0).add(betalogL, axis=0)
        w = np.exp(logw-logZ)
        logL_P = (betalogL*w).sum()
        if np.isscalar(logL_P):
            return logL_P
        else:
            return self._constructor_sliced(logL_P, name='logL_P',
                                            index=logw.columns).squeeze()

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
        return Samples(self[i]).set_weights(None)

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
        if inplace:
            self._update_inplace(samples)
        else:
            return samples.__finalize__(self, "importance_sample")

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
        if inplace:
            samples = self
        else:
            samples = self.copy()

        nlive_label = r'$n_\mathrm{live}$'
        if is_int(logL_birth):
            nlive = logL_birth
            samples.sort_values('logL', inplace=True)
            samples.reset_index(drop=True, inplace=True)
            n = np.ones(len(self), int) * nlive
            n[-nlive:] = np.arange(nlive, 0, -1)
            samples['nlive', nlive_label] = n
        else:
            if logL_birth is not None:
                label = r'$\ln\mathcal{L}_\mathrm{birth}$'
                samples['logL_birth'] = logL_birth
                if self.islabelled():
                    samples.set_label('logL_birth', label)

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
            nlive = compute_nlive(samples.logL, samples.logL_birth)
            samples['nlive'] = nlive
            if self.islabelled():
                samples.set_label('nlive', nlive_label)

        samples.beta = samples._beta

        if np.any(pandas.isna(samples.logL)):
            warnings.warn("NaN encountered in logL. If this is unexpected, you"
                          " should investigate why your likelihood is throwing"
                          " NaNs. Dropping these samples at prior level",
                          RuntimeWarning)
            samples = samples[samples.logL.notna()].recompute()

        if inplace:
            self._update_inplace(samples)
        else:
            return samples.__finalize__(self, "recompute")


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

    new_samples = []
    for s, w in zip(mcmc_samples, weights):
        # Normalize the given weights
        new_weights = s.get_weights() / s.get_weights().sum()
        new_weights *= w/np.sum(weights)
        s = Samples(s, weights=new_weights)
        new_samples.append(s)

    new_samples = pandas.concat(new_samples)

    new_weights = new_samples.get_weights()
    new_weights /= new_weights.max()
    new_samples.set_weights(new_weights, inplace=True)

    new_samples.label = label

    return new_samples
