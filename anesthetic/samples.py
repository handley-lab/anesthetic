"""Main classes for the anesthetic module.

- :class:`anesthetic.samples.Samples`
- :class:`anesthetic.samples.MCMCSamples`
- :class:`anesthetic.samples.NestedSamples`
"""
import numpy as np
import scipy
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
from anesthetic.plot import (make_1d_axes, make_2d_axes,
                             AxesSeries, AxesDataFrame)
from anesthetic.utils import adjust_docstrings


class WeightedLabelledDataFrame(WeightedDataFrame, LabelledDataFrame):
    """:class:`pandas.DataFrame` with weights and labels."""

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

    def get_labels_map(self, axis=1, fill=True):
        """Retrieve mapping from paramnames to labels from an axis."""
        return super().get_labels_map(axis=axis, fill=fill)

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

    Extends the :class:`pandas.DataFrame` by providing plotting methods and
    standardising sample storage.

    Example plotting commands include
        - ``samples.plot_1d(['paramA', 'paramB'])``
        - ``samples.plot_2d(['paramA', 'paramB'])``
        - ``samples.plot_2d([['paramA', 'paramB'], ['paramC', 'paramD']])``

    Parameters
    ----------
    data : np.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns : list(str)
        reference names of parameters

    weights : np.array
        weights of samples.

    logL : np.array
        loglikelihoods of samples.

    labels : dict or array-like
        mapping from columns to plotting labels

    label : str
        Legend label

    logzero : float, default=-1e30
        The threshold for `log(0)` values assigned to rejected sample points.
        Anything equal or below this value is set to `-np.inf`.

    """

    _metadata = WeightedLabelledDataFrame._metadata + ['label']

    def __init__(self, *args, **kwargs):
        # TODO: remove this in version >= 2.1
        if 'root' in kwargs:
            root = kwargs.pop('root')
            name = self.__class__.__name__
            raise ValueError(
                "As of anesthetic 2.0, root is no longer a keyword argument.\n"
                "To update your code, replace \n\n"
                ">>> from anesthetic import %s\n"
                ">>> %s(root=%s)\n\nwith\n\n"
                ">>> from anesthetic import read_chains\n"
                ">>> read_chains(%s)" % (name, name, root, root)
                )
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

    def plot_1d(self, axes=None, *args, **kwargs):
        """Create an array of 1D plots.

        Parameters
        ----------
        axes : plotting axes, optional
            Can be:

            * list(str) or str
            * :class:`pandas.Series` of :class:`matplotlib.axes.Axes`

            If a :class:`pandas.Series` is provided as an existing set of axes,
            then this is used for creating the plot. Otherwise, a new set of
            axes are created using the list or lists of strings.

            If not provided, then all parameters are plotted. This is intended
            for plotting a sliced array (e.g. `samples[['x0','x1]].plot_1d()`.

        kind : str, default='kde_1d'
            What kind of plots to produce. Alongside the usual pandas options
            {'hist', 'box', 'kde', 'density'}, anesthetic also provides

            * 'hist_1d': :func:`anesthetic.plot.hist_plot_1d`
            * 'kde_1d': :func:`anesthetic.plot.kde_plot_1d`
            * 'fastkde_1d': :func:`anesthetic.plot.fastkde_plot_1d`

            Warning -- while the other pandas plotting options
            {'line', 'bar', 'barh', 'area', 'pie'} are also accessible, these
            can be hard to interpret/expensive for :class:`Samples`,
            :class:`MCMCSamples`, or :class:`NestedSamples`.

        logx : list(str), optional
            Which parameters/columns to plot on a log scale.
            Needs to match if plotting on top of a pre-existing axes.

        label : str, optional
            Legend label added to each axis.

        Returns
        -------
        axes : :class:`pandas.Series` of :class:`matplotlib.axes.Axes`
            Pandas array of axes objects

        """
        # TODO: remove this in version >= 2.1
        if 'plot_type' in kwargs:
            raise ValueError(
                "You are using the anesthetic 1.0 kwarg \'plot_type\' instead "
                "of anesthetic 2.0 \'kind\'. Please update your code."
                )

        if axes is None:
            axes = self.drop_labels().columns

        if not isinstance(axes, AxesSeries):
            _, axes = make_1d_axes(axes, labels=self.get_labels_map(),
                                   logx=kwargs.pop('logx', None))
            logx = axes._logx
        else:
            logx = kwargs.pop('logx', axes._logx)
            if logx != axes._logx:
                raise ValueError(f"logx does not match the pre-existing axes."
                                 f"logx={logx}, axes._logx={axes._logx}")

        kwargs['kind'] = kwargs.get('kind', 'kde_1d')
        kwargs['label'] = kwargs.get('label', self.label)

        # TODO: remove this in version >= 2.1
        if kwargs['kind'] == 'kde':
            warnings.warn(
                "You are using \'kde\' as a plot kind. "
                "\'kde_1d\' is the appropriate keyword for anesthetic. "
                "Your plots may look odd if you use this argument."
                )
        elif kwargs['kind'] == 'hist':
            warnings.warn(
                "You are using \'hist\' as a plot kind. "
                "\'hist_1d\' is the appropriate keyword for anesthetic. "
                "Your plots may look odd if you use this argument."
                )

        for x, ax in axes.items():
            if x in self and kwargs['kind'] is not None:
                xlabel = self.get_label(x)
                self[x].plot(ax=ax, xlabel=xlabel, logx=x in logx,
                             *args, **kwargs)
                ax.set_xlabel(xlabel)
            else:
                ax.plot([], [])

        return axes

    def plot_2d(self, axes=None, *args, **kwargs):
        """Create an array of 2D plots.

        To avoid interfering with y-axis sharing, one-dimensional plots are
        created on a separate axis, which is monkey-patched onto the argument
        ax as the attribute ax.twin.

        Parameters
        ----------
        axes : plotting axes, optional
            Can be:
                - list(str) if the x and y axes are the same
                - [list(str),list(str)] if the x and y axes are different
                - :class:`pandas.DataFrame` of :class:`matplotlib.axes.Axes`

            If a :class:`pandas.DataFrame` is provided as an existing set of
            axes, then this is used for creating the plot. Otherwise, a new set
            of axes are created using the list or lists of strings.

            If not provided, then all parameters are plotted. This is intended
            for plotting a sliced array (e.g. `samples[['x0','x1]].plot_2d()`.
            It is not advisible to plot an entire frame, as it is
            computationally expensive, and liable to run into linear algebra
            errors for degenerate derived parameters.

        kind/kinds : dict, optional
            What kinds of plots to produce. Dictionary takes the keys
            'diagonal' for the 1D plots and 'lower' and 'upper' for the 2D
            plots. The options for 'diagonal' are:

                - 'kde_1d': :func:`anesthetic.plot.kde_plot_1d`
                - 'hist_1d': :func:`anesthetic.plot.hist_plot_1d`
                - 'fastkde_1d': :func:`anesthetic.plot.fastkde_plot_1d`
                - 'kde': :meth:`pandas.Series.plot.kde`
                - 'hist': :meth:`pandas.Series.plot.hist`
                - 'box': :meth:`pandas.Series.plot.box`
                - 'density': :meth:`pandas.Series.plot.density`

            The options for 'lower' and 'upper' are:

                - 'kde_2d': :func:`anesthetic.plot.kde_contour_plot_2d`
                - 'hist_2d': :func:`anesthetic.plot.hist_plot_2d`
                - 'scatter_2d': :func:`anesthetic.plot.scatter_plot_2d`
                - 'fastkde_2d': :func:`anesthetic.plot.fastkde_contour_plot_2d`
                - 'kde': :meth:`pandas.DataFrame.plot.kde`
                - 'scatter': :meth:`pandas.DataFrame.plot.scatter`
                - 'hexbin': :meth:`pandas.DataFrame.plot.hexbin`

            There are also a set of shortcuts provided in
            :attr:`plot_2d_default_kinds`:

                - 'kde_1d': 1d kde plots down the diagonal
                - 'kde_2d': 2d kde plots in lower triangle
                - 'kde': 1d & 2d kde plots in lower & diagonal
                - 'hist_1d': 1d histograms down the diagonal
                - 'hist_2d': 2d histograms in lower triangle
                - 'hist': 1d & 2d histograms in lower & diagonal

            Feel free to add your own to this list!
            Default:
            {'diagonal': 'kde_1d', 'lower': 'kde_2d', 'upper':'scatter_2d'}

        diagonal_kwargs, lower_kwargs, upper_kwargs : dict, optional
            kwargs for the diagonal (1D)/lower or upper (2D) plots. This is
            useful when there is a conflict of kwargs for different kinds of
            plots.  Note that any kwargs directly passed to plot_2d will
            overwrite any kwarg with the same key passed to <sub>_kwargs.
            Default: {}

        logx, logy : list(str), optional
            Which parameters/columns to plot on a log scale for the x-axis and
            y-axis, respectively.
            Needs to match if plotting on top of a pre-existing axes.

        label : str, optional
            Legend label added to each axis.

        Returns
        -------
        axes : :class:`pandas.DataFrame` of :class:`matplotlib.axes.Axes`
            Pandas array of axes objects

        """
        # TODO: remove this in version >= 2.1
        if 'types' in kwargs:
            raise ValueError(
                "You are using the anesthetic 1.0 kwarg \'types\' instead of "
                "anesthetic 2.0 \'kind' or \'kinds\' (synonyms). "
                "Please update your code."
                )
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

        if axes is None:
            axes = self.drop_labels().columns

        if not isinstance(axes, AxesDataFrame):
            _, axes = make_2d_axes(axes, labels=self.get_labels_map(),
                                   upper=('upper' in kind),
                                   lower=('lower' in kind),
                                   diagonal=('diagonal' in kind),
                                   logx=kwargs.pop('logx', None),
                                   logy=kwargs.pop('logy', None))
            logx = axes._logx
            logy = axes._logy
        else:
            logx = kwargs.pop('logx', axes._logx)
            logy = kwargs.pop('logy', axes._logy)
            if logx != axes._logx or logy != axes._logy:
                raise ValueError(f"logx or logy not matching existing axes:"
                                 f"logx={logx}, axes._logx={axes._logx}"
                                 f"logy={logy}, axes._logy={axes._logy}")

        local_kwargs = {pos: kwargs.pop('%s_kwargs' % pos, {})
                        for pos in ['upper', 'lower', 'diagonal']}
        kwargs['label'] = kwargs.get('label', self.label)

        for pos in local_kwargs:
            local_kwargs[pos].update(kwargs)

        for y, row in axes.iterrows():
            for x, ax in row.items():
                if ax is not None:
                    pos = ax.position
                    lkwargs = local_kwargs.get(pos, {})
                    lkwargs['kind'] = kind.get(pos, None)
                    # TODO: remove this in version >= 2.1
                    if lkwargs['kind'] == 'kde':
                        warnings.warn(
                            "You are using \'kde\' as a plot kind. "
                            "\'kde_1d\' and \'kde_2d\' are the appropriate "
                            "keywords for anesthetic. Your plots may look "
                            "odd if you use this argument."
                            )
                    elif lkwargs['kind'] == 'hist':
                        warnings.warn(
                            "You are using \'hist\' as a plot kind. "
                            "\'hist_1d\' and \'hist_2d\' are the appropriate "
                            "keywords for anesthetic. Your plots may look "
                            "odd if you use this argument."
                            )
                    if x in self and y in self and lkwargs['kind'] is not None:
                        xlabel = self.get_label(x)
                        ylabel = self.get_label(y)
                        if x == y:
                            self[x].plot(ax=ax.twin, xlabel=xlabel,
                                         logx=x in logx,
                                         *args, **lkwargs)
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel(ylabel)
                        else:
                            self.plot(x, y, ax=ax, xlabel=xlabel,
                                      logx=x in logx, logy=y in logy,
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
        'fastkde': {'diagonal': 'fastkde_1d', 'lower': 'fastkde_2d'},
        'hist': {'diagonal': 'hist_1d', 'lower': 'hist_2d'},
        'hist_1d': {'diagonal': 'hist_1d'},
        'hist_2d': {'lower': 'hist_2d'},
    }

    def importance_sample(self, logL_new, action='add', inplace=False):
        """Perform importance re-weighting on the log-likelihood.

        Parameters
        ----------
        logL_new : np.array
            New log-likelihood values. Should have the same shape as `logL`.

        action : str, default='add'
            Can be any of {'add', 'replace', 'mask'}.

            * add: Add the new `logL_new` to the current `logL`.
            * replace: Replace the current `logL` with the new `logL_new`.
            * mask: treat `logL_new` as a boolean mask and only keep the
              corresponding (True) samples.

        inplace : bool, default=False
            Indicates whether to modify the existing array, or return a new
            frame with importance sampling applied.

        Returns
        -------
        samples : :class:`Samples`/:class:`MCMCSamples`/:class:`NestedSamples`
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

    # TODO: remove this in version >= 2.1
    @property
    def tex(self):
        # noqa: disable=D102
        raise NotImplementedError(
            "This is anesthetic 1.0 syntax. You need to update, e.g.\n"
            "samples.tex[label] = tex        # anesthetic 1.0\n"
            "samples.set_label(label, tex)   # anesthetic 2.0\n\n"
            "tex = samples.tex[label]        # anesthetic 1.0\n"
            "tex = samples.get_label(label)  # anesthetic 2.0"
            )

    def to_hdf(self, path_or_buf, key, *args, **kwargs):  # noqa: D102
        import anesthetic.read.hdf
        return anesthetic.read.hdf.to_hdf(path_or_buf, key, self,
                                          *args, **kwargs)


class MCMCSamples(Samples):
    """Storage and plotting tools for MCMC samples.

    Any new functionality specific to MCMC (e.g. convergence criteria etc.)
    should be put here.

    Parameters
    ----------
    data : np.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns : array-like
        reference names of parameters

    weights : np.array
        weights of samples.

    logL : np.array
        loglikelihoods of samples.

    labels : dict or array-like
        mapping from columns to plotting labels

    label : str
        Legend label

    logzero : float, default=-1e30
        The threshold for `log(0)` values assigned to rejected sample points.
        Anything equal or below this value is set to `-np.inf`.

    """

    _metadata = Samples._metadata + ['root']

    @property
    def _constructor(self):
        return MCMCSamples

    def remove_burn_in(self, burn_in, reset_index=False, inplace=False):
        """Remove burn-in samples from each MCMC chain.

        Parameters
        ----------
        burn_in : int or float or array_like
            Fraction or number of samples to remove or keep:

            * ``if 0 < burn_in < 1``: remove first fraction of samples
            * ``elif 1 < burn_in``: remove first number of samples
            * ``elif -1 < burn_in < 0``: keep last fraction of samples
            * ``elif burn_in < -1``: keep last number of samples
            * ``elif type(burn_in)==list``: different burn-in for each chain

        reset_index : bool, default=False
            Whether to reset the index counter to start at zero or not.

        inplace : bool, default=False
            Indicates whether to modify the existing array or return a copy.

        """
        chains = self.groupby(('chain', '$n_\\mathrm{chain}$'), sort=False,
                              group_keys=False)
        nchains = chains.ngroups
        if isinstance(burn_in, (int, float)):
            ndrop = np.full(nchains, burn_in)
        elif isinstance(burn_in, (list, tuple, np.ndarray)) \
                and len(burn_in) == nchains:
            ndrop = np.array(burn_in)
        else:
            raise ValueError("`burn_in` has to be a scalar or an array of "
                             "length matching the number of chains "
                             "`nchains=%d`. However, you provided "
                             "`burn_in=%s`" % (nchains, burn_in))
        if np.all(np.abs(ndrop) < 1):
            nsamples = chains.count().iloc[:, 0].to_numpy()
            ndrop = ndrop * nsamples
        ndrop = ndrop.astype(int)
        data = self.drop(chains.apply(lambda g: g.head(ndrop[g.name-1])).index,
                         inplace=inplace)
        if reset_index:
            data = data.reset_index(drop=True, inplace=inplace)
        return data

    def Gelman_Rubin(self, params=None, per_param=False):
        """Gelman--Rubin convergence statistic of multiple MCMC chains.

        Determine the Gelman--Rubin convergence statistic ``R-1`` by computing
        and comparing the within-chain variance and the between-chain variance.
        This follows the routine as outlined in
        `Lewis (2013), section IV.A. <https://arxiv.org/abs/1304.4473>`_

        Note that this requires more than one chain. To circumvent this, you
        could overwrite the ``'chain'`` column, splitting the samples into two
        or more sets.

        Parameters
        ----------
        params : list(str)
            List of column names (i.e. parameters) to be included in the
            convergence calculation.
            Default: all parameters (except those parameters that contain
            'prior', 'chi2', or 'logL' in their names)

        per_param : bool or str, default=False
            Whether to return the per-parameter convergence statistic ``R-1``.

            * If ``False``: returns only the total convergence statistic.
            * If ``True``: returns the total convergence statistic and the
              per-parameter convergence statistic.
            * If ``'par'``: returns only the per-parameter convergence
              statistic.
            * If ``'cov'``: returns only the per-parameter covariant
              convergence statistic.
            * If ``'all'``: returns the total convergence statistic and the
              per-parameter covariant convergence statistic.

        Returns
        -------
        Rminus1 : float
            Total Gelman--Rubin convergence statistic ``R-1``. The smaller, the
            better converged. Aiming for ``Rminus1~0.01`` should normally work
            well.
        Rminus1_par : :class:`pandas.DataFrame`
            Per-parameter Gelman--Rubin convergence statistic.
        Rminus1_cov : :class:`pandas.DataFrame`
            Per-parameter covariant Gelman--Rubin convergence statistic.

        """
        self.columns.set_names(['params', 'labels'], inplace=True)
        if params is None:
            params = [key for key in self.columns.get_level_values('params')
                      if 'prior' not in key
                      and 'chi2' not in key
                      and 'logL' not in key
                      and 'chain' not in key]
        chains = self[params+['chain']].groupby(
                ('chain', '$n_\\mathrm{chain}$'), sort=False,
        )
        nchains = chains.ngroups

        # Within chain variance ``W``
        # (average variance within each chain):
        W = chains.cov().groupby(level=('params', 'labels'), sort=False).mean()
        # Between-chain variance ``B``
        # (variance of the chain means):
        B = chains.mean().drop_weights().cov()
        # We don't weight `B` with the effective number of samples (sum of the
        # weights), here, because we want to notice outliers from shorter
        # chains.
        # In order to be conservative, we generally want to underestimate `W`
        # and overestimate `B`, since `W` goes in the denominator and `B` in
        # the numerator of the Gelman--Rubin statistic `Rminus1`.

        try:
            # note: scipy's cholesky returns U, not L
            invU = np.linalg.inv(scipy.linalg.cholesky(W))
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Make sure you do not have linearly dependent parameters, "
                "e.g. having both `As` and `A=1e9*As` causes trouble.") from e
        D = np.linalg.eigvalsh(invU.T @ ((nchains+1)/nchains * B) @ invU)
        # The factor of `(nchains+1)/nchains` accounts for the additional
        # uncertainty from using a finite number of chains.
        Rminus1_tot = np.max(np.abs(D))
        if per_param is False:
            return Rminus1_tot
        Rminus1 = (nchains + 1) / nchains * B / W.drop_weights()
        Rminus1_par = pandas.DataFrame(np.diag(Rminus1), index=B.columns,
                                       columns=['R-1'])
        if per_param is True:
            return Rminus1_tot, Rminus1_par
        if per_param == 'par':
            return Rminus1_par
        Rminus1_cov = pandas.DataFrame(Rminus1, index=B.columns,
                                       columns=W.columns)
        if per_param == 'cov':
            return Rminus1_cov
        return Rminus1_tot, Rminus1_cov


class NestedSamples(Samples):
    """Storage and plotting tools for Nested Sampling samples.

    We extend the :class:`Samples` class with the additional methods:

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
    data : np.array
        Coordinates of samples. shape = (nsamples, ndims).

    columns : list(str)
        reference names of parameters

    logL : np.array
        loglikelihoods of samples.

    logL_birth : np.array or int
        birth loglikelihoods, or number of live points.

    labels : dict
        optional mapping from column names to plot labels

    label : str
        Legend label
        default: basename of root

    beta : float
        thermodynamic inverse temperature
        default: 1.

    logzero : float
        The threshold for `log(0)` values assigned to rejected sample points.
        Anything equal or below this value is set to `-np.inf`.
        default: -1e30

    """

    _metadata = Samples._metadata + ['root', '_beta']

    def __init__(self, *args, **kwargs):
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
        beta : float
            Inverse temperature to set.
            (``beta=0`` corresponds to the prior distribution.)

        inplace : bool, default=False
            Indicates whether to modify the existing array, or return a copy
            with the inverse temperature changed.

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

    # TODO: remove this in version >= 2.1
    def ns_output(self, *args, **kwargs):
        # noqa: disable=D102
        raise NotImplementedError(
            "This is anesthetic 1.0 syntax. You need to update, e.g.\n"
            "samples.ns_output(1000)  # anesthetic 1.0\n"
            "samples.stats(1000)      # anesthetic 2.0\n\n"
            "Check out the new temperature functionality: help(samples.stats),"
            " as well as average loglikelihoods: help(samples.logL_P)"
            )

    def stats(self, nsamples=None, beta=None):
        r"""Compute Nested Sampling statistics.

        Using nested sampling we can compute:

        - ``logZ``: Bayesian evidence

          .. math::
              \log Z = \int L \pi d\theta

        - ``D_KL``: Kullback--Leibler divergence

          .. math::
              D_{KL} = \int P \log(P / \pi) d\theta

        - ``logL_P``: posterior averaged log-likelihood

          .. math::
              \langle\log L\rangle_P = \int P \log L d\theta

        - ``d_G``: Gaussian model dimensionality
          (or posterior variance of the log-likelihood)

          .. math::
              d_G/2 = \langle(\log L)^2\rangle_P - \langle\log L\rangle_P^2

          see `Handley and Lemos (2019) <https://arxiv.org/abs/1903.06682>`_
          for more details on model dimensionalities.

        (Note that all of these are available as individual functions with the
        same signature.)

        In addition to point estimates nested sampling provides an error bar
        or more generally samples from a (correlated) distribution over the
        variables. Samples from this distribution can be computed by providing
        an integer nsamples.

        Nested sampling as an athermal algorithm is also capable of producing
        these as a function of inverse thermodynamic temperature beta. This is
        provided as a vectorised function. If nsamples is also provided a
        MultiIndex dataframe is generated.

        These obey Occam's razor equation:

        .. math::
            \log Z = \langle\log L\rangle_P - D_{KL},

        which splits a model's quality ``logZ`` into a goodness-of-fit
        ``logL_P`` and a complexity penalty ``D_KL``. See `Hergt et al. (2021)
        <https://arxiv.org/abs/2102.11511>`_ for more detail.

        Parameters
        ----------
        nsamples : int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
              values inferred by nested sampling

        beta : float, array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if beta is scalar and nsamples is None:
            Series, index ['logZ', 'd_G', 'DK_L', 'logL_P']
        elif beta is scalar and nsamples is int:
            :class:`Samples`, index range(nsamples),
            columns ['logZ', 'd_G', 'DK_L', 'logL_P']
        elif beta is array-like and nsamples is None:
            :class:`Samples`, index beta,
            columns ['logZ', 'd_G', 'DK_L', 'logL_P']
        elif beta is array-like and nsamples is int:
            :class:`Samples`, index :class:`pandas.MultiIndex` the product of
            beta and range(nsamples)
            columns ['logZ', 'd_G', 'DK_L', 'logL_P']
        """
        logw = self.logw(nsamples, beta)
        if nsamples is None and beta is None:
            samples = self._constructor_sliced(index=self.columns[:0],
                                               dtype=float)
        else:
            samples = Samples(index=logw.columns, columns=self.columns[:0])
        samples['logZ'] = self.logZ(logw)
        samples.set_label('logZ', r'$\ln\mathcal{Z}$')
        w = np.exp(logw-samples['logZ'])

        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - samples.logZ

        samples['D_KL'] = (S*w).sum()
        samples.set_label('D_KL', r'$\mathcal{D}_\mathrm{KL}$')

        samples['logL_P'] = samples['logZ'] + samples['D_KL']
        samples.set_label('logL_P',
                          r'$\langle\ln\mathcal{L}\rangle_\mathcal{P}$')

        samples['d_G'] = ((S-samples.D_KL)**2*w).sum()*2
        samples.set_label('d_G', r'$d_\mathrm{G}$')

        samples.label = self.label
        return samples

    def logX(self, nsamples=None):
        """Log-Volume.

        The log of the prior volume contained within each iso-likelihood
        contour.

        Parameters
        ----------
        nsamples : int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
              values inferred by nested sampling

        Returns
        -------
        if nsamples is None:
            WeightedSeries like self
        elif nsamples is int:
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

    # TODO: remove this in version >= 2.1
    def dlogX(self, nsamples=None):
        # noqa: disable=D102
        raise NotImplementedError(
            "This is anesthetic 1.0 syntax. You should instead use logdX."
            )

    def logdX(self, nsamples=None):
        """Compute volume of shell of loglikelihood.

        Parameters
        ----------
        nsamples : int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
              values inferred by nested sampling

        Returns
        -------
        if nsamples is None:
            WeightedSeries like self
        elif nsamples is int:
            WeightedDataFrame like self, columns range(nsamples)
        """
        logX = self.logX(nsamples)
        logXp = logX.shift(1, fill_value=0)
        logXm = logX.shift(-1, fill_value=-np.inf)
        logdX = np.log(1 - np.exp(logXm-logXp)) + logXp - np.log(2)
        logdX.name = 'logdX'

        return logdX

    def _betalogL(self, beta=None):
        """Log(L**beta) convenience function.

        Parameters
        ----------
        beta : scalar or array-like, optional
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
        nsamples : int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
              values inferred by nested sampling
            - If nsamples is array, nsamples is assumed to be logw and returned
              (implementation convenience functionality)

        beta : float, array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if nsamples is array-like:
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

        logdX = self.logdX(nsamples)
        betalogL = self._betalogL(beta)

        if logdX.ndim == 1 and betalogL.ndim == 1:
            logw = logdX + betalogL
        elif logdX.ndim > 1 and betalogL.ndim == 1:
            logw = logdX.add(betalogL, axis=0)
        elif logdX.ndim == 1 and betalogL.ndim > 1:
            logw = betalogL.add(logdX, axis=0)
        else:
            cols = MultiIndex.from_product([betalogL.columns, logdX.columns])
            logdX = logdX.reindex(columns=cols, level='samples')
            betalogL = betalogL.reindex(columns=cols, level='beta')
            logw = betalogL+logdX
        return logw

    def logZ(self, nsamples=None, beta=None):
        """Log-Evidence.

        Parameters
        ----------
        nsamples : int, optional
            - If nsamples is not supplied, calculate mean value
            - If nsamples is integer, draw nsamples from the distribution of
              values inferred by nested sampling
            - If nsamples is array, nsamples is assumed to be logw

        beta : float, array-like, optional
            inverse temperature(s) beta=1/kT. Default self.beta

        Returns
        -------
        if nsamples is array-like:
            :class:`pandas.Series`, index nsamples.columns
        elif beta is scalar and nsamples is None:
            float
        elif beta is array-like and nsamples is None:
            :class:`pandas.Series`, index beta
        elif beta is scalar and nsamples is int:
            :class:`pandas.Series`, index range(nsamples)
        elif beta is array-like and nsamples is int:
            :class:`pandas.Series`, :class:`pandas.MultiIndex` columns the
            product of beta and range(nsamples)
        """
        logw = self.logw(nsamples, beta)
        logZ = logsumexp(logw, axis=0)
        if np.isscalar(logZ):
            return logZ
        else:
            return logw._constructor_sliced(logZ, name='logZ',
                                            index=logw.columns).squeeze()

    _logZ_function_shape = '\n' + '\n'.join(logZ.__doc__.split('\n')[1:])

    # TODO: remove this in version >= 2.1
    def D(self, nsamples=None):
        # noqa: disable=D102
        raise NotImplementedError(
            "This is anesthetic 1.0 syntax. You need to update, e.g.\n"
            "samples.D(1000)     # anesthetic 1.0\n"
            "samples.D_KL(1000)  # anesthetic 2.0\n\n"
            "Check out the new temperature functionality: help(samples.D_KL), "
            "as well as average loglikelihoods: help(samples.logL_P)"
            )

    def D_KL(self, nsamples=None, beta=None):
        """Kullback--Leibler divergence."""
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

    # TODO: remove this in version >= 2.1
    def d(self, nsamples=None):
        # noqa: disable=D102
        raise NotImplementedError(
            "This is anesthetic 1.0 syntax. You need to update, e.g.\n"
            "samples.d(1000)     # anesthetic 1.0\n"
            "samples.d_G(1000)  # anesthetic 2.0\n\n"
            "Check out the new temperature functionality: help(samples.d_G), "
            "as well as average loglikelihoods: help(samples.logL_P)"
            )

    def d_G(self, nsamples=None, beta=None):
        """Bayesian model dimensionality."""
        logw = self.logw(nsamples, beta)
        logZ = self.logZ(logw, beta)
        betalogL = self._betalogL(beta)
        S = (logw*0).add(betalogL, axis=0) - logZ
        w = np.exp(logw-logZ)
        D_KL = (S*w).sum()
        d_G = ((S-D_KL)**2*w).sum()*2
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

    def contour(self, logL=None):
        """Convert contour from (index or None) to a float loglikelihood.

        Convention is that live points are inclusive of the contour.

        Helper function for:
            - NestedSamples.live_points,
            - NestedSamples.dead_points,
            - NestedSamples.truncate.

        Parameters
        ----------
        logL : float or int, optional
            Loglikelihood or iteration number
            If not provided, return the contour containing the last set of
            live points.

        Returns
        -------
        logL : float
            Loglikelihood of contour
        """
        if logL is None:
            logL = self.loc[self.logL > self.logL_birth.max()].logL.iloc[0]
        elif isinstance(logL, float):
            pass
        else:
            logL = float(self.logL[logL])
        return logL

    def live_points(self, logL=None):
        """Get the live points within a contour.

        Parameters
        ----------
        logL : float or int, optional
            Loglikelihood or iteration number to return live points.
            If not provided, return the last set of active live points.

        Returns
        -------
        live_points : Samples
            Live points at either:
                - contour logL (if input is float)
                - ith iteration (if input is integer)
                - last set of live points if no argument provided
        """
        logL = self.contour(logL)
        i = ((self.logL >= logL) & (self.logL_birth < logL)).to_numpy()
        return Samples(self[i]).set_weights(None)

    def dead_points(self, logL=None):
        """Get the dead points at a given contour.

        Convention is that dead points are exclusive of the contour.

        Parameters
        ----------
        logL : float or int, optional
            Loglikelihood or iteration number to return dead points.
            If not provided, return the last set of dead points.

        Returns
        -------
        dead_points : Samples
            Dead points at either:
                - contour logL (if input is float)
                - ith iteration (if input is integer)
                - last set of dead points if no argument provided
        """
        logL = self.contour(logL)
        i = ((self.logL < logL)).to_numpy()
        return Samples(self[i]).set_weights(None)

    def truncate(self, logL=None):
        """Truncate the run at a given contour.

        Returns the union of the live_points and dead_points.

        Parameters
        ----------
        logL : float or int, optional
            Loglikelihood or iteration number to truncate run.
            If not provided, truncate at the last set of dead points.

        Returns
        -------
        truncated_run : NestedSamples
            Run truncated at either:
                - contour logL (if input is float)
                - ith iteration (if input is integer)
                - last set of dead points if no argument provided
        """
        dead_points = self.dead_points(logL)
        live_points = self.live_points(logL)
        index = np.concatenate([dead_points.index, live_points.index])
        return self.loc[index].recompute()

    def posterior_points(self, beta=1):
        """Get equally weighted posterior points at temperature beta."""
        return self.set_beta(beta).compress('equal')

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
        logL_new : np.array
            New log-likelihood values. Should have the same shape as `logL`.

        action : str, default='add'
            Can be any of {'add', 'replace', 'mask'}.

            * add: Add the new `logL_new` to the current `logL`.
            * replace: Replace the current `logL` with the new `logL_new`.
            * mask: treat `logL_new` as a boolean mask and only keep the
              corresponding (True) samples.

        inplace : bool, optional
            Indicates whether to modify the existing array, or return a new
            frame with importance sampling applied.
            default: False

        Returns
        -------
        samples : :class:`NestedSamples`
            Importance re-weighted samples.

        """
        samples = super().importance_sample(logL_new, action=action)
        mask = (samples.logL > samples.logL_birth).to_numpy()
        samples = samples[mask].recompute()
        if inplace:
            self._update_inplace(samples)
        else:
            return samples.__finalize__(self, "importance_sample")

    def recompute(self, logL_birth=None, inplace=False):
        """Re-calculate the nested sampling contours and live points.

        Parameters
        ----------
        logL_birth : array-like or int, optional

            * array-like: the birth contours.
            * int: the number of live points.
            * default: use the existing birth contours to compute nlive

        inplace : bool, default=False
            Indicates whether to modify the existing array, or return a new
            frame with contours resorted and nlive recomputed

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

            invalid = (samples.logL <= samples.logL_birth).to_numpy()
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
            samples = samples[samples.logL.notna().to_numpy()].recompute()

        if inplace:
            self._update_inplace(samples)
        else:
            return samples.__finalize__(self, "recompute")


def merge_nested_samples(runs):
    """Merge one or more nested sampling runs.

    Parameters
    ----------
    runs : list(:class:`NestedSamples`)
        List or array-like of one or more nested sampling runs.
        If only a single run is provided, this recalculates the live points and
        as such can be used for masked runs.

    Returns
    -------
    samples : :class:`NestedSamples`
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
    samples : list(:class:`NestedSamples`) or list(:class:`MCMCSamples`)
        List or array-like of one or more MCMC or nested sampling runs.

    weights : list(double) or None
        Weight for each run in samples (normalized internally).
        Can be omitted if samples are :class:`NestedSamples`,
        then exp(logZ) is used as weight.

    label : str or None, default=None
        Label for the new samples.

    Returns
    -------
    new_samples : :class:`Samples`
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


adjust_docstrings(Samples.to_hdf, r'(pd|pandas)\.DataFrame', 'DataFrame')
adjust_docstrings(Samples.to_hdf, 'DataFrame', 'pandas.DataFrame')
adjust_docstrings(Samples.to_hdf, r'(pd|pandas)\.read_hdf', 'read_hdf')
adjust_docstrings(Samples.to_hdf, 'read_hdf', 'pandas.read_hdf')
adjust_docstrings(Samples.to_hdf, ':func:`open`', '`open`')
