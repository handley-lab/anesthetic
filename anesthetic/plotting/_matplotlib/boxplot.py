import pandas.plotting._matplotlib.boxplot
from pandas.plotting._matplotlib.boxplot import BoxPlot as _BoxPlot
from anesthetic.plotting._matplotlib.core import _WeightedMPLPlot, _get_weights
from anesthetic.utils import quantile
from pandas.core.dtypes.missing import remove_na_arraylike
import numpy as np
from pandas.io.formats.printing import pprint_thing
from anesthetic._code_utils import replace_inner_function


def _bxpstat(y, weights, whis):
    q1, med, q3 = quantile(y, [0.25, 0.5, 0.75], weights)
    bxpstat = {}
    bxpstat['med'] = med
    bxpstat['q1'] = q1
    bxpstat['q3'] = q3
    bxpstat['whislo'] = q1 - whis*(q3-q1)
    bxpstat['whishi'] = q3 + whis*(q3-q1)
    return bxpstat


class BoxPlot(_WeightedMPLPlot, _BoxPlot):
    # noqa: disable=D101
    @classmethod
    def _plot(cls, ax, y, column_num=None, return_type="axes", **kwds):
        if y.ndim == 2:
            y = [remove_na_arraylike(v) for v in y]
            # Boxplot fails with empty arrays, so need to add a NaN
            #   if any cols are empty
            # GH 8181
            y = [v if v.size > 0 else np.array([np.nan]) for v in y]
        else:
            y = remove_na_arraylike(y)

        weights = kwds.pop("weights", None)

        if weights is None:
            bp = ax.boxplot(y, **kwds)
        else:
            whis = kwds.pop("whis", 1.5)
            kwds['showfliers'] = False
            y = np.atleast_2d(y)
            bp = ax.bxp([_bxpstat(yi, weights, whis) for yi in y], **kwds)

        if return_type == "dict":
            return bp, bp
        elif return_type == "both":
            return cls.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp


def boxplot(data, *args, **kwds):
    # noqa: disable=D103
    _get_weights(kwds, data)

    def create_plot_group():
        fontsize = None  # pragma: no cover
        maybe_color_bp = None  # pragma: no cover
        return_type = None  # pragma: no cover
        rot = None  # pragma: no cover

        def plot_group(keys, values, ax, **kwds):  # pragma: no cover
            # GH 45465: xlabel/ylabel need to be popped out before plotting
            xlabel = kwds.pop("xlabel", None)
            ylabel = kwds.pop("ylabel", None)
            if xlabel:
                ax.set_xlabel(pprint_thing(xlabel))
            if ylabel:
                ax.set_ylabel(pprint_thing(ylabel))

            weights = kwds.pop("weights", None)
            keys = [pprint_thing(x) for x in keys]
            if weights is None:
                values = [np.asarray(remove_na_arraylike(v), dtype=object)
                          for v in values]
                bp = ax.boxplot(values, **kwds)
            else:
                whis = kwds.pop("whis", 1.5)
                kwds['showfliers'] = False
                from anesthetic.plotting._matplotlib.boxplot import _bxpstat
                bp = ax.bxp([_bxpstat(v, weights, whis) for v in values],
                            **kwds)

            if fontsize is not None:
                ax.tick_params(axis="both", labelsize=fontsize)

            # GH 45465: x/y are flipped when "vert" changes
            is_vertical = kwds.get("vert", True)
            ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
            if len(ticks) != len(keys):
                i, remainder = divmod(len(ticks), len(keys))
                assert remainder == 0, remainder
                keys *= i
            if is_vertical:
                ax.set_xticklabels(keys, rotation=rot)
            else:
                ax.set_yticklabels(keys, rotation=rot)
            maybe_color_bp(bp, **kwds)

            # Return axes in multiplot case, maybe revisit later # 985
            if return_type == "dict":
                return bp
            elif return_type == "both":
                return BoxPlot.BP(ax=ax, lines=bp)
            else:
                return ax

    boxplot = replace_inner_function(pandas.plotting._matplotlib.boxplot,
                                     create_plot_group.__code__.co_consts[1])
    return boxplot(data, *args, **kwds)


def boxplot_frame(data, *args, **kwds):
    # noqa: disable=D103
    _get_weights(kwds, data)
    import matplotlib.pyplot as plt

    ax = boxplot(data, *args, **kwds)
    plt.draw_if_interactive()
    return ax
