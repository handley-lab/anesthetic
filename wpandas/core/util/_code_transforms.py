"""Utility functions for code transformation."""
import inspect
import re


def adjust_docstrings(cls, pattern, repl, *args, **kwargs):
    """Adjust the docstrings of a class using regular expressions.

    After the first argument, the remaining arguments are identical to re.sub.

    Parameters
    ----------
    cls : class
        class to adjust

    pattern : str
        regular expression pattern

    repl : str
        replacement string
    """
    for key, val in cls.__dict__.items():
        doc = inspect.getdoc(val)
        if doc is not None:
            newdoc = re.sub(pattern, repl, doc, *args, **kwargs)
            try:
                cls.__dict__[key].__doc__ = newdoc
            except AttributeError:
                pass


def adjust_weighted_docstrings(cls):
    adjust_docstrings(cls, r'\bDataFrame\b', 'WeightedDataFrame')
    adjust_docstrings(cls, r'\bDataFrames\b', 'WeightedDataFrames')
    adjust_docstrings(cls, r'\bSeries\b', 'WeightedSeries')
    adjust_docstrings(cls, 'core', 'pandas.core')
    adjust_docstrings(cls, 'pandas.core.window.Rolling.quantile',
                           'pandas.core.window.rolling.Rolling.quantile')
    adjust_docstrings(cls, r'\bDataFrameGroupBy\b', 'WeightedDataFrameGroupBy')
    adjust_docstrings(cls, r'\bSeriesGroupBy\b', 'WeightedSeriesGroupBy')
    adjust_docstrings(cls, 'resample', 'pandas.DataFrame.resample')
    adjust_docstrings(cls, 'resample', 'pandas.Series.resample')


def replace_inner_function(outer, new_inner):
    """Replace a nested function code object used by outer with new_inner.

    The replacement new_inner must use the same name and must at most use the
    same closures as the original.

    Code taken from https://stackoverflow.com/a/27550237
    """
    # find original code object so we can validate the closures match
    ocode = outer.__code__
    function, code = type(outer), type(ocode)
    iname = new_inner.co_name
    orig_inner = next(
        const for const in ocode.co_consts
        if isinstance(const, code) and const.co_name == iname)

    # you can ignore later closures, but since they are matched by position
    # the new sequence must match the start of the old.
    assert (orig_inner.co_freevars[:len(new_inner.co_freevars)] ==
            new_inner.co_freevars), 'New closures must match originals'

    # replace the code object for the inner function
    new_consts = tuple(
        new_inner if const is orig_inner else const
        for const in outer.__code__.co_consts)

    # create a new code object with the new constants
    ncode = ocode.replace(co_consts=new_consts)

    # and a new function object using the updated code object
    return function(
        ncode, outer.__globals__, outer.__name__,
        outer.__defaults__, outer.__closure__
    )
