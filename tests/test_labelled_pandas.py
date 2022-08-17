%load_ext autoreload
%autoreload 2
import numpy as np
from numpy.testing import assert_array_equal
from anesthetic.labelled_pandas import LabelledSeries, LabelledDataFrame
from pandas import Series, DataFrame, MultiIndex
import pandas.testing
import pytest

def assert_series_equal(x, y):
    pandas.testing.assert_series_equal(Series(x), Series(y))


def assert_frame_equal(x, y):
    pandas.testing.assert_frame_equal(DataFrame(x), DataFrame(y))


def test_LabelledSeries():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index))

    lseries = LabelledSeries(data, index, labels=labels)
    series = Series(data, index)

    assert_series_equal(lseries.drop_labels(), series)

    assert lseries.A == series.A

    assert lseries['A'] == series['A']
    assert lseries['A', '$A$'] == series['A']
    assert lseries['A', 'foo'] == series['A']
    assert_series_equal(lseries[:], series[:])
    assert_series_equal(lseries[:, '$A$'], series[:])
    assert_series_equal(lseries[:, 'foo'], series[:])
    assert lseries['A', :] == series['A']

    assert lseries.loc['A'] == series.loc['A']
    assert lseries.loc['A', '$A$'] == series.loc['A']
    assert lseries.loc['A', 'foo'] == series.loc['A']
    assert_series_equal(lseries.loc[:, '$A$'], series.loc[:])
    assert_series_equal(lseries.loc[:, 'foo'], series.loc[:])
    assert lseries.loc['A', :] == series.loc['A']

    assert lseries.at[('A', '$A$')] == series.at['A']
    assert lseries.at[('A', 'foo')] == series.at['A']
    assert lseries.at['A', '$A$'] == series.at['A']
    assert lseries.at['A', 'foo'] == series.at['A']
    assert lseries.at['A'] == series.at['A']

    assert lseries.xs(('A', '$A$')) == series.xs('A')
    assert lseries.xs(('A', 'foo')) == series.xs('A')
    assert lseries.xs('A') == series.xs('A')
    with pytest.raises(TypeError):
        lseries.xs('$A$', level=1)
    with pytest.raises(TypeError):
        series.xs('$A$', level=1)


def test_LabelledSeries_MultiIndex():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    vowels = ['vowel' if c in 'AEIOU' else 'consonant' for c in index]
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index))

    multiindex = MultiIndex.from_arrays([vowels, index],
                                        names=['vowels', None])
    lseries = LabelledSeries(data, multiindex, labels=labels)
    series = Series(data, multiindex)

    assert_array_equal(lseries, series)
    assert_series_equal(lseries.drop_labels(), series)

    assert_series_equal(lseries.vowel, series.vowel)
    assert lseries.vowel.A == series.vowel.A

    assert lseries['vowel', 'A', '$A$'] == series['vowel', 'A']
    assert lseries['vowel', 'A', 'foo'] == series['vowel', 'A']
    assert lseries['vowel', 'A'] == series['vowel', 'A']
    assert_series_equal(lseries['vowel'], series['vowel'])

    assert_series_equal(lseries[:, :, '$A$'], series[:, :])
    assert_series_equal(lseries[:, 'A', :], series[:, 'A'])
    assert_series_equal(lseries['vowel', :, :], series['vowel', :])
    assert_series_equal(lseries[:, 'A', '$A$'], series[:, 'A'])
    assert_series_equal(lseries['vowel', :, '$A$'], series['vowel', :])
    assert lseries['vowel', 'A', :] == series['vowel', 'A']

    assert lseries.loc['vowel', 'A', '$A$'] == series.loc['vowel', 'A']
    assert lseries.loc['vowel', 'A'] == series.loc['vowel', 'A']
    assert_series_equal(lseries.loc['vowel'], series.loc['vowel'])

    assert_series_equal(lseries.loc[:, :, '$A$'], series.loc[:, :])
    assert_series_equal(lseries.loc[:, 'A', :], series.loc[:, 'A'])
    assert_series_equal(lseries.loc['vowel', :, :], series.loc['vowel', :])
    assert_series_equal(lseries.loc[:, 'A', '$A$'], series.loc[:, 'A'])
    assert_series_equal(lseries.loc['vowel', :, '$A$'], series.loc['vowel', :])
    assert lseries.loc['vowel', 'A', :] == lseries.loc['vowel', 'A']

    assert lseries.at['vowel', 'A', '$A$'] == series.at['vowel', 'A']
    assert lseries.at['vowel', 'A'] == series.at['vowel', 'A']
    assert_series_equal(lseries.at['vowel'], series.at['vowel'])

    assert (lseries.xs(('vowel', 'A', '$A$')) == series.xs(('vowel', 'A')))
    assert lseries.xs(('vowel', 'A')) == series.xs(('vowel', 'A'))
    with pytest.raises(IndexError):
        lseries.xs('$A$', level=2)
    with pytest.raises(IndexError):
        series.xs('$A$', level=2)

    assert_series_equal(lseries.xs('A', level=1), series.xs('A', level=1))
    assert_series_equal(lseries.xs('vowel', level=0), series.xs('vowel', level=0))
    assert_series_equal(lseries.xs(('vowel', 'A'), level=[0, 1]),
                        series.xs(('vowel', 'A'), level=[0, 1]))


def test_LabelledDataFrame_index():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index), 4)

    lframe = LabelledDataFrame(data, index, labels=labels)
    frame = DataFrame(data, index)
    #frame = DataFrame(data, [index, labels])
    #frame.index.names = [None, 'labels']

    assert_array_equal(lframe, frame)
    assert_frame_equal(lframe.drop_labels(), frame)

    #assert lseries['A', '$A$'] == series['A', '$A$']
    #assert lseries['A'] == series['A', '$A$']
    #assert_series_equal(lseries[:, '$A$'], series[:, '$A$'])
    #assert_series_equal(lseries[:, '$A$'], series[:, '$A$'])
    #assert lseries['A', :] == series['A', '$A$']

    assert_series_equal(lframe.loc['A', '$A$'], frame.loc['A'])
    assert_series_equal(lframe.loc['A'], frame.loc['A'])
    assert_frame_equal(lframe.loc[:, '$A$', :], frame.loc[:, :])
    assert_series_equal(lframe.loc['A', :], frame.loc['A'])

tup = (('A', '$A$'), 0)
    self = frame.loc
    self._expand_ellipsis((('A', '$A$'), 0))
    self._getitem_lowerdim((('A', '$A$'), 0))
    self._is_nested_tuple_indexer((('A', '$A$'), 0))

    self._getitem_nested_tuple((('A', '$A$'), 0))
    key = (('A', '$A$'), 0)
    self._getitem_axis(key)

    self = DataFrame(lframe).loc
    self._multi_take(tup)
    self._getitem_tuple_same_dim(tup)

    lframe.loc[('A', '$A$'), 0]
    DataFrame(lframe).loc[('A', '$A$'), 0]
    DataFrame(lframe).loc[('A', '$A$'), 0]

    assert lframe.loc['A', '$A$', 0] == frame.loc['A', 0]
    assert lframe.loc['A', 0] == frame.loc['A', 0]
    assert lframe.loc['A', 0] == uframe.loc['A', 0]
    assert_series_equal(lframe.loc[(slice(None), '$A$'), 0], frame.loc[(slice(None), '$A$'),0])
    assert lframe.loc['A', 0] == uframe.loc['A', 0]
    assert_series_equal(lframe.loc[('A', slice(None)), 0], frame.loc[('A', slice(None)), 0])

    assert lframe.at[('A', '$A$'), 0] == frame.at[('A', '$A$'), 0]
    assert lframe.at['A', 0] == frame.at[('A', '$A$'), 0]
    uframe.at['A', 0]
    lframe.at[('A', '$A$'), 0]
    lframe.at['A', 0]

    assert lseries.xs(('A', '$A$')) == series.xs(('A', '$A$'))
    assert lseries.xs('A') == series.xs(('A', '$A$'))
    assert_series_equal(lseries.xs('$A$', level=1),
                        series.xs('$A$', level=1))

    assert_series_equal(lframe.T.A, frame.loc['A', '$A$'])

def test_LabelledDataFrame_index_MultiIndex():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    vowels = ['vowel' if c in 'AEIOU' else 'consonant' for c in index]
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index), 4)

    multiindex = MultiIndex.from_arrays([vowels, index],
                                        names=['vowels', None])
    lframe = LabelledDataFrame(data, multiindex, labels=labels)

    multiindex = MultiIndex.from_arrays([vowels, index, labels],
                                        names=['vowels', None, 'labels'])
    frame = DataFrame(data, multiindex)

    assert_frame_equal(DataFrame(lframe), frame)
