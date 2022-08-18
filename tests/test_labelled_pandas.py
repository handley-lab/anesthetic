import numpy as np
from numpy.testing import assert_array_equal
from anesthetic.labelled_pandas import LabelledSeries, LabelledDataFrame
from pandas import Series, DataFrame, MultiIndex
import pandas.testing
import pytest


def assert_series_equal(x, y):
    pandas.testing.assert_series_equal(Series(x), Series(y))


def assert_series_equal_not_index(x, y):
    assert_series_equal(x.drop_labels(), y)
    with pytest.raises(AssertionError):
        assert_series_equal(x, y)


def assert_series_equal_not_name(x, y):
    with pytest.raises(AssertionError):
        assert_series_equal(x, y)
    x.name = y.name
    assert_series_equal(x, y)


def assert_frame_equal(x, y):
    pandas.testing.assert_frame_equal(DataFrame(x), DataFrame(y))


def assert_frame_equal_not_index(x, y):
    assert_frame_equal(x.drop_labels(), y)
    with pytest.raises(AssertionError):
        assert_frame_equal(x, y)


def test_LabelledSeries():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index))

    lseries = LabelledSeries(data, index, labels=labels)
    series = Series(data, index)

    assert_series_equal_not_index(lseries, series)

    assert lseries.A == series.A
    assert lseries['A'] == series['A']
    assert lseries.loc['A'] == series.loc['A']
    assert lseries.at['A'] == series.at['A']
    assert lseries.xs('A') == series.xs('A')
    assert_series_equal_not_index(lseries['A':'D'], series['A':'D'])

    assert lseries[('A', '$A$')] == lseries['A']
    assert lseries.loc['A', '$A$'] == lseries['A']
    assert lseries.at['A', '$A$'] == lseries['A']
    assert lseries.xs(('A', '$A$')) == lseries['A']


def test_LabelledSeries_MultiIndex():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    vowels = ['vowel' if c in 'AEIOU' else 'consonant' for c in index]
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index))

    multiindex = MultiIndex.from_arrays([vowels, index],
                                        names=['vowels', None])
    lseries = LabelledSeries(data, multiindex, labels=labels)
    series = Series(data, multiindex)

    assert_series_equal_not_index(lseries, series)

    assert_series_equal_not_index(lseries.vowel, series.vowel)
    assert lseries.vowel.A == series.vowel.A

    assert lseries['vowel', 'A'] == series['vowel', 'A']
    assert_series_equal_not_index(lseries['vowel'], series['vowel'])

    assert_series_equal_not_index(lseries[:, 'A'], series[:, 'A'])
    assert_series_equal_not_index(lseries['vowel', :], series['vowel', :])

    assert lseries.loc['vowel', 'A'] == series.loc['vowel', 'A']
    assert_series_equal_not_index(lseries.loc['vowel'], series.loc['vowel'])

    assert_series_equal_not_index(lseries.loc[:, 'A'], series.loc[:, 'A'])
    assert_series_equal_not_index(lseries.loc['vowel', :],
                                  series.loc['vowel', :])
    assert lseries.loc['vowel', 'A'] == lseries.loc['vowel', 'A']

    assert lseries.at['vowel', 'A'] == series.at['vowel', 'A']
    assert_series_equal_not_index(lseries.at['vowel'], series.at['vowel'])

    assert (lseries.xs(('vowel', 'A')) == series.xs(('vowel', 'A')))

    assert_series_equal_not_index(lseries.xs('A', level=1),
                                  series.xs('A', level=1))
    assert_series_equal_not_index(lseries.xs('vowel', level=0),
                                  series.xs('vowel', level=0))
    assert_array_equal(lseries.xs(('vowel', 'A'), level=[0, 1]),
                       series.xs(('vowel', 'A'), level=[0, 1]))

    with pytest.raises(KeyError):
        lseries['foo']


def test_LabelledDataFrame_index():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index), 4)

    lframe = LabelledDataFrame(data, index, labels=labels)
    frame = DataFrame(data, index)

    assert_frame_equal_not_index(lframe, frame)

    assert_series_equal(lframe.loc['A'], frame.loc['A'])
    assert lframe.at['A', 0] == frame.at['A', 0]
    assert_series_equal(lframe.xs('A'), frame.xs('A'))

    assert_array_equal(lframe.loc[('A', '$A$')], frame.loc['A'])
    assert lframe.at[('A', '$A$'), 0] == frame.at['A', 0]
    assert_array_equal(lframe.xs(('A', '$A$')), frame.xs('A'))

    assert_series_equal(lframe.T['A'], lframe.loc['A'])
    assert_series_equal(lframe.T.A, lframe.loc['A'])
    assert_series_equal(lframe.T.loc[:, 'A'], lframe.loc['A'])
    assert_series_equal(lframe.T.loc[0, 'A'], lframe.loc['A', 0])

    assert_series_equal(lframe.T.at[0, 'A'], lframe.at['A', 0])

    assert_frame_equal_not_index(lframe['A':'C'], frame['A':'C'])
    assert_frame_equal_not_index(lframe.loc['A':'C', 0:3],
                                 frame.loc['A':'C', 0:3])
    assert_frame_equal_not_index(lframe.loc['A':'C', 0:3],
                                 frame.loc['A':'C', 0:3])

    with pytest.raises(KeyError):
        lframe['foo']


def test_LabelledDataFrame_index_MultiIndex():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    vowels = ['vowel' if c in 'AEIOU' else 'consonant' for c in index]
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index), 4)

    multiindex = MultiIndex.from_arrays([vowels, index],
                                        names=['vowels', None])
    lframe = LabelledDataFrame(data, multiindex, labels=labels)
    frame = DataFrame(data, multiindex)

    assert_frame_equal_not_index(lframe, frame)

    assert_frame_equal_not_index(lframe.loc['vowel'], frame.loc['vowel'])
    assert_series_equal(lframe.loc['vowel', 'A'], frame.loc['vowel', 'A'])
    assert_series_equal_not_name(lframe.loc['vowel', 'A', '$A$'],
                                 frame.loc['vowel', 'A'])

    assert_frame_equal_not_index(lframe.xs('vowel'), frame.xs('vowel'))
    assert_series_equal(lframe.xs(('vowel', 'A')), frame.xs(('vowel', 'A')))
    assert_series_equal_not_name(lframe.xs(('vowel', 'A', '$A$')),
                                 frame.xs(('vowel', 'A')))

    assert_series_equal(lframe.loc[('vowel', 'A'), 0],
                        frame.loc[('vowel', 'A'), 0])
    assert_series_equal(lframe.loc[('vowel', 'A', '$A$'), 0],
                        frame.loc[('vowel', 'A'), 0])
    assert lframe.at[('vowel', 'A'), 0] == frame.at[('vowel', 'A'), 0]
    assert lframe.at[('vowel', 'A', '$A$'), 0] == frame.at[('vowel', 'A'), 0]

    assert_frame_equal(lframe.T['vowel'], lframe.loc['vowel'].T)
    assert_series_equal(lframe.T['vowel', 'A'], lframe.loc['vowel', 'A'])
    assert_series_equal_not_name(lframe.T['vowel', 'A', '$A$'],
                                 lframe.loc['vowel', 'A'])
    assert_frame_equal(lframe.T.vowel, lframe.loc['vowel'].T)
    assert_series_equal_not_name(lframe.T.vowel.A, lframe.loc['vowel', 'A'])
    assert_frame_equal(lframe.T.loc[:, 'vowel'], lframe.loc['vowel'].T)
    assert_series_equal(lframe.T.loc[:, ('vowel', 'A')],
                        lframe.loc[('vowel', 'A')])
    assert_series_equal_not_name(lframe.T.loc[:, ('vowel', 'A', '$A$')],
                                 lframe.loc[('vowel', 'A')])
    assert_series_equal(lframe.T.loc[0, 'vowel'], lframe.loc['vowel', 0])
    assert lframe.T.loc[0, ('vowel', 'A')] == lframe.loc[('vowel', 'A'), 0]
    assert (lframe.T.loc[0, ('vowel', 'A', '$A$')]
            == lframe.loc[('vowel', 'A'), 0])

    assert_series_equal(lframe.T.at[0, ('vowel', 'A')],
                        lframe.at[('vowel', 'A'), 0])
    assert_series_equal(lframe.T.at[0, ('vowel', 'A', '$A$')],
                        lframe.at[('vowel', 'A'), 0])

    with pytest.raises(KeyError):
        lframe['foo']
