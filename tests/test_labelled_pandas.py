import numpy as np
from anesthetic.labelled_pandas import LabelledSeries, LabelledDataFrame
from pandas import Series, DataFrame, MultiIndex
from pandas.testing import assert_series_equal, assert_frame_equal


def test_LabelledSeries():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index))

    lseries = LabelledSeries(data, index, labels=labels)
    series = Series(data, [index, labels])
    series.index.names = [None, 'labels']

    assert_series_equal(Series(lseries), series)

    assert lseries.A == series['A', '$A$']

    assert lseries['A', '$A$'] == series['A', '$A$']
    assert lseries['A'] == series['A', '$A$']
    assert_series_equal(Series(lseries[:, '$A$']), series[:, '$A$'])
    assert_series_equal(Series(lseries[:, '$A$']), series[:, '$A$'])
    assert lseries['A', :] == series['A', '$A$']

    assert lseries.loc['A', '$A$'] == series.loc['A', '$A$']
    assert lseries.loc['A'] == series.loc['A', '$A$']
    assert_series_equal(Series(lseries.loc[:, '$A$']), series.loc[:, '$A$'])
    assert lseries.loc['A', :] == series.loc['A', '$A$']

    assert lseries.at['A', '$A$'] == series.at['A', '$A$']
    assert lseries.at['A'] == series.at['A', '$A$']

    assert lseries.xs(('A', '$A$')) == series.xs(('A', '$A$'))
    assert lseries.xs('A') == series.xs(('A', '$A$'))
    assert_series_equal(Series(lseries.xs('$A$', level=1)),
                        series.xs('$A$', level=1))


def test_LabelledSeries_MultiIndex():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    vowels = ['vowel' if c in 'AEIOU' else 'consonant' for c in index]
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index))

    multiindex = MultiIndex.from_arrays([vowels, index],
                                        names=['vowels', None])
    lseries = LabelledSeries(data, multiindex, labels=labels)

    multiindex = MultiIndex.from_arrays([vowels, index, labels],
                                        names=['vowels', None, 'labels'])
    series = Series(data, multiindex)

    assert_series_equal(series, Series(lseries))

    assert_series_equal(Series(lseries.vowel), series.vowel)
    assert lseries.vowel.A == series['vowel', 'A', '$A$']

    assert lseries['vowel', 'A', '$A$'] == series['vowel', 'A', '$A$']
    assert lseries['vowel', 'A'] == series['vowel', 'A', '$A$']
    assert_series_equal(Series(lseries['vowel']), series['vowel'])

    assert_series_equal(Series(lseries[:, :, '$A$']),
                        series[:, :, '$A$'])
    assert_series_equal(Series(lseries[:, 'A', :]),
                        series[:, 'A', :])
    assert_series_equal(Series(lseries['vowel', :, :]),
                        series['vowel', :, :])
    assert_series_equal(Series(lseries[:, 'A', '$A$']),
                        series[:, 'A', '$A$'])
    assert_series_equal(Series(lseries['vowel', :, '$A$']),
                        series['vowel', :, '$A$'])
    assert lseries['vowel', 'A', :] == series['vowel', 'A', '$A$']

    assert lseries.loc['vowel', 'A', '$A$'] == series.loc['vowel', 'A', '$A$']
    assert lseries.loc['vowel', 'A'] == series.loc['vowel', 'A', '$A$']
    assert_series_equal(Series(lseries.loc['vowel']), series.loc['vowel'])

    assert_series_equal(Series(lseries.loc[:, :, '$A$']),
                        series.loc[:, :, '$A$'])
    assert_series_equal(Series(lseries.loc[:, 'A', :]),
                        series.loc[:, 'A', :])
    assert_series_equal(Series(lseries.loc['vowel', :, :]),
                        series.loc['vowel', :, :])
    assert_series_equal(Series(lseries.loc[:, 'A', '$A$']),
                        series.loc[:, 'A', '$A$'])
    assert_series_equal(Series(lseries.loc['vowel', :, '$A$']),
                        series.loc['vowel', :, '$A$'])
    assert lseries.loc['vowel', 'A', :] == lseries.loc['vowel', 'A', '$A$']

    assert lseries.at['vowel', 'A', '$A$'] == series.at['vowel', 'A', '$A$']
    assert lseries.at['vowel', 'A'] == series.at['vowel', 'A', '$A$']
    assert_series_equal(Series(lseries.at['vowel']), series.at['vowel'])

    assert (lseries.xs(('vowel', 'A', '$A$'))
            == series.xs(('vowel', 'A', '$A$')))
    assert lseries.xs(('vowel', 'A')) == series.xs(('vowel', 'A', '$A$'))
    assert_series_equal(Series(lseries.xs('$A$', level=2)),
                        series.xs('$A$', level=2))
    assert_series_equal(Series(lseries.xs('A', level=1)),
                        series.xs('A', level=1))
    assert_series_equal(Series(lseries.xs('vowel', level=0)),
                        series.xs('vowel', level=0))
    assert_series_equal(Series(lseries.xs(('A', '$A$'), level=[1, 2])),
                        series.xs(('A', '$A$'), level=[1, 2]))
    assert_series_equal(Series(lseries.xs(('vowel', '$A$'), level=[0, 2])),
                        series.xs(('vowel', '$A$'), level=[0, 2]))
    assert_series_equal(Series(lseries.xs(('vowel', 'A'), level=[0, 1])),
                        series.xs(('vowel', 'A', '$A$'), level=[0, 1, 2]).
                        droplevel('labels'))


def test_LabelledDataFrame_index():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index), 4)

    lframe = LabelledDataFrame(data, index, labels=labels)
    frame = DataFrame(data, [index, labels])
    frame.index.names = [None, 'labels']

    assert_frame_equal(DataFrame(lframe), frame)


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
