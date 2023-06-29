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
    assert_frame_equal(x.drop_labels([0, 1]), y)
    with pytest.raises(AssertionError):
        assert_frame_equal(x, y)


@pytest.fixture
def lseries():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index))

    lseries = LabelledSeries(data, index, labels=labels)
    series = Series(data, index)

    assert_series_equal_not_index(lseries, series)
    isinstance(lseries, LabelledSeries)

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

    for c in index:
        assert lseries.get_labels_map()[c] == '$%s$' % c
        assert lseries.get_label(c) == '$%s$' % c

    return lseries


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
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

    for c, v in zip(index, vowels):
        assert lseries.get_labels_map()[v, c] == '$%s$' % c
        assert lseries.get_label((v, c)) == '$%s$' % c


@pytest.fixture
def lframe_index():
    index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in index]
    data = np.random.rand(len(index), 4)

    lframe = LabelledDataFrame(data, index, labels=labels)
    frame = DataFrame(data, index)

    assert_frame_equal_not_index(lframe, frame)

    assert_series_equal_not_name(lframe.loc['A'], frame.loc['A'])
    assert lframe.loc['A'].name == '$A$'
    assert lframe.at['A', 0] == frame.at['A', 0]
    assert_series_equal_not_name(lframe.xs('A'), frame.xs('A'))
    assert lframe.xs('A').name == '$A$'

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

    for c in index:
        assert lframe.get_labels_map()[c] == '$%s$' % c
        assert lframe.get_label(c) == '$%s$' % c

    return lframe


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
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
    assert_series_equal_not_name(lframe.loc['vowel', 'A'],
                                 frame.loc['vowel', 'A'])
    assert lframe.loc['vowel', 'A'].name == '$A$'
    assert_series_equal_not_name(lframe.loc['vowel', 'A', '$A$'],
                                 frame.loc['vowel', 'A'])

    assert_frame_equal_not_index(lframe.xs('vowel'), frame.xs('vowel'))
    assert_series_equal_not_name(lframe.xs(('vowel', 'A')),
                                 frame.xs(('vowel', 'A')))
    assert lframe.xs(('vowel', 'A')).name == '$A$'
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
    assert_series_equal(lframe.T.vowel.A, lframe.loc['vowel', 'A'])
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

    for c, v in zip(index, vowels):
        assert lframe.get_labels_map()[v, c] == '$%s$' % c
        assert lframe.get_label((v, c)) == '$%s$' % c


def test_LabelledDataFrame_column():
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = ['$%s$' % i for i in columns]
    data = np.random.rand(4, len(columns))

    lframe = LabelledDataFrame(data, columns=columns)
    lframe.set_labels(labels, axis=1, inplace=True)
    frame = DataFrame(data, columns=columns)

    assert_frame_equal_not_index(lframe, frame)

    assert_series_equal_not_name(lframe.A, frame.A)
    assert lframe.A.name == '$A$'
    assert_series_equal_not_name(lframe['A'], frame['A'])
    assert lframe['A'].name == '$A$'
    assert_series_equal_not_name(lframe['A', '$A$'], frame['A'])

    assert_series_equal_not_name(lframe.loc[:, 'A'], frame.loc[:, 'A'])
    assert lframe.loc[:, 'A'].name == '$A$'
    assert lframe.at[0, 'A'] == frame.at[0, 'A']
    assert_series_equal_not_name(lframe.xs('A', axis=1), frame.xs('A', axis=1))
    assert lframe.xs('A', axis=1).name == '$A$'

    assert_array_equal(lframe.loc[:, ('A', '$A$')], frame.loc[:, 'A'])
    assert lframe.at[0, ('A', '$A$')] == frame.at[0, 'A']
    assert_array_equal(lframe.xs(('A', '$A$'), axis=1), frame.xs('A', axis=1))

    assert_series_equal(lframe.T.loc['A'], lframe['A'])
    assert_series_equal(lframe.T.loc['A', 0], lframe.loc[0, 'A'])

    assert_series_equal(lframe.T.at['A', 0], lframe.at[0, 'A'])

    assert_frame_equal_not_index(lframe.loc[0:3, 'A':'C'],
                                 frame.loc[0:3, 'A':'C'])
    assert_frame_equal_not_index(lframe.loc[0:3, 'A':'C'],
                                 frame.loc[0:3, 'A':'C'])

    with pytest.raises(KeyError):
        lframe['foo']

    for c in columns:
        assert lframe.get_labels_map(1)[c] == '$%s$' % c
        assert lframe.get_label(c, 1) == '$%s$' % c


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_LabelledDataFrame_column_MultiIndex():
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    vowels = ['vowel' if c in 'AEIOU' else 'consonant' for c in columns]
    labels = ['$%s$' % i for i in columns]
    data = np.random.rand(4, len(columns))

    multiindex = MultiIndex.from_arrays([vowels, columns],
                                        names=['vowels', None])
    lframe = LabelledDataFrame(data, columns=multiindex)
    lframe.set_labels(labels, axis=1, inplace=True)
    frame = DataFrame(data, columns=multiindex)

    assert_frame_equal_not_index(lframe, frame)

    assert_frame_equal_not_index(lframe.vowel, frame.vowel)
    assert_series_equal_not_name(lframe.vowel.A, frame.vowel.A)
    assert lframe.vowel.A.name == '$A$'

    assert_frame_equal_not_index(lframe['vowel'], frame['vowel'])
    assert_series_equal_not_name(lframe['vowel', 'A'], frame['vowel', 'A'])
    assert lframe['vowel', 'A'].name == '$A$'
    assert_series_equal_not_name(lframe['vowel', 'A', '$A$'],
                                 frame['vowel', 'A'])

    assert_frame_equal_not_index(lframe.xs('vowel', axis=1),
                                 frame.xs('vowel', axis=1))
    assert_series_equal_not_name(lframe.xs(('vowel', 'A'), axis=1),
                                 frame.xs(('vowel', 'A'), axis=1))
    assert lframe.xs(('vowel', 'A'), axis=1).name == '$A$'
    assert_series_equal_not_name(lframe.xs(('vowel', 'A', '$A$'), axis=1),
                                 frame.xs(('vowel', 'A'), axis=1))

    assert_series_equal(lframe.loc[0, ('vowel', 'A')],
                        frame.loc[0, ('vowel', 'A')])
    assert_series_equal(lframe.loc[0, ('vowel', 'A', '$A$')],
                        frame.loc[0, ('vowel', 'A')])
    assert lframe.at[0, ('vowel', 'A')] == frame.at[0, ('vowel', 'A')]
    assert lframe.at[0, ('vowel', 'A', '$A$')] == frame.at[0, ('vowel', 'A')]

    assert_frame_equal(lframe.T.loc['vowel'], lframe['vowel'].T)
    assert_series_equal(lframe.T.loc['vowel', 'A'], lframe['vowel', 'A'])
    assert_series_equal_not_name(lframe.T.loc['vowel', 'A', '$A$'],
                                 lframe['vowel', 'A'])
    assert_series_equal_not_name(lframe.T.loc[('vowel', 'A', '$A$')],
                                 lframe[('vowel', 'A')])
    assert_series_equal(lframe.T.loc['vowel', 0], lframe.loc[0, 'vowel'])
    assert lframe.T.loc[('vowel', 'A'), 0] == lframe.loc[0, ('vowel', 'A')]
    assert (lframe.T.loc[('vowel', 'A', '$A$'), 0]
            == lframe.loc[0, ('vowel', 'A')])

    assert_series_equal(lframe.T.at[('vowel', 'A'), 0],
                        lframe.at[0, ('vowel', 'A')])
    assert_series_equal(lframe.T.at[('vowel', 'A', '$A$'), 0],
                        lframe.at[0, ('vowel', 'A')])

    with pytest.raises(KeyError):
        lframe['foo']

    for c, v in zip(columns, vowels):
        assert lframe.get_labels_map(1)[v, c] == '$%s$' % c
        assert lframe.get_label((v, c), 1) == '$%s$' % c


def test_set_labels(lseries):
    labels = lseries.get_labels()
    labels[1] = '$b$'
    assert_array_equal(lseries.set_labels(labels).get_labels(), labels)
    assert lseries.get_labels()[1] != labels[1]
    lseries.set_labels(labels, inplace=True)
    assert lseries.get_labels()[1] == labels[1]

    assert lseries.drop_labels().get_labels() is None
    assert lseries.get_labels() is not None
    assert lseries.set_labels(None).get_labels() is None
    assert lseries.get_labels() is not None
    lseries.set_labels(None, inplace=True)
    assert lseries.get_labels() is None


def test_constructors(lseries, lframe_index):
    lframe = lframe_index
    assert isinstance(lseries, LabelledSeries)
    assert isinstance(lseries.to_frame(), LabelledDataFrame)
    assert isinstance(lframe, LabelledDataFrame)
    assert isinstance(lframe[0], LabelledSeries)
    assert isinstance(lframe.loc['A'], LabelledSeries)


def test_transpose(lframe_index):
    lframe = lframe_index
    lframe._labels = ("labels0", "labels")
    assert lframe.T._labels == ("labels", "labels0")
    assert lframe.transpose()._labels == ("labels", "labels0")


@pytest.fixture
def test_multiaxis(lframe_index):
    lframe = lframe_index.iloc[:4]
    lframe._labels = ('labels', 'aliases')
    columns = MultiIndex.from_arrays([['one', 'two', 'three', 'four'],
                                      [1, 2, 3, 4]],
                                     names=[None, 'aliases'])
    lframe.columns = columns
    result = lframe.copy()

    assert_array_equal(lframe.get_labels(0), ['$A$', '$B$', '$C$', '$D$'])
    assert_array_equal(lframe.get_labels(1), [1, 2, 3, 4])
    assert_array_equal(lframe.T.get_labels(1), ['$A$', '$B$', '$C$', '$D$'])
    assert_array_equal(lframe.T.get_labels(0), [1, 2, 3, 4])

    assert lframe.islabelled(0)
    assert lframe.islabelled(1)

    assert isinstance(lframe.loc['A'], LabelledSeries)
    assert isinstance(lframe['one'], LabelledSeries)
    lframe._labels = ('labels', 'foo')
    assert isinstance(lframe.loc['A'], LabelledSeries)
    assert isinstance(lframe['one'], LabelledDataFrame)

    assert lframe.islabelled(0)
    assert not lframe.islabelled(1)

    lframe._labels = ('labels', 'foo')
    assert isinstance(lframe['one'], LabelledDataFrame)

    assert lframe.islabelled(0)
    assert not lframe.islabelled(1)

    return result


def test_set_label(lseries):
    assert isinstance(lseries.get_labels_map(), Series)

    nolabels_map = lseries.drop_labels().get_labels_map()
    assert isinstance(nolabels_map, Series)
    assert_array_equal(nolabels_map.index, lseries.drop_labels().index)
    assert nolabels_map['A'] == 'A'
    assert nolabels_map['B'] == 'B'

    assert lseries.get_label('A') == '$A$'
    assert lseries.set_label('A', '$a$').get_label('A') == '$a$'
    lseries.set_label('A', '$a$')
    assert lseries.get_label('A') == '$A$'
    lseries.set_label('A', '$a$', inplace=True)
    assert lseries.get_label('A') == '$a$'

    lseries = lseries.drop_labels()
    assert lseries.get_label('A') == 'A'
    assert lseries.index.nlevels == 1

    lseries.set_label('A', '$A$', inplace=True)
    assert lseries.get_label('A') == '$A$'
    assert lseries.get_label('B') == 'B'
    assert lseries.index.nlevels == 2

    lseries['Z'] = 1.0
    assert lseries.get_label('Z') == 'Z'


def test_multiaxis_slice(test_multiaxis):
    lframe = test_multiaxis
    assert_series_equal_not_name(lframe['one'], lframe[('one', 1)])
    assert_series_equal_not_name(lframe.loc['A'], lframe.loc[('A', '$A$')])
    assert_series_equal_not_name(lframe.loc[:, 'one'],
                                 lframe.loc[:, ('one', 1)])

    assert_frame_equal(lframe.loc[['A', 'B'], ['one', 'two']],
                       lframe.loc[[('A', '$A$'), ('B', '$B$')],
                                  [('one', 1), ('two', 2)]])

    assert_series_equal_not_name(lframe.loc['A', ['one', 'two']],
                                 lframe.loc[('A', '$A$'),
                                            [('one',  1), ('two', 2)]])
    assert_series_equal_not_name(lframe.loc[['A', 'B'], 'one'],
                                 lframe.loc[[('A', '$A$'), ('B', '$B$')],
                                            ('one',  1)])


def test_reset_index(lframe_index):
    ldf = lframe_index
    assert ldf.reset_index().index.names == [None, 'labels']

    assert not np.array_equal(ldf.reset_index().columns, ldf.columns)
    assert_array_equal(ldf.reset_index(drop=True).columns, ldf.columns)

    new = ldf.copy()
    assert not np.array_equal(new.index, ldf.reset_index())
    new.reset_index(inplace=True)
    assert_array_equal(new.index, ldf.reset_index().index)


def test_drop_labels(lframe_index):
    ldf = lframe_index
    assert ldf.islabelled()
    nolabels = ldf.drop_labels()
    assert not nolabels.islabelled()
    assert_frame_equal_not_index(ldf, nolabels)
    assert_frame_equal(nolabels.drop_labels(), nolabels)
    assert nolabels.drop_labels() is not nolabels
