from anesthetic.weighted_labelled_pandas import (WeightedLabelledDataFrame,
                                                 read_csv)
import numpy as np
import pandas


def test_read_csv(tmp_path):
    filename = tmp_path / 'mcmc_wldf.csv'

    wlframe = WeightedLabelledDataFrame(np.random.rand(3, 3),
                                        index=[0, 1, 2],
                                        columns=['a', 'b', 'c'])
    wlframe.to_csv(filename)
    wlframe_ = read_csv(filename)
    pandas.testing.assert_frame_equal(wlframe, wlframe_)

    wlframe.set_labels(['$A$', '$B$', '$C$'], inplace=True)
    wlframe.to_csv(filename)
    wlframe_ = read_csv(filename)
    pandas.testing.assert_frame_equal(wlframe, wlframe_)

    wlframe.set_weights([0.5, 0.6, 0.7], inplace=True)
    wlframe.to_csv(filename)
    wlframe_ = read_csv(filename)
    pandas.testing.assert_frame_equal(wlframe, wlframe_)

    wlframe = wlframe.drop_labels()
    wlframe.to_csv(filename)
    wlframe_ = read_csv(filename)
    pandas.testing.assert_frame_equal(wlframe, wlframe_)
