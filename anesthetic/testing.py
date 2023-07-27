"""Anesthetic testing utilities."""
import pandas.testing
import numpy.testing


def assert_frame_equal(left, right, *args, **kwargs):
    """Assert frames are equal, including metadata."""
    check_metadata = kwargs.pop('check_metadata', True)
    pandas.testing.assert_frame_equal(left, right, *args, **kwargs)
    numpy.testing.assert_array_equal(left._metadata, right._metadata)
    if check_metadata:
        for key in left._metadata:
            assert getattr(left, key) == getattr(right, key)
