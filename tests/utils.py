import pytest
import sys


condition = 'astropy' not in sys.modules
reason = "requires astropy package"
astropy_mark_skip = pytest.mark.skipif(condition, reason=reason)


def astropy_skipif(param):
    return pytest.param(param, marks=astropy_mark_skip)


reason = "You need to install fastkde to use fastkde"
condition = 'fastkde' not in sys.modules
fastkde_mark_skip = pytest.mark.skipif(condition, reason=reason)


def fastkde_skipif(param):
    return pytest.param(param, marks=fastkde_mark_skip)
