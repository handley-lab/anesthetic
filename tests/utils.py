from importlib.util import find_spec
import pytest

try:
    import astropy  # noqa: F401
except ImportError:
    pass

condition = find_spec('astropy') is None
reason = "requires astropy package"
raises = ImportError
astropy_mark_skip = pytest.mark.skipif(condition, reason=reason)
astropy_mark_xfail = pytest.mark.xfail(condition, raises=raises, reason=reason)


def skipif_no_astropy(param):
    return pytest.param(param, marks=astropy_mark_skip)


try:
    import fastkde  # noqa: F401
except ImportError:
    pass
reason = "requires fastkde package"
condition = find_spec('fastkde') is None
raises = ImportError
fastkde_mark_skip = pytest.mark.skipif(condition, reason=reason)
fastkde_mark_xfail = pytest.mark.xfail(condition, raises=raises, reason=reason)


def skipif_no_fastkde(param):
    return pytest.param(param, marks=fastkde_mark_skip)


try:
    import getdist  # noqa: F401
except ImportError:
    pass
condition = find_spec('getdist') is None
reason = "requires getdist package"
raises = ImportError
getdist_mark_skip = pytest.mark.skipif(condition, reason=reason)
getdist_mark_xfail = pytest.mark.xfail(condition, raises=raises, reason=reason)


def skipif_no_getdist(param):
    return pytest.param(param, marks=getdist_mark_skip)


try:
    from tables import Array  # noqa: F401
    pytables_imported = True
except ImportError:
    pytables_imported = False

reason = "requires tables package"
raises = ImportError
pytables_mark_skip = pytest.mark.skipif(not pytables_imported, reason=reason)
pytables_mark_xfail = pytest.mark.xfail(not pytables_imported, raises=raises,
                                        reason=reason)


try:
    import h5py  # noqa: F401
except ImportError:
    pass

condition = find_spec('h5py') is None
reason = "requires h5py package"
raises = ImportError
h5py_mark_skip = pytest.mark.skipif(condition, reason=reason)
h5py_mark_xfail = pytest.mark.xfail(condition, raises=raises, reason=reason)


def skipif_no_h5py(param):
    return pytest.param(param, marks=h5py_mark_skip)
