from anesthetic import read_chains
from anesthetic.convert import to_getdist
from numpy.testing import assert_array_equal
from utils import getdist_mark_xfail


@getdist_mark_xfail
def test_to_getdist():
    anesthetic_samples = read_chains('./tests/example_data/gd')
    getdist_samples = to_getdist(anesthetic_samples)

    assert_array_equal(getdist_samples.samples, anesthetic_samples)
    assert_array_equal(getdist_samples.weights,
                       anesthetic_samples.get_weights())

    for param, p in zip(getdist_samples.getParamNames().names,
                        anesthetic_samples.drop_labels().columns):

        assert param.name == p
