from anesthetic import MCMCSamples
from anesthetic.convert import to_getdist
from numpy.testing import assert_array_equal


def test_to_getdist():
    try:
        anesthetic_samples = MCMCSamples(root='./tests/example_data/gd')
        getdist_samples = to_getdist(anesthetic_samples)

        assert_array_equal(getdist_samples.samples, anesthetic_samples)
        assert_array_equal(getdist_samples.weights, anesthetic_samples.weights)

        for param, p in zip(getdist_samples.getParamNames().names,
                            anesthetic_samples.columns):
            assert param.name == p
    except ModuleNotFoundError:
        pass
