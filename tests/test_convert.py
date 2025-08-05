import pytest
from numpy.testing import assert_array_equal
from chainconsumer import ChainConsumer
from anesthetic import read_chains
from anesthetic.convert import to_chainconsumer, from_chainconsumer, to_getdist
from anesthetic.samples import MCMCSamples
from utils import getdist_mark_xfail, chainconsumer_mark_xfail


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


@chainconsumer_mark_xfail
def test_to_chainconsumer():
    """Expanded test for to_chainconsumer to cover all input variations."""
    s1 = read_chains('./tests/example_data/gd')
    s2 = read_chains('./tests/example_data/pc')
    s1.label = 'Sample 1'
    s2.label = None  # For testing default naming

    params = ['x0', 'x1']

    # Test 1: Basic conversion of a single sample
    c = to_chainconsumer(s1)
    assert len(c.chains) == 1
    assert c.chains[0].name.strip() == 'Sample 1'
    assert_array_equal(c.chains[0].parameters, s1.get_labels().tolist())

    # Test 2: Conversion with specified params and name
    c = to_chainconsumer(s1, params=params, names='test_chain')
    assert len(c.chains) == 1
    chain = c.chains[0]
    assert chain.name.strip() == 'test_chain'
    assert_array_equal(chain.chain, s1[params].to_numpy())
    assert_array_equal(chain.parameters, s1[params].get_labels().tolist())

    # Test 3: Multiple chains with specified names
    c = to_chainconsumer([s1, s2], params=params, names=['c1', 'c2'])
    assert len(c.chains) == 2
    assert c.chains[0].name.strip() == 'c1'
    assert c.chains[1].name.strip() == 'c2'

    # Test 4: Test default naming (s1 has a label, s2 does not)
    c = to_chainconsumer([s1, s2])
    assert len(c.chains) == 2
    assert c.chains[0].name.strip() == 'Sample 1'
    assert c.chains[1].name.strip() == 'chain2'  # Falls back to default naming

    # Test 5: Add to an existing ChainConsumer object
    c_existing = ChainConsumer()
    c_existing.add_chain(s1.to_numpy(), name="pre-existing")
    to_chainconsumer(s2, chainconsumer=c_existing)
    assert len(c_existing.chains) == 2
    assert c_existing.chains[0].name.strip() == "pre-existing"
    assert c_existing.chains[1].name.strip() == "chain1"  # Default name

    # Test 6: Test kwargs (single dict and list of dicts)
    c = to_chainconsumer([s1, s2], grid=True, linestyle="--")
    assert c.chains[0].grid is True
    assert c.chains[1].grid is True
    assert c.chains[0].linestyle == "--"

    specific_kwargs_list = [
        {'linestyle': "--", 'shade_alpha': 0.8},
        {'linestyle': ':', 'shade_alpha': 0.6}
    ]

    c = to_chainconsumer([s1, s2], chain_specific_kwargs=specific_kwargs_list)

    assert c.chains[0].linestyle == '--'
    assert c.chains[0].shade_alpha == 0.8
    assert c.chains[1].linestyle == ':'
    assert c.chains[1].shade_alpha == 0.6

    # Test 7: unlabelled sample (covers else branch for labels)
    s2_unlabelled = s2.drop_labels()
    c_unlabelled = to_chainconsumer(s2_unlabelled, params=params)
    assert c_unlabelled.chains[0].parameters == params


@chainconsumer_mark_xfail
def test_to_chainconsumer_error_handling():
    """Test the ValueError exceptions in to_chainconsumer."""
    s1 = read_chains('./tests/example_data/gd')
    s2 = read_chains('./tests/example_data/pc')

    with pytest.raises(ValueError,
                       match="If providing string name, "
                             "samples must be a single object"):
        to_chainconsumer([s1, s2], names='a_single_name')

    with pytest.raises(ValueError,
                       match="Length of names must match "
                             "length of samples list"):
        to_chainconsumer([s1, s2], names=['only_one_name'])

    with pytest.raises(ValueError,
                       match="chain_specific_kwargs must be a "
                             "list with the same length as samples"):
        to_chainconsumer([s1, s2],
                         chain_specific_kwargs=[{'shade_alpha': 0.5}])


@chainconsumer_mark_xfail
def test_from_chainconsumer_conversion():
    """Expanded test for from_chainconsumer to cover all paths."""
    s1 = read_chains('./tests/example_data/gd')
    s2 = read_chains('./tests/example_data/pc')
    s1.label = 'chain1'
    s2.label = 'chain2'

    params = ['x0', 'x1']

    c = to_chainconsumer([s1, s2], params=params)

    # Test 1: Conversion of multiple chains
    samples_dict = from_chainconsumer(c)
    assert isinstance(samples_dict, dict)
    assert len(samples_dict) == 2
    assert 'chain1' in samples_dict
    assert 'chain2' in samples_dict

    chain1_original = c.chains[0]
    chain1_converted = samples_dict['chain1']
    assert_array_equal(chain1_converted.to_numpy(), chain1_original.chain)
    assert_array_equal(chain1_converted.get_weights(), chain1_original.weights)
    assert_array_equal(chain1_converted.columns.get_level_values(0),
                       chain1_original.parameters)

    # Test 2: Conversion of a single chain object
    c_single = to_chainconsumer(s1)
    samples_single = from_chainconsumer(c_single)
    assert isinstance(samples_single, MCMCSamples)
    assert_array_equal(samples_single.to_numpy(), s1.to_numpy())

    # Test 3: Conversion with specified columns
    samples_dict_cols = from_chainconsumer(c, columns=params)
    chain1_converted_cols = samples_dict_cols['chain1']
    assert list(chain1_converted_cols.columns.get_level_values(0)) == params
