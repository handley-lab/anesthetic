from numpy.testing import assert_array_equal

from anesthetic import read_chains
from anesthetic.convert import to_chainconsumer, from_chainconsumer, to_getdist
from anesthetic.samples import MCMCSamples
from utils import getdist_mark_xfail, chainconsumer_mark_xfail

try:
    from importlib.metadata import version, PackageNotFoundError
    cc_version = version('chainconsumer')
    CHAINCONSUMER_V1 = cc_version.startswith('1.')
except PackageNotFoundError:
    CHAINCONSUMER_V1 = False


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
def test_to_chainconsumer_v1():
    """Comprehensive test for to_chainconsumer v1+ functionality."""
    if not CHAINCONSUMER_V1:
        return

    s1 = read_chains('./tests/example_data/gd')
    s1.label = 'Sample 1'
    params = ['x0', 'x1']

    # Test 1: Basic conversion of a single sample
    chain = to_chainconsumer(s1)
    assert chain.name == 'Sample 1'
    # The conversion should preserve the parameter columns that exist
    expected_labels = s1.get_labels().tolist()
    actual_cols = [c for c in chain.samples.columns
                   if c not in ['weight', 'log_posterior']]
    # Check that labels match what's available
    available_labels = s1.get_labels().tolist()
    assert actual_cols == available_labels

    # Test 2: Conversion with specified params and name
    chain = to_chainconsumer(s1, params=params, name='test_chain')
    assert chain.name == 'test_chain'

    # Test 3: Test LaTeX labels are properly used with full conversion
    chain_full = to_chainconsumer(s1)
    if s1.islabelled():
        expected_labels = s1.get_labels().tolist()
        actual_columns = [col for col in chain_full.samples.columns
                          if col not in ['weight', 'log_posterior']]
        assert actual_columns == expected_labels

    # Test 4: Test log_posterior handling
    if 'logL' in s1:
        assert 'log_posterior' in chain.samples.columns

    # Test 5: Test default naming when no label
    s1_no_label = s1.copy()
    s1_no_label.label = None
    chain_default_name = to_chainconsumer(s1_no_label)
    assert chain_default_name.name == 'anesthetic_chain'

    # Test 6: Test zero weight filtering (automatic in v1.x)
    import numpy as np
    np.random.seed(42)
    data = np.random.randn(50, 2)
    weights = np.ones(50)
    weights[0:5] = 0
    logL = np.random.randn(50)
    samples_data = np.column_stack([data, logL])
    samples = MCMCSamples(samples_data, weights=weights,
                          columns=['x', 'y', 'logL'])

    chain = to_chainconsumer(samples)
    assert len(chain.weights) == 45
    assert all(w > 0 for w in chain.weights)
    if 'log_posterior' in chain.samples.columns:
        assert len(chain.samples['log_posterior']) == 45
    assert len(chain.weights) == len(chain.samples)
    assert len(chain.weights) == 45

    # Test 7: Error handling - Multiple samples not supported in v1.x
    s2 = read_chains('./tests/example_data/pc')
    try:
        to_chainconsumer([s1, s2])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "only supports converting a single" in str(e)


@chainconsumer_mark_xfail
def test_to_chainconsumer_v0():
    """Comprehensive test for to_chainconsumer v0.x functionality."""
    if CHAINCONSUMER_V1:
        return

    s1 = read_chains('./tests/example_data/gd')
    s2 = read_chains('./tests/example_data/pc')
    s1.label = 'Sample 1'
    s2.label = None
    params = ['x0', 'x1']

    # Test 1: Basic conversion of a single sample
    c = to_chainconsumer(s1)
    assert len(c.chains) == 1
    assert c.chains[0].name.strip() == 'Sample 1'
    assert_array_equal(c.chains[0].parameters, s1.get_labels().tolist())

    # Test 2: Conversion with specified params and name
    c = to_chainconsumer(s1, params=params, name='test_chain')
    assert len(c.chains) == 1
    chain = c.chains[0]
    assert chain.name.strip() == 'test_chain'
    assert_array_equal(chain.chain, s1[params].to_numpy())
    assert_array_equal(chain.parameters, s1[params].get_labels().tolist())

    # Test 3: Multiple chains with specified names
    c = to_chainconsumer([s1, s2], params=params, name=['c1', 'c2'])
    assert len(c.chains) == 2
    assert c.chains[0].name.strip() == 'c1'
    assert c.chains[1].name.strip() == 'c2'

    # Test 4: Test default naming (s1 has a label, s2 does not)
    c = to_chainconsumer([s1, s2])
    assert len(c.chains) == 2
    assert c.chains[0].name.strip() == 'Sample 1'
    assert c.chains[1].name.strip() == 'chain2'  # Falls back to default naming

    # Test 5: Add to an existing ChainConsumer object
    from chainconsumer import ChainConsumer
    c_existing = ChainConsumer()
    c_existing.add_chain(s1.to_numpy(), name="pre-existing")
    to_chainconsumer(s2, cc=c_existing)
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

    # Test 7: unlabelled sample
    s2_unlabelled = s2.drop_labels()
    c_unlabelled = to_chainconsumer(s2_unlabelled, params=params)
    assert c_unlabelled.chains[0].parameters == params

    # Test 8: Test weights are preserved
    s1_copy = s1.copy()
    original_weights = s1_copy.get_weights().copy()
    original_zero_count = sum(w == 0 for w in original_weights)

    nonzero_indices = [i for i, w in enumerate(original_weights) if w > 0][:2]
    modified_weights = original_weights.copy()
    for idx in nonzero_indices:
        modified_weights[idx] = 0

    s1_copy.set_weights(modified_weights, inplace=True)

    c = to_chainconsumer(s1_copy)
    assert len(c.chains[0].weights) == len(s1_copy)
    expected_zero_count = original_zero_count + len(nonzero_indices)
    assert sum(w == 0 for w in c.chains[0].weights) == expected_zero_count

    # Test 9: Error handling - String name with multiple samples
    try:
        to_chainconsumer([s1, s2], name="single_name")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "String 'names' is only valid" in str(e)

    # Test 10: Error handling - Mismatched names length
    try:
        to_chainconsumer([s1, s2], name=["only_one_name"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Length of 'names' must match" in str(e)

    # Test 11: Error handling - chain_specific_kwargs wrong type
    try:
        to_chainconsumer([s1, s2],
                         chain_specific_kwargs={'shade_alpha': 0.5})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "chain_specific_kwargs must be a list of dictionaries" in str(e)

    # Test 12: Error handling - Mismatched chain_specific_kwargs length
    try:
        to_chainconsumer([s1, s2],
                         chain_specific_kwargs=[{'shade_alpha': 0.5}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert ("chain_specific_kwargs must be a list with the same "
                "length as samples" in str(e))


@chainconsumer_mark_xfail
def test_from_chainconsumer_v1():
    """Comprehensive test for from_chainconsumer v1+ functionality."""
    if not CHAINCONSUMER_V1:
        return

    s1 = read_chains('./tests/example_data/gd')
    s1.label = 'chain1'
    params = ['x0', 'x1']

    # Test 1: Basic conversion back from Chain object
    chain_v1 = to_chainconsumer(s1, params=params)
    samples_back = from_chainconsumer(chain_v1)

    assert isinstance(samples_back, MCMCSamples)
    assert_array_equal(samples_back.get_weights(), chain_v1.weights)
    if chain_v1.log_posterior is not None:
        assert 'logL' in samples_back.columns

    # Test 2: Test conversion with full dataset
    chain_full = to_chainconsumer(s1)
    samples_full_back = from_chainconsumer(chain_full)

    assert isinstance(samples_full_back, MCMCSamples)
    expected_labels = s1.get_labels().tolist()
    actual_labels = [col for col in
                     samples_full_back.columns.get_level_values(0)
                     if col not in ['logL']]
    assert actual_labels == expected_labels

    # Test 3: Test with specified columns
    chain_cols = to_chainconsumer(s1, params=params)
    samples_cols_back = from_chainconsumer(chain_cols, columns=params)

    assert list(samples_cols_back.columns.get_level_values(0)) == params
    assert_array_equal(samples_cols_back.to_numpy(),
                       s1[params].to_numpy()[s1.get_weights() > 0])


@chainconsumer_mark_xfail
def test_from_chainconsumer_v0():
    """Comprehensive test for from_chainconsumer v0.x functionality."""
    if CHAINCONSUMER_V1:
        return

    s1 = read_chains('./tests/example_data/gd')
    s2 = read_chains('./tests/example_data/pc')
    s1.label = 'chain1'
    s2.label = 'chain2'
    params = ['x0', 'x1']

    # Test 1: Conversion of multiple chains
    c_v0 = to_chainconsumer([s1, s2], params=params)
    samples_dict = from_chainconsumer(c_v0)
    assert isinstance(samples_dict, dict)
    assert len(samples_dict) == 2
    assert 'chain1' in samples_dict and 'chain2' in samples_dict

    chain1_original = c_v0.chains[0]
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
    samples_dict_cols = from_chainconsumer(c_v0, columns=params)
    chain1_converted_cols = samples_dict_cols['chain1']
    assert list(chain1_converted_cols.columns.get_level_values(0)) == params

    # Test 4: Test full conversion roundtrip
    c_full = to_chainconsumer([s1, s2])
    samples_full_dict = from_chainconsumer(c_full)

    for name, original_sample in [('chain1', s1), ('chain2', s2)]:
        converted_sample = samples_full_dict[name]
        assert isinstance(converted_sample, MCMCSamples)
        assert_array_equal(converted_sample.get_weights(),
                           original_sample.get_weights())
