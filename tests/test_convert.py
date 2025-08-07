import pytest
from numpy.testing import assert_array_equal
from anesthetic import read_chains
from anesthetic.convert import to_chainconsumer, from_chainconsumer, to_getdist
from anesthetic.samples import MCMCSamples
from utils import getdist_mark_xfail, chainconsumer_mark_xfail


def _filter_zero_weights(samples, params=None):
    """Helper function to filter zero weights like to_chainconsumer does."""
    weights = samples.get_weights()
    valid_mask = weights > 0
    
    if params is not None:
        positions = [samples.drop_labels().columns.get_loc(p) for p in params]
        filtered_data = samples.to_numpy()[valid_mask][:, positions]
    else:
        filtered_data = samples.to_numpy()[valid_mask]
    
    filtered_weights = weights[valid_mask]
    return filtered_data, filtered_weights, valid_mask


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
    """Test for to_chainconsumer conversion to Chain object."""
    s1 = read_chains('./tests/example_data/gd')
    s1.label = 'Sample 1'
    
    params = ['x0', 'x1']

    # Test 1: Basic conversion with default name
    chain = to_chainconsumer(s1)
    assert chain.name == 'Sample 1'
    
    expected_data, expected_weights, _ = _filter_zero_weights(s1)
    assert_array_equal(chain.samples[s1.get_labels().tolist()].values, expected_data)
    assert_array_equal(chain.weights, expected_weights)

    # Test 2: Conversion with specified params and name
    chain = to_chainconsumer(s1, params=params, name='test_chain')
    assert chain.name == 'test_chain'
    expected_labels = s1[params].get_labels().tolist()
    
    expected_param_data, _, _ = _filter_zero_weights(s1, params=params)
    assert_array_equal(chain.samples[expected_labels].values, expected_param_data)

    # Test 3: Test with unlabelled samples
    s1_unlabelled = s1.drop_labels()
    chain = to_chainconsumer(s1_unlabelled, params=params)
    
    expected_param_data, _, _ = _filter_zero_weights(s1_unlabelled, params=params)
    assert_array_equal(chain.samples[params].values, expected_param_data)
    
    # Test 4: Test with samples that have logL
    if hasattr(s1, 'logL') and 'logL' in s1.columns:
        chain = to_chainconsumer(s1)
        assert 'log_posterior' in chain.samples.columns
    
    # Test 5: Test default name when label is None
    s1_no_label = s1.copy()
    s1_no_label.label = None
    chain_default_name = to_chainconsumer(s1_no_label)
    assert chain_default_name.name == 'anesthetic_chain'
    
    # Test 6: Test kwargs passing
    chain_with_kwargs = to_chainconsumer(s1, color='red', linestyle='--')
    # Note: These would be stored in the Chain object properties if ChainConsumer supports them


@chainconsumer_mark_xfail
def test_to_chainconsumer_latex_labels():
    """Test that LaTeX labels are properly used in Chain object."""
    s1 = read_chains('./tests/example_data/gd')
    
    # Test that when samples have LaTeX labels, they are used as column names
    chain = to_chainconsumer(s1)
    if s1.islabelled():
        expected_labels = s1.get_labels().tolist()
        actual_columns = [col for col in chain.samples.columns if col not in ['weight', 'log_posterior']]
        assert actual_columns == expected_labels


@chainconsumer_mark_xfail
def test_from_chainconsumer_conversion():
    """Test for from_chainconsumer conversion from Chain object."""
    s1 = read_chains('./tests/example_data/gd')
    s1.label = 'test_chain'
    
    params = ['x0', 'x1']

    # Convert to Chain and back
    chain = to_chainconsumer(s1, params=params)
    samples_back = from_chainconsumer(chain)
    
    # Test basic conversion
    assert isinstance(samples_back, MCMCSamples)
    assert_array_equal(samples_back.get_weights(), chain.weights)
    
    # Test with specified columns
    samples_with_cols = from_chainconsumer(chain, columns=params)
    # When columns are specified, logL should not be automatically added
    assert len(samples_with_cols.columns.get_level_values(0)) == len(params)
    
    # Test with no columns specified (full conversion)
    samples_full = from_chainconsumer(chain)
    # Should include all data columns plus logL if available
    if chain.log_posterior is not None:
        assert 'logL' in samples_full.columns
    expected_cols = len(chain.data_columns) + (1 if chain.log_posterior is not None else 0)
    assert len(samples_full.columns.get_level_values(0)) == expected_cols
