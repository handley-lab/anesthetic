from anesthetic import read_chains
from anesthetic.convert import to_getdist, from_chainconsumer, to_chainconsumer
from numpy.testing import assert_array_equal
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
    anesthetic_samples = read_chains('./tests/example_data/gd')

    # Test 1: No specification (uses all parameters, default names)
    chainconsumer_obj = to_chainconsumer(anesthetic_samples)
    assert len(chainconsumer_obj.chains) == 1
    chain = chainconsumer_obj.chains[0]
    assert chain.name == 'gd'  # Uses sample label as default name

    assert_array_equal(chain.chain,
                       anesthetic_samples.drop_labels().to_numpy())
    assert_array_equal(chain.weights, anesthetic_samples.get_weights())
    assert_array_equal(chain.parameters,
                       anesthetic_samples.get_labels().tolist())

    # Test 2: Specified params, name, and color
    params = ['x0', 'x1']
    chainconsumer_obj = to_chainconsumer(anesthetic_samples, params,
                                         names='test_chain', colors='red')

    assert len(chainconsumer_obj.chains) == 1
    chain = chainconsumer_obj.chains[0]
    assert chain.name == 'test_chain'

    assert_array_equal(chain.chain, anesthetic_samples[params].to_numpy())
    assert_array_equal(chain.weights, anesthetic_samples.get_weights())
    assert_array_equal(chain.parameters, ['$x_0$', '$x_1$'])

    # Test 3: Multiple chains
    anesthetic_samples2 = read_chains('./tests/example_data/pc')
    samples_list = [anesthetic_samples, anesthetic_samples2]

    chainconsumer_obj = to_chainconsumer(samples_list, params,
                                         names=['chain1', 'chain2'],
                                         colors=['blue', 'red'])

    assert len(chainconsumer_obj.chains) == 2

    # Check first chain
    chain1 = chainconsumer_obj.chains[0]
    assert chain1.name == 'chain1'
    assert_array_equal(chain1.chain,
                       anesthetic_samples[params].to_numpy())
    assert_array_equal(chain1.weights, anesthetic_samples.get_weights())
    assert_array_equal(chain1.parameters, ['$x_0$', '$x_1$'])

    # Check second chain
    chain2 = chainconsumer_obj.chains[1]
    assert chain2.name == 'chain2'
    assert_array_equal(chain2.chain,
                       anesthetic_samples2[params].to_numpy())
    assert_array_equal(chain2.weights, anesthetic_samples2.get_weights())
    assert_array_equal(chain2.parameters, ['$x_0$', '$x_1$'])


@chainconsumer_mark_xfail
def test_from_chainconsumer():
    anesthetic_samples = read_chains('./tests/example_data/gd')

    # Test 1: Single chain without parameter or name specification
    chainconsumer_obj = to_chainconsumer(anesthetic_samples)
    converted_samples = from_chainconsumer(chainconsumer_obj)

    chain = chainconsumer_obj.chains[0]

    assert_array_equal(converted_samples.to_numpy(), chain.chain)
    assert_array_equal(converted_samples.get_weights(), chain.weights)
    assert_array_equal(converted_samples.get_labels().tolist(),
                       chain.parameters)

    # Test 2: With parameter and name specification
    params = ['x0', 'x1']
    chainconsumer_obj = to_chainconsumer(anesthetic_samples, params,
                                         names='test_chain',
                                         colors='red')
    converted_samples = from_chainconsumer(chainconsumer_obj,
                                           columns=params)

    chain = chainconsumer_obj.chains[0]

    assert chain.name == 'test_chain'
    assert_array_equal(converted_samples[params].to_numpy(), chain.chain)
    assert_array_equal(converted_samples.get_weights(), chain.weights)
    assert converted_samples.columns.nlevels == 2
    params_level = converted_samples.columns.get_level_values(0)[:2]
    assert list(params_level) == params
    assert_array_equal(chain.parameters, ['$x_0$', '$x_1$'])

    # Test 3: Multiple chains
    anesthetic_samples2 = read_chains('./tests/example_data/pc')
    samples_list = [anesthetic_samples, anesthetic_samples2]

    chainconsumer_obj = to_chainconsumer(samples_list, params,
                                         names=['chain1', 'chain2'])
    samples_dict = from_chainconsumer(chainconsumer_obj,
                                      columns=params)

    assert len(samples_dict) == 2
    assert 'chain1' in samples_dict
    assert 'chain2' in samples_dict

    converted_samples1 = samples_dict['chain1']
    converted_samples2 = samples_dict['chain2']

    chain1 = chainconsumer_obj.chains[0]
    chain2 = chainconsumer_obj.chains[1]
    assert_array_equal(converted_samples1[params].to_numpy(), chain1.chain)
    assert_array_equal(converted_samples1.get_weights(), chain1.weights)
    assert_array_equal(converted_samples2[params].to_numpy(), chain2.chain)
    assert_array_equal(converted_samples2.get_weights(), chain2.weights)
    assert converted_samples1.columns.nlevels == 2
    assert converted_samples2.columns.nlevels == 2
    params1_level = converted_samples1.columns.get_level_values(0)[:2]
    assert list(params1_level) == params
    params2_level = converted_samples2.columns.get_level_values(0)[:2]
    assert list(params2_level) == params
