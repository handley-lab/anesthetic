from anesthetic.utils import nest_level

def test_nest_level():
    assert(nest_level(0) == 0)
    assert(nest_level([]) == 1)
    assert(nest_level(['a']) == 1)
    assert(nest_level(['a','b']) == 1)
    assert(nest_level([['a'],'b']) == 2)
    assert(nest_level(['a',['b']]) == 2)
    assert(nest_level([['a'],['b']]) == 2)
