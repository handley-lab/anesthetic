def replace_inner_function(outer, new_inner):
    """Replace a nested function code object used by outer with new_inner.

    The replacement new_inner must use the same name and must at most use the
    same closures as the original.

    Code taken from https://stackoverflow.com/a/27550237
    """
    if hasattr(new_inner, '__code__'):
        # support both functions and code objects
        new_inner = new_inner.__code__

    # find original code object so we can validate the closures match
    ocode = outer.__code__
    function, code = type(outer), type(ocode)
    iname = new_inner.co_name
    orig_inner = next(
        const for const in ocode.co_consts
        if isinstance(const, code) and const.co_name == iname)

    # you can ignore later closures, but since they are matched by position
    # the new sequence must match the start of the old.
    assert (orig_inner.co_freevars[:len(new_inner.co_freevars)] ==
            new_inner.co_freevars), 'New closures must match originals'

    # replace the code object for the inner function
    new_consts = tuple(
        new_inner if const is orig_inner else const
        for const in outer.__code__.co_consts)

    # create a new code object with the new constants
    try:
        # Python 3.8 added code.replace(), so much more convenient!
        ncode = ocode.replace(co_consts=new_consts)
    except AttributeError:
        # older Python versions, argument counts vary so we need to check
        # for specifics.
        args = [
            ocode.co_argcount, ocode.co_nlocals, ocode.co_stacksize,
            ocode.co_flags, ocode.co_code,
            new_consts,  # replacing the constants
            ocode.co_names, ocode.co_varnames, ocode.co_filename,
            ocode.co_name, ocode.co_firstlineno, ocode.co_lnotab,
            ocode.co_freevars, ocode.co_cellvars,
        ]
        if hasattr(ocode, 'co_kwonlyargcount'):
            # Python 3+, insert after co_argcount
            args.insert(1, ocode.co_kwonlyargcount)
        # Python 3.8 adds co_posonlyargcount, but also has code.replace(), used
        # above
        ncode = code(*args)

    # and a new function object using the updated code object
    return function(
        ncode, outer.__globals__, outer.__name__,
        outer.__defaults__, outer.__closure__
    )
