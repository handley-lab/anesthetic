
    #if hasattr(reader, 'birth_file') or hasattr(reader, 'ev_file'):
    #    raise ValueError("The file root %s seems to point to a Nested "
    #                     "Sampling chain. Please use NestedSamples "
    #                     "instead which has the same features as "
    #                     "Samples and more. MCMCSamples should be "
    #                     "used for MCMC chains only." % root)
