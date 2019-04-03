---
title: 'anesthetic: nested sampling post-processing'
tags:
  - Python
  - Statistics
  - Bayesian inference
  - Nested sampling
  - Astronomy
authors:
  - name: Will Handley
    orcid: 0000-0002-5866-0445
    affiliation: "1, 2, 3"
affiliations:
 - name: Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, UK
   index: 1
 - name: Kavli Institute for Cosmology, Madingley Road, Cambridge, CB3 0HA, UK
   index : 2
 - name: Gonville & Caius College, Trinity Street, Cambridge, CB2 1TA, UK
   index: 3


date: 05 April 2019
bibliography: paper.bib
---

# Summary

Contour plots such as Figure 1 can be created using two-dimensional kernel
density estimation using packages such as
[scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
[@scipy], [getdist](http://getdist.readthedocs.io/en/latest/intro.html)
[@getdist], [corner](https://corner.readthedocs.io/en/latest/) [@corner] and
[pygtc](https://pygtc.readthedocs.io/en/latest/) [@pygtc], where the samples
provided as inputs to such programs are typically created by a
Markov Chain Monte Carlo (MCMC) analysis. For further information on MCMC and
Bayesian analysis in general, "Information Theory, Inference and Learning
Algorithms" is highly recommended [@mackay], which is available freely
[online](http://www.inference.org.uk/itprnn/book.html).

``anesthetic`` is a Python package for plotting marginalised probability
distributions, currently used in astronomy, but will be of use to scientists
performing any Bayesian analysis which has predictive posteriors that are
functions. The source code for ``fgivenx`` is available on
[GitHub](https://github.com/williamjameshandley/anesthetic) and has been archived
as ``v1.0.0`` to Zenodo with the linked DOI: [@zenodo].

# Acknowledgements

Contributions and bug-testing were provided by Ed Higson and Sonke Hee.

# References
