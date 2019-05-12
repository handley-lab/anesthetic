---
title: 'anesthetic: nested sampling visualisation'
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


date: 19 April 2019
bibliography: paper.bib
---

# Summary
``anesthetic`` is a Python package for processing nested sampling runs, and will be useful for any scientist or statistician who uses nested sampling software. ``anesthetic`` unifies many existing tools and techniques in an extensible framework that is intuitive for users familiar with the standard Python packages, namely [NumPy](https://www.numpy.org/), [SciPy](https://www.scipy.org/), [Matplotlib](https://matplotlib.org/) and [pandas](https://pandas.pydata.org/). It has been extensively used in recent cosmological papers [@tension;@dimensionality].



# Nested sampling

Nested sampling [@skilling] is an alternative to Markov-Chain-Monte-Carlo techniques [@mcmc]. Given some data $D$, for a scientific model $M$ with free parameters $\theta$, Bayes theorem states:

$$ P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}. $$

Traditional MCMC approaches ignore the Bayesian evidence $P(D)$ and instead focus on the problem of generating samples from the posterior $P(\theta|D)$ using knowledge of the prior $P(\theta)$ and likelihood $P(D|\theta)$. Nested sampling reverses this priority, and instead computes the evidence $P(D)$ (the critical quantity in Bayesian model comparison [@trotta]), producing posterior samples as a by-product. Nested sampling does this by evolving a set of live points drawn from the prior under a hard likelihood constraint which steadily increases, causing the live points to contract around the peak(s) of the likelihood. The history of the live-point evolution can be used to reconstruct both the evidence and posterior samples, as well as the density of states and consequently the full partition function.

Current publicly available implementations of nested sampling include MultiNest [@multinest], PolyChord [@polychord0;@polychord1;@dypolychord], DNest [@dnest] and dynesty [@dynesty], all of which have been incorporated into a wide range of cosmological [@cosmomc;@cosmosis;@montepython] and particle physics [@gambit] codes.

![Marginalised posterior plots produced by ``anesthetic``. The x axes indicate the fraction of normal matter, dark matter and dark energy respectively, whilst the y-axis is the amplitude of mass fluctuation in our late-time universe. The three measurements were performed using measurements of baryonic acoustic oscillations, large scale structure and the cosmic microwave background [@tension]. It is an open cosmological and statistical questions whether the LSS and CMB are consistent with one another.](2d.png) 

# aNESThetic
``anesthetic`` acts on outputs of nested sampling software packages. It can:

1. Compute inferences of the Bayesian evidence [@trotta], the Kullback-Leibler
   divergence [@KL] of the distribution, the Bayesian model
   dimensionality [@dimensionality] and the full partition function.
2. Dynamically replay nested sampling runs.
3. Produce one- and two-dimensional marginalised posterior plots (Figure 1).

A subset of computations from item 1 is provided by many of the nested sampling software packages. ``anesthetic`` allows you to compute these independently and more accurately, providing a unified set of outputs and separating these computations from the generation of nested samples.

Item 2 is useful for users that have experienced the phenomenon of 'live point watching' -- the process of continually examining the evolution of the live points as the run progresses in an attempt to diagnose problems in likelihood and/or sampling implementations. The GUI provided allows users to fully reconstruct the run at any iteration, and examine the effect of dynamically adjusting the thermodynamic temperature.

Finally, it is important to recognise that the functionality from item 3 is also provided by many other high-quality software packages, such as getdist [@getdist], [corner](https://corner.readthedocs.io/en/latest/) [@corner], [pygtc](https://pygtc.readthedocs.io/en/latest/) [@pygtc], [dynesty](https://dynesty.readthedocs.io) [@dynesty] and [MontePython](http://baudren.github.io/montepython.html) [@montepython]. ``anesthetic`` adds to this functionality by: 

- Performing kernel density estimation using the state-of-the-art
  [fastkde](https://pypi.org/project/fastkde/) [@fastkde] algorithm.
- Storing samples and plotting grids as a weighted ``pandas.DataFrame``, which
  is more consistent with the scientific Python canon, allows for unambiguous
  access to samples and plots via their reference names, and easy definition of
  new parameters.
- Using a contour colour scheme that is better suited to plotting distributions
  with uniform probability, which is important if one wishes to plot priors
  along with posteriors.

The source code for ``anesthetic`` is available on GitHub, with its automatically generated documentation at [ReadTheDocs](https://anesthetic.readthedocs.io/) and a pip-installable package on [PyPi](https://pypi.org/project/anesthetic/). An example interactive Jupyter notebook is given using [Binder](https://mybinder.org/v2/gh/williamjameshandley/anesthetic/master?filepath=demo.ipynb) [@binder]. Continuous integration is implemented with [Travis](https://travis-ci.org/williamjameshandley/anesthetic) and [Circle](https://circleci.com/gh/williamjameshandley/anesthetic).

# Acknowledgements

Bug-testing was provided by Pablo Lemos.

# References
