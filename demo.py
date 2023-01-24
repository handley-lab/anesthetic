#| # anesthetic plot gallery
#| This functions as both some examples of plots that can be produced, and a tutorial.
#| Any difficulties/issues/requests should be posted as a [GitHub issue](https://github.com/williamjameshandley/anesthetic/issues)

#--------------------------

#| ## Download example data
#| Download some example data from github (or alternatively use your own chains files)
#|
#| This downloads the PLA chains for the planck baseline cosmology,
#| and the equivalent nested sampling chains:

import requests
import tarfile

for filename in ["plikHM_TTTEEE_lowl_lowE_lensing.tar.gz","plikHM_TTTEEE_lowl_lowE_lensing_NS.tar.gz"]:
    github_url = "https://github.com/williamjameshandley/cosmo_example/raw/master/"
    url = github_url + filename
    open(filename, 'wb').write(requests.get(url).content)
    tarfile.open(filename).extractall()


#| ## Marginalised posterior plotting
#| Import anesthetic and load the MCMC samples:

import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes
mcmc_root = 'plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing'
mcmc = read_chains(mcmc_root)

#| We have plotting tools for 1D plots ...

mcmc.plot_1d('omegabh2') ;

#| ... multiple 1D plots ...

mcmc.plot_1d(['omegabh2','omegach2','H0','tau','logA','ns']);

#| ... triangle plots ...

mcmc.plot_2d(['omegabh2','omegach2','H0'], kind={'lower':'kde_2d','diagonal':'kde_1d'});

#| ... triangle plots (with the equivalent scatter plot filling up the left hand side) ...

mcmc.plot_2d(['omegabh2','omegach2','H0']);

#| ... and rectangle plots.

mcmc.plot_2d([['omegabh2','omegach2','H0'], ['logA', 'ns']]);

#| Rectangle plots are pretty flexible with what they can do:

mcmc.plot_2d([['omegabh2','omegach2','H0'], ['H0','omegach2']]);

#| ## Changing the appearance
#| 
#| Anesthetic tries to follow matplotlib conventions as much as possible, so 
#| most changes to the appearance should be relatively straight forward. 
#| Here are some examples:
#| 
#| * figure size:

fig = plt.figure(figsize=(5, 5))
fig, axes = make_2d_axes(['omegabh2', 'omegach2', 'H0'], fig=fig)
mcmc.plot_2d(axes);

#| * legends:

fig, axes = make_2d_axes(['omegabh2', 'omegach2', 'H0'])
mcmc.plot_2d(axes, label='Posterior');
axes.iloc[0, 0].legend(loc='upper left');
fig.tight_layout()

#| * unfilled contours  &  modifying individual axes:

fig, axes = make_2d_axes(['omegabh2', 'omegach2', 'H0'])
mcmc.plot_2d(axes.iloc[0:1, :], kind=dict(upper='kde_2d', lower='kde_2d', diagonal='kde_1d'), fc=None);
mcmc.plot_2d(axes.iloc[1:2, :], kind=dict(upper='kde_2d', lower='kde_2d', diagonal='kde_1d'), fc=None, cmap=plt.cm.Oranges, lw=3);
mcmc.plot_2d(axes.iloc[2:3, :], kind=dict(upper='kde_2d', lower='kde_2d', diagonal='kde_1d'), fc='C2', ec='C3', c='C4', lw=2);

#| ## Defining new parameters
#|
#| You can see that samples are stored as a pandas array

mcmc[:6]

#| Since it's a (weighted) pandas array, we compute things like the mean and variance 
#| of samples

mcmc.mean()

#| We can define new parameters with relative ease.
#| For example, the default cosmoMC setup does not include omegab, only omegabh2:

'omegab' in mcmc

#| However, this is pretty trivial to recompute:

h = mcmc['H0']/100
mcmc['omegab'] = mcmc['omegabh2']/h**2
mcmc.set_label('omegab', '$\Omega_b$')
mcmc.plot_1d('omegab');

#| ## Nested sampling plotting
#| Anethestic really comes to the fore for nested sampling. We can do all of
#| the above, and more with the power that NS chains provide

nested_root = 'plikHM_TTTEEE_lowl_lowE_lensing_NS/NS_plikHM_TTTEEE_lowl_lowE_lensing'
nested = read_chains(nested_root)

#| We can infer the evidence, KL divergence and Bayesian model dimensionality:

ns_output = nested.stats(1000)

#| This is a set of ``MCMCSamples``, with columns yielding the log of the Bayesian evidence 
#| (logZ), the Kullback-Leibler divergence (D) and the Bayesian model dimensionality (d).

ns_output[:6]

#| The evidence, KL divergence and Bayesian model dimensionality, with their corresponding errors, are:

for x in ['logZ', 'D_KL', 'd_G', 'logL_P']:
    print('%10s = %9.2f +/- %4.2f' % (ns_output.get_label(x), ns_output[x].mean(), ns_output[x].std()))

#| Since ``ns_output`` is a set of ``MCMCSamples``, it may be plotted as usual. 
#| Here we illustrate slightly more fine-grained control of the axes construction 
#| (demanding three columns)

from anesthetic import make_1d_axes, MCMCSamples
fig, axes = make_1d_axes(['logZ', 'D_KL', 'd_G', 'logL_P'], ncol=4)
MCMCSamples(ns_output).plot_1d(axes);

#| We can also inspect the correlation between these inferences:

MCMCSamples(ns_output).plot_2d(['logZ','D_KL']);

#| Here is a comparison of the base and NS output

h = nested['H0']/100
nested['omegab'] = nested['omegabh2']/h**2
nested.set_label('omegab', '$\Omega_b$')

axes = mcmc.plot_2d(['sigma8','omegab'])
nested.plot_2d(axes=axes);

#| With nested samples, we can plot the prior (or any temperature), by
#| passing beta=0. We also introduce here how to create figure legends."

prior = nested.set_beta(0)
axes = prior.plot_2d(['ns','tau'], label='prior')
nested.plot_2d(axes=axes, label='posterior')
handles, labels = axes['ns']['tau'].get_legend_handles_labels()
fig = axes.loc['ns','tau'].figure
leg = fig.legend(handles, labels)
fig.tight_layout()

#| We can also set up an interactive plot, which allows us to replay a nested
#| sampling run after the fact.

nested.gui()

#| There are also tools for converting to alternative formats, in case you have
#| pipelines in other plotters:

from anesthetic.convert import to_getdist
getdist_samples = to_getdist(nested)
