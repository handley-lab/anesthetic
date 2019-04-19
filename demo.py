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

from anesthetic import MCMCSamples
mcmc_root = 'plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing'
mcmc = MCMCSamples.read(mcmc_root)

#| We have plotting tools for 1D plots ...

fig, axes = mcmc.plot_1d('omegabh2') ;

#| ... multiple 1D plots ...

fig, axes = mcmc.plot_1d(['omegabh2','omegach2','H0','tau','logA','ns']);
fig.tight_layout()

#| ... triangle plots ...

mcmc.plot_2d(['omegabh2','omegach2','H0'], types=['kde']);

#| ... triangle plots (with the equivalent scatter plot filling up the left hand side) ...

mcmc.plot_2d(['omegabh2','omegach2','H0']);

#| ... and rectangle plots.

mcmc.plot_2d([['omegabh2','omegach2','H0'], ['logA', 'ns']]);

#| Rectangle plots are pretty flexible with what they can do:

mcmc.plot_2d([['omegabh2','omegach2','H0'], ['H0','omegach2']]);

#| ## Defining new parameters
#|
#| You can see that samples are stored as a pandas array

print(mcmc[:6])

#| We can define new parameters with relative ease.
#| For example, the default cosmoMC setup does not include omegab, only omegabh2:

'omegab' in mcmc

#| However, this is pretty trivial to recompute:

h = mcmc['H0']/100
mcmc['omegab'] = mcmc['omegabh2']/h**2
mcmc.tex['omegab'] = '$\Omega_b$'
mcmc.plot_1d('omegab');

#| ## Nested sampling plotting
#| Anethestic really comes to the fore for nested sampling. We can do all of
#| the above, and more with the power that NS chains provide

from anesthetic import NestedSamples
nested_root = 'plikHM_TTTEEE_lowl_lowE_lensing_NS/NS_plikHM_TTTEEE_lowl_lowE_lensing'
nested = NestedSamples.read(nested_root)

#| We can infer the evidence, KL divergence and Bayesian model dimensionality:

ns_output = nested.ns_output(1000)
print(ns_output[:6])

#| This is a set of MCMC samples that may be plotted as usual:

from anesthetic import make_1d_axes
fig, axes = make_1d_axes(ns_output.params, ncols=3)
ns_output.plot_1d(axes);

#| We can also inspect the correlation between these inferences:

ns_output.plot_2d(['logZ','D']);

#| Here is a comparison of the base and NS output

h = nested['H0']/100
nested['omegab'] = nested['omegabh2']/h**2
nested.tex['omegab'] = '$\Omega_b$'

fig, axes = mcmc.plot_2d(['sigma8','omegab'])
nested.plot_2d(axes=axes);

#| With nested samples, we can plot the prior (or any temperature), by
#| passing beta=0

from anesthetic import get_legend_proxy

fig, axes = nested.plot_2d(['ns','tau'], beta=0)
nested.plot_2d(axes=axes)
proxy = get_legend_proxy(fig)
fig.legend(proxy, ['prior', 'posterior'])

#| We can also set up an interactive plot, which allows us to replay a nested
#| sampling run after the fact.

nested.gui()
