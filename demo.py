#| Download some example data from github

import requests
import tarfile

for filename in ["plikHM_TTTEEE_lowl_lowE_lensing.tar.gz","plikHM_TTTEEE_lowl_lowE_lensing_NS.tar.gz"]:
    github_url = "https://github.com/williamjameshandley/cosmo_example/raw/master/"
    url = github_url + filename
    open('data/' + filename, 'wb').write(requests.get(url).content)
    tarfile.open('data/' + filename).extractall()

#| This downloaded the PLA chains for the planck baseline cosmology

import os
os.listdir('data/plikHM_TTTEEE_lowl_lowE_lensing')

#| And the equivalent nested sampling chains

os.listdir('data/plikHM_TTTEEE_lowl_lowE_lensing_NS')

#| Now import anesthetic and load the MCMC samples

from anesthetic.anesthetic import MCMCSamples
mcmc = MCMCSamples.read('plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing')

#| You can see that these are stored as a pandas array

print(mcmc)

#| We have plotting tools for 1D plots ...

mcmc.plot_1d('omegabh2');

#| ... multiple 1D plots ...

mcmc.plot_1d(['omegabh2','omegach2','H0','tau','logA','ns']);

#| ... triangle plots (with the equivalent scatter plot filling up the left hand side) ...

mcmc.plot_2d(['omegabh2','omegach2','H0']);

#| ... and rectangle plots.

mcmc.plot_2d(['omegabh2','omegach2','H0'], ['logA', 'ns']);

#| Rectangle plots are pretty flexible with what they can do:

mcmc.plot_2d(['omegabh2','omegach2','H0'], ['H0','omegach2']);

#| More importantly, since this is a pandas array, we can redefine new parameters with relative ease.
#| For example, the default cosmoMC setup does not include omegab, only omegabh2:

'omegab' in mcmc

#| However, this is pretty trivial to recompute:

h = mcmc['H0']/100
mcmc['omegab'] = mcmc['omegabh2']/h**2
mcmc.tex['omegab'] = '$\Omega_b$'
mcmc.plot_1d('omegab');

#| Anethestic really comes to the fore for nested sampling. We can do all of
#| the above, and more with the power that NS chains provide

from anesthetic.anesthetic import NestedSamples
nested = NestedSamples.read('plikHM_TTTEEE_lowl_lowE_lensing_NS/NS_plikHM_TTTEEE_lowl_lowE_lensing')

#| We can infer the evidence, KL divergence and Bayesian model dimensionality:

ns_output = nested.ns_output(1000)
print(ns_output)

#| This is a set of MCMC samples that may be plotted as usual:

ns_output.plot_1d();

#| We can also inspect the correlation between these inferences:

ns_output.plot_2d('logZ','D');

#| Here is a comparison of the base and NS output

h = nested['H0']/100
nested['omegab'] = nested['omegabh2']/h**2
nested.tex['omegab'] = '$\Omega_b$'

fig, axes = mcmc.plot_2d(['sigma8','omegab'])
nested.plot_2d(['sigma8','omegab'], axes=axes, colorscheme='r');

#| Finally, with nested samples, we can plot the prior (or any temperature), by
#| passing beta=0

fig, axes = nested.plot_2d(['ns','tau'], colorscheme='r', beta=0)
nested.plot_2d(['ns','tau'], axes=axes, colorscheme='b');

