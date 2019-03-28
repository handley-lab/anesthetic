%load_ext autoreload
%autoreload 2

from anesthetic.anesthetic import MCMCSamples, NestedSamples
from anesthetic.information_theory import normalise_weights, channel_capacity

samples = MCMCSamples.read('./plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing')

fig, axes = samples.plot_2d(['logA','tau'], colorscheme='b')

numpy.log(samples.w).hist()
w = samples.w

w /= w.sum()
numpy.exp((w*-numpy.log(w)).sum())
samples.w.sum()


samples = NestedSamples.read('./plikHM_TTTEEE_lowl_lowE_lensing_NS/NS_plikHM_TTTEEE_lowl_lowE_lensing')
fig, axes = samples.plot_2d(['logA','tau'], axes=axes, colorscheme='b')

samples = NestedSamples.read('./chains/example')
infer = samples.infer()
infer.plot_2d(['D','d','logZ'])

samples['C'] = samples.B+samples.A
samples.tex['C'] = '$C$'
samples.limits['A']=(-2,2)
samples.limits['B']=(-2,2)

samples.plot_2d(['A','B'])

numpy.repeat([1,2,3],[1,0.5,3])

p = samples.weights
p /= p.sum()
sum(-numpy.log(p)*p)
import matplotlib.pyplot as plt
plt.hist(samples.weights,bins=28)


samples.plot_2d(['A','B'],axes=axes,prior=False,color='b')
samples.plot_2d(['A','B'],color='b')

samples = load_nested_samples('/data/will/data/pablo/runs/chains/planck')
samples.paramnames
h = samples['H0']/100
samples['omegab'] = samples['omegabh2']/h**2
samples.tex['omegab'] = '\Omega_b'
samples['omegac'] = samples['omegach2']/h**2
samples.tex['omegac'] = '\Omega_c'

samples.plot_1d('omegab')

samples.plot_1d(['omegam', 'sigma8', 'theta', 'tau'])
samples.plot_2d( ['omegam', 'omegab', 'omegac'], 'H0')
samples.plot_2d( ['omegam', 'omegab', 'omegac'])



samples_2 = load_nested_samples('/data/will/data/pablo/runs/chains/DES')
samples_3 = load_nested_samples('/data/will/data/pablo/runs/chains/DES_planck')

paramnames = ['omegam', 'sigma8']
fig, axes = samples_2.plot_2d(paramnames, color='b')
samples.plot_2d(paramnames, axes=axes, color='r')
samples_3.plot_2d(paramnames, axes=axes, color='g')

fig, axes = samples_2.plot_2d('omegam', 'sigma8', color='b')
samples.plot_2d('omegam', 'sigma8', axes=axes, color='r')
samples_3.plot_2d('omegam', 'sigma8', axes=axes, color='g')

fig, axes = samples.plot_2d(['omegam', 'omegab', 'omegac'], ['H0','tau'])
for ax in axes.flatten():
    ax.label_outer()

samples.infer()
