%load_ext autoreload
%autoreload 2

from anesthetic.anesthetic import load_nested_samples

# Load the samples
samples = load_nested_samples('./chains/example')
samples['C'] = samples.B/samples.A
samples.tex['C'] = 'C'

fig, axes = samples.plot_2d(['A','B'],prior=True,color='r')
samples.plot_2d(['A','B'],axes=axes,prior=True,color='b')
#samples.plot_2d(['A','B','C'],color='b')

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

fig, axes = samples.plot_2d(['omegam', 'omegab', 'omegac'], ['H0','tau'])
for ax in axes.flatten():
    ax.label_outer()

samples.infer()
