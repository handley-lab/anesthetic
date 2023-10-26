from anesthetic import read_chains, make_2d_axes
from anesthetic.tension import tension_stats


# Change dir1, dir2 and dir3 to change datasets for comparison
dir1 = 'bao.sdss_dr16'
dir2 = 'planck_2018_plik' #'planck_2018_CamSpec'#'planck_2018_plik_nolens' #'planck_2018_CamSpec_nolens'
dir3 = "+".join(sorted([dir1,dir2])) # Put dir1 and dir2 in alphabetical order

dir_list = ['bao.sdss_dr16','bicep_keck_2018','des_y1.joint','planck_2018_CamSpec','planck_2018_CamSpec_nolens','planck_2018_plik','planck_2018_plik_nolens','sn.pantheon','bicep_keck_2018']


# Read the chains for the 3 datasets to be compared
samples1 = read_chains(f"../ns/klcdm/{dir1}/{dir1}_polychord_raw/{dir1}")
samples2 = read_chains(f"../ns/klcdm/{dir2}/{dir2}_polychord_raw/{dir2}")
samples3 = read_chains(f"../ns/klcdm/{dir3}/{dir3}_polychord_raw/{dir3}")

#%%
# Make a fig and axes, set the parameters to plot
fig, axes = make_2d_axes(['omk', 'H0', 'omegam']) # must match the column index
# Plot datasets on the same ax
samples1.plot_2d(axes, label='BAO')
samples2.plot_2d(axes, label='Planck')
samples3.plot_2d(axes, label='BAO+Planck')
# Set location of legend
axes.iloc[-1,  0].legend(loc='lower center', bbox_to_anchor=(len(axes)/2, len(axes)))


#%%
nsamples = 1000
beta = 1
samples = tension_stats(samples1,samples2,samples3,nsamples,beta)
#%%
fig, ax = plt.subplots(2,2)

#%%
plt.figure()
plt.title('logR')
samples.logR.plot.hist()
plt.show()

plt.figure()
plt.title('logI')
samples.logI.plot.hist()
plt.show()

plt.figure()
plt.title('logS')
samples.logS.plot.hist()
plt.show()

plt.figure()
plt.title('d_G')
samples.d_G.plot.hist()
plt.show()


# %%
