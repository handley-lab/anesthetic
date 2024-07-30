import dnest4.classic as dn4
dn4.postprocess(cut=0.0)

# Plots from the blog post
import matplotlib.pyplot as plt
import numpy as np

posterior_sample = dn4.my_loadtxt("posterior_sample.txt")
plt.plot(posterior_sample[:,0], posterior_sample[:,1],
         "k.", markersize=1, alpha=0.2)
plt.show()

plt.imshow(np.corrcoef(posterior_sample.T), cmap="coolwarm",
           vmin=-1.0, vmax=1.0, interpolation="nearest")
plt.show()

