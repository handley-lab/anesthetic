from anesthetic import NestedSamples
import numpy
ns = NestedSamples(root='./tests/example_data/pc')
fig, axes = ns.plot_2d(['x0', 'x1', 'x2', 'x3', 'x4'])

sigma0, sigma1 = 0.1, 0.1 
eps = 0.9                     # x0 and x1 parameters
sigma2 = 0.1                  # x2 parameter
a, b, c, d = 2., 4., 1., 2.   # x4 parameters

n = 1000
ls = 'r--'

x = numpy.linspace(-0.4,0.4,n)
p = numpy.exp(-x**2/sigma0**2/2)/numpy.sqrt(2*numpy.pi)/sigma0
axes['x0']['x0'].twin.plot(x, p/p.max(), ls)

x = numpy.linspace(-0.4,0.4,n)
p = numpy.exp(-x**2/sigma1**2/2)/numpy.sqrt(2*numpy.pi)/sigma1
axes['x1']['x1'].twin.plot(x, p/p.max(), ls)

x = numpy.linspace(-0.1,0.6,n)
p = numpy.exp(-x/sigma2)/sigma2 * (x>0)
axes['x2']['x2'].twin.plot(x, p/p.max(), ls)

x = numpy.linspace(-0.1,1.1,n)
p = (x<1) & (x>0)
axes['x3']['x3'].twin.plot(x, p/p.max(), ls)

x = numpy.linspace(a-0.1,b+0.1,n)
p = ((x<b) & (x>a)) * (d*(x-a)/(b-a) + c*(b-x)/(b-a))/((b-a)*(c+d)/2)
axes['x4']['x4'].twin.plot(x, p/p.max(), ls)



x3 = numpy.linspace(0,1,n)
x0 = numpy.ones_like(x3) * sigma0
axes['x0']['x3'].plot(2*x0, x3, ls)
axes['x0']['x3'].plot(x0, x3, ls)
axes['x0']['x3'].plot(-x0, x3, ls)
axes['x0']['x3'].plot(-2*x0, x3, ls)

axes['x1']['x3'].plot(2*x0, x3, ls)
axes['x1']['x3'].plot(x0, x3, ls)
axes['x1']['x3'].plot(-x0, x3, ls)
axes['x1']['x3'].plot(-2*x0, x3, ls)

for p in [0.66, 0.95]:
    axes['x2']['x3'].plot(-numpy.log(1-p)*x0, x3, ls)

from scipy.optimize import root
from scipy.special import erf

for p in [0.66, 0.95]:
    k = root(lambda k: -2*numpy.exp(-k)*numpy.sqrt(k/numpy.pi) + erf(numpy.sqrt(k)) - p, 1).x[0]
    x = numpy.linspace(-numpy.sqrt(2*k), numpy.sqrt(2*k), n)
    y = k - x**2/2
    axes['x0']['x2'].plot(x*sigma0, y*sigma2, ls)
    axes['x1']['x2'].plot(x*sigma1, y*sigma2, ls)

t = numpy.linspace(0, 2*numpy.pi, n)
x0 = sigma1*eps*numpy.cos(t) + sigma0 * numpy.sin(t)
x1 = numpy.sqrt(1-eps**2) * sigma1 * numpy.cos(t)


x0 = sigma0 * numpy.sin(t)
x1 = sigma1 * (numpy.sqrt(1-eps**2) * numpy.cos(t)  + eps*numpy.sin(t))
for p in [0.66, 0.95]:
    r = numpy.sqrt(-2*numpy.log(1-p))
    axes['x0']['x1'].plot(r*x1, r*x0, ls)

x3 = numpy.linspace(0,1,n)
x4 = numpy.ones_like(x3)

for p in [0.66, 0.95]:
    #x = (b*(c+(c**2*(1-p)+d**2*p)**0.5) - a*(d+(c**2*(1-p)+d**2*p)**0.5))/(c-d)
    x = (b*c-a*d + (a-b)*(c**2*(1-p)+d**2*p)**0.5)/(c-d)
    axes['x3']['x4'].plot(x3, x*x4, ls)

