from anesthetic import NestedSamples
import numpy
ns = NestedSamples(root='./tests/example_data/pc')
fig, axes = ns.plot_2d(['x0', 'x1', 'x2', 'x3', 'x4'])

sigma0, sigma1 = 0.1, 0.1 
eps = 0.9                     # x0 and x1 parameters
sigma2 = 0.1                  # x2 parameter
a, b, m = 2., 4., 0.5         # x4 parameters

n = 1000
ls = 'k--'

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
p = ((x<b) & (x>a)) * (1/(b-a) + m * (x-(b+a)/2.))
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
for p in [0.66, 0.95]:
    x4 = 1/(a-b)/m + (a+b)/2 + (((m*(a-b)**2+2)/(a-b))**2 - 8*m*p)**0.5/2/m
    x4 = x4 * numpy.ones_like(x3)
    axes['x3']['x4'].plot(x3, x4, ls)

axes['x3']['x4'].plot(x3, numpy.ones_like(x3)*b, ls)

axes['x3']['x4'].plot(x3[0] * numpy.ones(n), numpy.linspace(x4[0],b,n), ls)
axes['x3']['x4'].plot(x3[-1] * numpy.ones(n), numpy.linspace(x4[0],b,n), ls)

x4 = numpy.linspace(a,b,n)
for p in [0.66, 0.95]:
    k = 1/(b-a) + m/2 * (b-a - numpy.sqrt(8*p/m))
    x4 = numpy.linspace((a+b)/2+(k/m + 1/(a-b)/m), b,n)
    x2 = sigma2*(-numpy.log(k) + numpy.log(1/(b-a) + m * (x4-(b+a)/2)))
    axes['x2']['x4'].plot(x2, x4, ls)

axes['x2']['x4'].plot(numpy.zeros_like(x4), x4, ls)
axes['x2']['x4'].plot(x2, b*numpy.ones_like(x2), ls)

from scipy.special import erf, erfi
from scipy.optimize import root

for p in [0.66, 0.95]:
    k = root(lambda k: (1/(b-a)**2*(2+(b-a)**2*m)**2*erf(numpy.sqrt(numpy.log((2+(b-a)**2*m)/2/(b-a)/k))) - 4*k**2*erfi(numpy.sqrt(numpy.log((2+(b-a)**2*m)/2/(b-a)/k))))/8/m-p, 0.5).x[0]
    x1 = numpy.sqrt(2*numpy.log((2+(a-b)**2*m)/2/(b-a)/k))
    x1 = numpy.linspace(-x1,x1,n)
    x4 = (a+b)/2 + numpy.exp(x1**2/2)*k/m - 1/(b-a)/m
    x1 *= sigma1
    axes['x1']['x4'].plot(x1, x4, ls)

    axes['x0']['x4'].plot(x1, x4, ls)

axes['x1']['x4'].plot(x1, b*numpy.ones_like(x1), ls)
axes['x0']['x4'].plot(x1, b*numpy.ones_like(x1), ls)
