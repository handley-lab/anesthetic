import numpy
from fastkde import fastKDE
import matplotlib.pyplot as plt
import scipy

def kde(d, xmin=None, xmax=None):
    if xmin is not None and xmax is not None:
        d_ = scipy.special.erfinv(2*(d-xmin)/(xmax-xmin)-1) *2/numpy.sqrt(numpy.pi) 
    elif xmin is not None:
        xmed = numpy.median(d)
        d_ = numpy.concatenate((scipy.special.erfinv((d[d<xmed]-xmed)/(xmed-xmin))*2/numpy.sqrt(numpy.pi), (d[d>=xmed]-xmed)/(xmed-xmin)))  
    elif xmax is not None:
        p, x = kde(-d, xmin=-xmax) 
        return p, -x
    else:
        d_ = d

    p, x = fastKDE.pdf(d_)

    if xmin is not None and xmax is not None:
        x_ = xmin+(xmax-xmin)/2*(1+scipy.special.erf(x*numpy.sqrt(numpy.pi)/2))
        i = p>1e-2
        p = p[i] * numpy.exp(x[i]**2*numpy.pi/4)*2/(xmax-xmin)
        x = x_[i]
    elif xmin is not None:
        low = numpy.logical_and(x<0,p>1e-2)
        high = numpy.logical_and(x>=0,p>1e-2)
        x_ = numpy.concatenate((xmed + (xmed-xmin)*scipy.special.erf(x[low]*numpy.sqrt(numpy.pi)/2), xmed+(xmed-xmin)*x[high]))
        p = numpy.concatenate((p[low],p[high]))
        p[:sum(low)] *= numpy.exp(x[low]**2*numpy.pi/4)/(xmed-xmin)
        p[sum(low):] *= 1/(xmed-xmin)
        x = x_
        
    return p, x

d = numpy.random.randn(10000)
d = d[d>0]

plt.hist(d,density=True)
p, x = kde(d)
plt.plot(x, p)
p, x = kde(d,xmin=0)
plt.plot(x, p)



p, x = kde(d,xmax=xmax)
plt.plot(x, p)
p, x = kde(d,xmin=xmin)
plt.plot(x, p)

xmin = 0

d = 1-numpy.random.rand(1000)**(1/5.)
#plt.hist(d)
plt.hist(numpy.log(d),density=True,bins=50)
p, x = fastKDE.pdf(numpy.log(d))
plt.plot(x, p)

def trans(x, xmin=None, xmax=None):
    if xmin is not None and xmax is not None:
        return scipy.special.erfinv((x-xmin)/(xmax-xmin)-1)*2/numpy.sqrt(numpy.pi)



x = numpy.linspace(-1,1,1000)
plt.plot(x, scipy.special.erfinv(x)*2/numpy.sqrt(numpy.pi))
plt.plot(x, x)

xmax=1
xmin=0
fig, axes = plt.subplots(2,3)
for ax in axes.flatten():
    d = numpy.random.rand(10000)
    p, x = kde(d)
    ax.plot(x, p)
    p, x = kde(d,0,1)
    ax.plot(x, p)
    ax.hist(d, density=True)
    ax.plot(x, numpy.logical_and(x>0,x<1))

fig, ax = plt.subplots()

plt.ylim(0,2)

x_.min()
x = scipy.special.expit(x_)
p = p_/x/(1-x)
plt.plot(x, p)
plt.ylim(0,2)


d = numpy.random.rand(1000)
plt.hist(d,density=True)
p, x = fastKDE.pdf(d)
plt.plot(x, p)
p, x = fastKDE.pdf(d, axisExpansionFactor=0.0)
plt.plot(x, p)

d = numpy.random.randn(1000)
plt.hist(d,density=True)
p, x = fastKDE.pdf(d)
plt.plot(x, p)
p, x = fastKDE.pdf(d, axisExpansionFactor=0.0)
plt.plot(x, p)
plt.subplots()
plt
plt.hist(d)
plt.hist(4-d)

xmin = 1
xmax = 3
d = xmin + (xmax-xmin)*numpy.random.rand(100)
xmax-(d-xmin)

d = numpy.random.randn(1000)
plt.hist(d)
xmin = -3
xmax = 4
plt.hist(2*xmax-d)
d
a + -1*(r-a)
2a-r

def kde(d, xmin=None, xmax=None):
    axisExpansionFactor=1.
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        d_ = numpy.concatenate((2*xmin-d[d<xmed],d,2*xmax-d[d>=xmed]))
        axisExpansionFactor=0.
    elif xmin is not None:
        d_ = numpy.concatenate((2*xmin-d,d))
    elif xmax is not None:
        d_ = numpy.concatenate((d,2*xmax-d))
    else:
        d_ = d

    p, x = fastKDE.pdf(d_,axisExpansionFactor=axisExpansionFactor)
    #return x, p

    if xmin is not None and xmax is not None:
        p = 2*p[numpy.logical_and(x>=xmin,x<=xmax)]
        x = x[numpy.logical_and(x>=xmin,x<=xmax)]
    elif xmin is not None:
        p = 2*p[x>=xmin]
        x = x[x>=xmin]
    elif xmax is not None:
        p = 2*p[x<=xmax]
        x = x[x<=xmax]

    return x, p

n = 10000
d = numpy.concatenate((numpy.random.rand(n//2),numpy.random.rand(n//2)**0.5))
#d = numpy.random.rand(n)**0.5
#d = numpy.random.rand(n)
#d = numpy.random.randn(n)
plt.hist(d,density=True,bins=100)
plt.plot(*kde(d,xmin=0,xmax=1))
plt.plot(*kde(d,xmin=0))
plt.plot(*kde(d,xmax=1))

plt.plot(*kde(d))

