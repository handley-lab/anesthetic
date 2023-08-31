class TensionCalculator:
    def __init__(self,A,B,AB,nsamples):
        self.nsamples = nsamples
        self.A = A.stats(nsamples)
        self.B = B.stats(nsamples)
        self.AB = AB.stats(nsamples)