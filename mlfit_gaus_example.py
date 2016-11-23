import math
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Minimizer():
    def __init__(self, points, npars, ipars, verbosity=0, maxiter=1000):
        self.points = points
        self.verbosity = verbosity
        
        # initialise
        self.minuit = ROOT.TMinuit(npars) 
        self.minuit.SetPrintLevel(verbosity)
        self.minuit.SetMaxIterations(maxiter)
        
        # initialise paramters
        for i, par in enumerate(ipars):
            self.minuit.mnparm(i, par[0], par[1], par[2], par[3], par[4], ROOT.Long(0))
        
        # set FCN
        self.minuit.SetFCN(self.fcn)
            
    def func(self, mean, sigma):
        fcn = -np.log(norm(mean, sigma).pdf(self.points)).sum()        
        if self.verbosity > 0:
            print 'testing mean %.4f sigma %.4f\tfcn %.4f' %(mean, sigma, fcn)
        return fcn
    
    def fcn(self, npars, gin, f, par, iflag):
        f[0] = self.func(par[0], par[1])
    
    def Migrad(self):
        self.results = self.minuit.Migrad()


if __name__ == '__main__':
    # fix random seed
    rng = np.random.RandomState(1986)
    
    # generate multivariate normal
    mvg = rng.normal(5., 1., 1000)
    
    # plot the random points
    plt.hist(mvg)
    # plt.show()
    
    # name, initial value, step, low bound, up bound
    ipars = [
        ('mean' , -10., 0.1, 0,     0),
        ('sigma', 2. , 0.1, 0, 10000),
    ]
    
    minimiser = Minimizer(mvg, 2, ipars, 1)
    
    minimiser.Migrad()
