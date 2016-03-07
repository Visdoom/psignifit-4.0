# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:17:07 2016

 normalization of given priors
 function Priors=normalizePriors(options)
 This function normalizes the priors from the given options dict, to
 obtain normalized priors.
 This normalization makes later computations for the Bayesfactor and
 plotting of the prior easier.

 This should be run with the original borders to obtain the correct
 normalization
 
@author: root
"""

import numpy as np

def normalizePriors(options):
    
    priors = []
    for idx in range(0,len(options.priors)):
        if options.borders[idx,1] > options.borders[idx,0]:
            #choose xValues for calculation of the integral
            x = np.linspace(options.borders[idx,0], options.borders[idx,1], 1000)
            # evaluate unnormalized prior
            y = options.priors[idx](x)
            w = np.convolve(np.diff(x), np.array([.5,.5]))
            integral = sum(y[:]*w[:])
            priors[idx] = lambda x: options.priors[idx] / integral
        else:
            priors[idx] = lambda x: 1
    
    return priors

if __name__ == "__main__":
    import sys
    normalizePriors(sys.argv[1])
