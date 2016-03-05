# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:16:39 2016

 move parameter-boundaries to save computing power 
function borders=moveBorders(data, options)
 this function evaluates the likelihood on a much sparser, equally spaced
 grid definded by mbStepN and moves the borders in so that that 
 marginals below tol are taken away from the borders.

 this is meant to save computing power by not evaluating the likelihood in
 areas where it is practically 0 everywhere.
 
 """
import numpy as np
import warnings 

def moveBorders(data,options):
    borders = []
    
    tol = options.maxBorderValue
    d = options.borders.shape[0]
    
    MBresult =  lambda: 0
    
    ''' move borders out
    should our borders be to tight, e.g. the distribution does not go to zero
    at the borders we move them out until this is the case. 
    
    TODO it was disabled in MATLAB version. What to do with it?
    '''
    
    ''' move borders inwards '''
    
    for idx in range(0,d):
        if (len(options.mbStepN) >= idx and options.mbStepN[idx] >= 2 
            and options.borders[idx,0] != options.borders[idx,1]) :
            MBresult.X1D[idx] = np.linspace(options.borders[idx,0], options.borders[idx,1], options.mbStepN[idx])
        else:
            if (options.borders[idx,0] != options.borders[idx,1] and options.expType != 'equalAsymptote'):
                warnings.warn('MoveBorders: You set only one evaluation for moving the borders!') 
            MBresult.X1D[idx] = .5*(sum(options.borders[idx])
    
        
    MBresult.weight = getWeights(MBresult.X1D)
    MBresult.Posterior = likelihood(data, options, MBresult.X1D) # TODO check!
    integral = sum(MBresult.Posterior[:] * MBresult.weight[:])
    MBresult.Posterior /= integral

    borders = np.zeros([d,2])    
    
    for idx in range(0,d):
        (L1D,x,w) = marginalize(MBresult, idx)
        x1 = x[np.max(np.where(L1D*w >= tol)[0] - 1, 1)]
        x2 = x[np.min(np.where(L1D*w >= tol)[-1]+1, len(x))]
        
        borders[idx,:] = [x1,x2]
    
    return borders
    