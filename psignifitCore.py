# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:26:00 2015

this is the Core processing of psignifit, call the frontend psignifit!
function result=psignifitCore(data,options)
Data nx3 matrix with values [x, percCorrect, NTrials]

sigmoid should be a handle to a function, which accepts
X,parameters as inputs and gives back a value in [0,1]. ideally
parameters(1) should correspond to the threshold and parameters(2) to
the width (distance containing 95% of the function

@author: root
"""
import numpy as np
import copy
import scipy
import warnings

import likelihood as l 

from gridSetting import gridSetting
from getWeights import getWeights
from getConfRegion import getConfRegion
from getSeed import getSeed
from marginalize import marginalize


def psignifitCore(data, options):
    
    d = len(options.borders)
    result = lambda:0
    
    '''Choose grid dynamically from data'''
    if options.dynamicGrid:
        # get seed from linear regression with logit transform
        Seed = getSeed(data,options)
        
        # further optimize the logliklihood to obtain a good estimate of the MAP
        if options.expType == 'YesNo':
            calcSeed = lambda X: -l.logLikelihood(data, options, X[0], X[1], X[2], X[3], X[4])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = Seed)
        elif options.expType == 'nAFC':
            calcSeed = lambda X: -l.logLikelihood(data, options, X[0], X[1], X[2], 1/options.expN, X[3])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = [Seed[0:2], Seed[4]])
            Seed = [Seed[0:2], 1/options.expN, Seed[3]] #ToDo check whether row or colum vector
        result.X1D = gridSetting(data,options, Seed) 
    
    
    else: # for types which do not need a MAP estimate
        if (options.gridSetType == 'priorlike' or options.gridSetType == 'STD'
            or options.gridSetType == 'exp' or options.gridSetType == '4power'):
                result.X1D = gridSetting(data,options) 
        else: # Use a linear grid
            for idx in range[0:d]:
                # If there is an actual Interval
                if options.borders[idx, 0] < options.borders[idx,1]: 
                    #result.X1D[id] = linspace(bla)
                    result.X1D[idx] = np.linspace(options.borders[idx,1], options.borders[idx,2],
                                    num=options.stepN[idx])
                # if parameter was fixed
                else:
                    result.X1D[idx] = options.borders[idx,0]
                    
    '''Evaluate likelihood and form it into a posterior'''
    
    [result.Posterior, result.logPmax] = l.likelihood(data, options, result.X1D[:])
    result.weight = getWeights(result.X1D)
    integral = np.sum(np.array(result.Posterior[:])*np.array(result.weight[:]))
    result.Posterior = result.Posterior/integral
    result.integral = integral
    
    '''Compute marginal distributions'''
    
    for idx in range[0,d]:
        [result.marginals[idx], result.marginalsX[idx], result.marginalsW[idx]] = marginalize(result, id)
        
    '''Find point estimate'''
    if (options.estimateType == 'MAP' or options.estimateType == 'MLE'):
        # get MLE estimate
    
        #start at most likely grid point
        (_, idx) = max(result.Posterior[:])
        #index = cell(d,1)  #ToDo
        index = np.unravel_index(idx, result.Posterior.shape)
        Fit = np.zeros([d,1])
        for idx in range[0:d]:
            Fit[idx] = result.X1D[idx](index[idx]) #ToDo the round braces?
        
        if options.expType == 'YesNo':
            fun = lambda X: -l.logLikelihood(data, options, X[0], X[1], X[2], X[3], X[4])
            x0 = copy.deepcopy(Fit)
        elif options.expType == 'nAFC':
            fun = lambda X:  -l.logLikelihood(data,options, X[0], X[1], X[2], 1/options.expN, X[3])
            x0 = copy.deepcopy(Fit[0:2])
            x0 = np.append(x0,Fit[4])
            x0 = np.transpose(x0)
        elif options.expType == 'equalAsymptote':
            fun = lambda X: -l.logLikelihood(data,options, X[0], X[1], X[2], np.nan, X[3])
            x0 = copy.deepcopy(Fit[0:2])
            x0 = np.append(x0,Fit[4])
            x0 = np.transpose(x0)
        else:
            raise ValueError('unknown expType')
            
        if options.fastOptim:
            #todo check if dictionary works
            optimiseOptions = {'xtol':0, 'ftol':0, 'maxiter': 100, 'maxfun': 100}
            # or maybe optimiseOptions = (0,0,100,100)
            warnings.warn('changed options for optimization')
        else:
            optimiseOptions = {'disp':False}
            #or maybe optimiseOptions = (_,_,_,_,_,False)
        
        Fit =scipy.optimize.fmin(fun, x0, optimiseOptions) #ToDo check if that works this way         
        
        if options.expType == 'YesNo':
            result.Fit = copy.deepcopy(Fit)
        elif options.expType == 'nAFC': #TODO is this row or column vectors?
            result.Fit = np.transpose([Fit[0:2], 1/options.expN, Fit[3]])
        elif options.expType =='equalAsymptote':
            result.Fit = np.transpose([Fit[0:2], Fit[2], Fit[3]])
        else:
            raise ValueError('unknown expType')
    
    #TODO result.Fit[~np.isnan(options.fixedPars)] = options.fixedPars[~np.isnan(options.fixedPars)]
            
    elif options.estimateType == 'mean':
        # get mean estimate
        Fit = np.zeros([d,1])
        for idx in range[0:d]:
            Fit[idx] = np.sum(np.array(result.marginals[idx])*np.array(result.marginalsW[idx])*np.array(result.marginalsX[idx]))
        
        result.Fit = copy.deepcopy(Fit)
        Fit = np.empty(Fit.shape)
    '''Include input into result'''
    result.options = options # no copies here, because they are not changing
    result.data = data
    
    '''Compute confidence intervals'''
    if ~options.fastOptim:
        result.conf_Intervals = getConfRegion(result)
        
    return result
        
if __name__ == "__main__":
    import sys
    psignifitCore(sys.argv[1], sys.argv[2])

