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
from copy import deepcopy
import scipy
import warnings

import likelihood as l 

from gridSetting import gridSetting
from getWeights import getWeights
from getConfRegion import getConfRegion
from getSeed import getSeed
from marginalize import marginalize
from utils import fill_kwargs

def psignifitCore(data, options):
        
    
    d = len(options['borders'])
    result = {'X1D': [], 'marginals': [], 'marginalsX': [], 'marginalsW': []}
    
    '''Choose grid dynamically from data'''
    if options['dynamicGrid']:
        # get seed from linear regression with logit transform
        Seed = getSeed(data,options)
        
        # further optimize the logliklihood to obtain a good estimate of the MAP
        if options['expType'] == 'YesNo':
            calcSeed = lambda X: -l.logLikelihood(data, options, X[0], X[1], X[2], X[3], X[4])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = Seed)
        elif options['expType'] == 'nAFC':
            calcSeed = lambda X: -l.logLikelihood(data, options, X[0], X[1], X[2], 1/options['expN'], X[3])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = [Seed[0:2], Seed[4]])
            Seed = [Seed[0:2], 1/options['expN'], Seed[3]] #ToDo check whether row or colum vector
        result['X1D'] = gridSetting(data,options, Seed) 
    
    
    else: # for types which do not need a MAP estimate
        if (options['gridSetType'] == 'priorlike' or options['gridSetType'] == 'STD'
            or options['gridSetType'] == 'exp' or options['gridSetType'] == '4power'):
                result['X1D'] = gridSetting(data,options) 
        else: # Use a linear grid
            for idx in range(0,d):
                # If there is an actual Interval
                if options['borders'][idx, 0] < options['borders'][idx,1]: 
                    #result.X1D[id] = linspace(bla)
                    result['X1D'].append(np.linspace(options['borders'][idx,0], options['borders'][idx,1],
                                    num=options['stepN'][idx]))
                # if parameter was fixed
                else:
                    result['X1D'].append(np.array([options['borders'][idx,0]]))
                    
    '''Evaluate likelihood and form it into a posterior'''
    #kwargs = {'alpha': None, 'beta':None , 'lambda': None,'gamma':None , 'varscale':None }
    #fill_kwargs(kwargs,result['X1D'])
    (result['Posterior'], result['logPmax']) = l.likelihood(data, options, result['X1D'])
    result['weight'] = getWeights(result['X1D'])
    integral = np.sum(np.array(result['Posterior'][:])*np.array(result['weight'][:]))
    result['Posterior'] = result['Posterior']/integral
    result['integral'] = integral
    
    '''Compute marginal distributions'''
    
    for idx in range(0,d):
        m, mX, mW = marginalize(result, np.array([idx]))
        result['marginals'].append(m)
        result['marginalsX'].append(mX)
        result['marginalsW'].append(mW) 
        
    '''Find point estimate'''
    if (options['estimateType'] in ['MAP','MLE']):
        # get MLE estimate
    
        #start at most likely grid point
        index = np.where(result['Posterior'] == np.max(result['Posterior'].ravel()))
      
        Fit = np.zeros([d,1])
        for idx in range(0,d):
            Fit[idx] = result['X1D'][idx][index[idx]] 
        
        if options['expType'] == 'YesNo':
            fun = lambda X, f: -l.logLikelihood(data, options, [X[0],X[1],X[2],X[3],X[4]])
            x0 = deepcopy(Fit)
            a = None
            
        elif options['expType'] == 'nAFC':
            #def func(X,f):
            #    return -l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            #fun = func
            fun = lambda X, f:  -l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            x0 = deepcopy(Fit[0:3]) # Fit[3]  is excluded
            x0 = np.append(x0,deepcopy(Fit[4]))
            a = np.array([1/options['expN']])
            
        elif options['expType'] == 'equalAsymptote':
            fun = lambda X, f: -l.logLikelihood(data,options,[X[0], X[1], X[2], f, X[3]])
            x0 = deepcopy(Fit[0:3])
            x0 = np.append(x0,deepcopy(Fit[4]))
            a =  np.array([np.nan])
           
        else:
            raise ValueError('unknown expType')
            
        if options['fastOptim']:           
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), xtol=0, ftol = 0, maxiter = 100, maxfun=100)
            warnings.warn('changed options for optimization')
        else:            
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), disp = False)
          
        if options['expType'] == 'YesNo':
            result['Fit'] = deepcopy(Fit)
        elif options['expType'] == 'nAFC': #TODO is this row or column vectors?
            fit = deepcopy(Fit[0:3])
            fit = np.append(fit, np.array([1/options['expN']]))
            fit = np.append(fit, deepcopy(Fit[3]))
            result['Fit'] = fit.transpose()
        elif options['expType'] =='equalAsymptote':
            fit = deepcopy(Fit[0:3])
            fit = np.append(fit, Fit[2])
            fit = np.append(fit, Fit[3])
            result['Fit'] = fit.transpose()
        else:
            raise ValueError('unknown expType')
    
        par_idx = np.where(np.isnan(options['fixedPars']) == False)
        result['Fit'][par_idx] = options['fixedPars'][par_idx] #TODO check
            
    elif options['estimateType'] == 'mean':
        # get mean estimate
        Fit = np.zeros([d,1])
        for idx in range[0:d]:
            Fit[idx] = np.sum(result['marginals'][idx]*result['marginalsW'][idx]*result['marginalsX'][idx])
        
        result['Fit'] = deepcopy(Fit)
        Fit = np.empty(Fit.shape)
    '''Include input into result'''
    result['options'] = options # no copies here, because they are not changing
    result['data'] = data
    
    '''Compute confidence intervals'''
    if ~options['fastOptim']:
        result['conf_Intervals'] = getConfRegion(result)
        
    return result
        
if __name__ == "__main__":
    import sys
    psignifitCore(sys.argv[1], sys.argv[2])

