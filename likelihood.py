# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp

def likelihood(data, options, alpha, beta, l, gamma, varscale):
    """
    calculates the (normalized) likelihood for the data from given parameters
    function [p,logPmax] = likelihood(typeHandle,data,alpha,beta,lambda,gamma)
    This function computes the likelihood for specific parameter values from
    the log-Likelihood
    The result is normalized to have maximum=1 because the Likelihoods become
    very small and this way stay in the range representable in floats

    """
    
    p = logLikelihood(data, options, alpha, beta, l, gamma, varscale)
        
    '''We never need the actual value of the likelihood. Something proportional
    is enough and this circumvents numerical problems for the likelihood to
    become exactly 0'''
    
    logPmax = np.max(p)    
    
    p = p -np.max(p)
    p = np.exp(p)
    
    return (p,logPmax)


def logLikelihood(data,options, alpha, beta, lamb, gamma, varscale ):
    """
    Created on Mon Nov 30 22:19:05 2015
    the core function to evaluate the logLikelihood of the data
    function p=logLikelihood(data,options,alpha,beta,lambda,gamma,varscale)
    Calculates the logLikelihood of the given data with given parameter
    values. It is fully vectorized and contains the core calculations of
    psignifit.
    this actually adds the log priors as well. Technically it calculates the
    unnormalized log-posterior
    
    @author: root
    """

    
    sigmoidHandle = options.sigmoidHandle
    
    if (not(alpha in locals()) or not(alpha)): #TODO check how alpha looks if empty
        raise ValueError('not enough input parameters')
    if (not(beta in locals()) or not(beta)):
        raise ValueError('not enough input parameters')
    if (not(lamb in locals()) or not(lamb)):
        lamb = 0
    if (not(gamma in locals()) or not(gamma)):
        gamma = .5
    if (not(varscale in locals()) or not(varscale)):
        varscale = 1
    
    # is the input only one point?
    oneParameter = not(len(alpha) > 1 or len(beta) > 1 or len(lamb) > 1 
                or len(gamma) > 1 or len(varscale) > 1)
    
    if oneParameter:     # in optimization if the parameter supplied is not the fixed value
        if np.isfinite(np.array(options.fixedPars[0])):
            alpha = options.fixedPars[0]
        if np.isfinite(np.array(options.fixedPars[1])):
            beta = options.fixedPars[1]
        if np.isfinite(np.array(options.fixedPars[2])):
            lamb = options.fixedPars[2]
        if np.isfinite(np.array(options.fixedPars[3])):
            gamma = options.fixedPars[3]
        if np.isfinite(np.array(options.fixedPars[4])):
            varscale = options.fixedPars[4]
            
    #issues for automization: limit range for lambda & gamma
            #TODO check!
    lamb[lamb < 0 | lamb > (1-np.max(gamma))] = np.nan
    gamma[gamma < 0 | gamma > (1-np.max(lamb))] = np.nan
    varscale[varscale < 0 | varscale > 1] = np.nan
    
    varscaleOrig = np.reshape(varscale, 1,1,1,1,[]);
    
    useGPU = (options.useGPU and ~oneParameter)
    
    if oneParameter:
        if options.expType == 'equalAsymptote':
            gamma = lamb
        p = 0
        scale = 1-gamma -lamb
        psi = np.array([sigmoidHandle(x,alpha, beta) for x in data[:,0]])
        psi = gamma + scale*psi
        n = np.array(data[:,2])
        k = np.array(data[:,1])
        varscale = varscale**2;
        
        if varscale < 10**-9:
            p = p + k * np.log(psi) + (n-k)*np.log(1-psi)   # binomial model
        else:
            v = 1/varscale - 1
            a = psi*v                                       # alpha for binomial
            b = (1-psi)*v                                   # beta for binomial
            p = p + sp.gammaln(k+a) + sp.gammaln(n-k+b)
            p = p -sp.gammaln(n+v) - sp.gammaln(a) - sp.gammaln(b)
            p = p + sp.gammaln(v)
        p = sum(p)  # add up log likelihood
        if np.isnan(p):
            p = - np.inf
    else:       # for grid evaluation
        
        alpha = np.reshape(alpha, [],1)
        beta = np.reshape(beta, 1, [])
        lamb = np.reshape(lamb, 1,1,[])
        gamma = np.reshape(gamma, 1,1,1,[])
        varscale = np.reshape(varscale,1,1,1,1,[])
        varscale = varscale**2          # go from sd to variance
        vbinom = (varscale < 10**-9)    # for variance is smaller than we assume use the binomial model
        
        v = varscale[~vbinom]
        v = 1/v -1
        v = np.reshape(v, 1,1,1,1, [])
        p = 0                           # posterior
        pbin = 0                        # posterior for binomial work
        n = np.size(data,0)
        levels = np.array(data[:,0])    # needed for GPU work
        
        if useGPU:
            gamma = gamma
            lamb = lamb
            v = v
            data = data
            p = p
            pbin = pbin
            # TODO create GPU array
        
        if options.expType == 'equalAsymptote':
            gamma = lamb
        
        scale = 1-gamma-lamb
        for i in range(0,n):
            if options.verbose > 3: 
                print('\r%d/%d', i,n)
            xi = levels[i]
            psi = sigmoidHandle(xi,alpha,beta) #TODO
            psi = psi*scale + gamma
            if useGPU:
                psi = psi # TODO gpuArray
            ni = np.array(data[i,2])
            ki = np.array(data[i,1])
            
            if ((ni-ki)>0 and ki > 0):
                pbin = pbin + ki * np.log(psi) + (ni-ki)*np.log(1-psi)
                if ~np.empty(v):
                    a = psi * v
                    b = (1-psi)*v
                    p = p + sp.gammaln(ki+a) + sp.gammaln(ni-ki+b)
                    p = p - sp.gammaln(ni+v) - sp.gammaln(a) - sp.gammaln(b)
                    p = p + sp.gammaln(v)
                else:
                    p = np.array([])
            elif ki > 0:    # --> ni-ki == 0
                pbin  = pbin + ki * np.log(psi);
                if ~np.empty(v):                                             
                    a = psi*v
                    p = p + sp.gammaln(ki + a)
                    p = p - sp.gammaln(ni+v)
                    p = p - sp.gammaln(a)
                    p = p + sp.gammaln(v)
                else:
                    p = np.array([])
            
            elif (ni-ki) > 0 :  # --> ki ==0
                pbin = pbin  + (ni-ki)*np.log(1-psi)
                if ~np.empty(v):
                    b = (1-psi)*v
                    p = p + sp.gammaln(ni-ki+b)
                    p = p - sp.gammaln(ni+v) - sp.gammaln(b)
                    p = p + sp.gammaln(v)
                else:
                    p = np.array([])
        
        if options.verbose > 3 :
            print('\n')
        
        p = np.concatenate(5,np.matlib.repmat(pbin, [1,1,1,1,sum(vbinom)]),p)
        if useGPU:
            # TODO gather data from GPU
            p = p
            lamb = lamb
            gamma = gamma
            alpha = alpha
            beta = beta
        
        p[np.isnan(p)] = -np.inf

    if ~np.empty(options.priors):
        if isinstance(options.priors, list):
            if hasattr(options.priors[0], '__call__'):
                p = p + np.log(options.priors[0](alpha))
            if hasattr(options.priors[1], '__call__'):
                p = p + np.log(options.priors[1](beta))
            if hasattr(options.priors[2], '__call__'):
                p = p + np.log(options.priors[2](lamb))
            if hasattr(options.priors[3], '__call__'):
                p = p + np.log(options.priors[3](gamma))
            if hasattr(options.priors[4], '__call__'):
                p = p + np.log(options.priors[4](varscaleOrig))
                
    return p  

        
if __name__ == "__main__":
    import sys
    likelihood(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

    
        
