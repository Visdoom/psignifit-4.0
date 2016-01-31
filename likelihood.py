# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:35:27 2015

calculates the (normalized) likelihood for the data from given parameters
function [p,logPmax] = likelihood(typeHandle,data,alpha,beta,lambda,gamma)
This function computes the likelihood for specific parameter values from
the log-Likelihood
The result is normalized to have maximum=1 because the Likelihoods become
very small and this way stay in the range representable in floats

@author: root
"""
import numpy as np
import logLikelihood

def likelihood(data, options, alpha, beta, l, gamma, varscale):
    
    p = logLikelihood(data, options, alpha, beta, l, gamma, varscale)
        
    '''We never need the actual value of the likelihood. Something proportional
    is enough and this circumvents numerical problems for the likelihood to
    become exactly 0'''
    
    logPmax = np.max(p)    
    
    p = p -np.max(p)
    p = np.exp(p)
    
    return (p,logPmax)
