# -*- coding: utf-8 -*-
"""
 this runs a short test whether the provided priors are functional
 function checkPriors(data,options)
 concretely the priors are evaluated for a 25 values on each dimension and
 a warning is issued for zeros and a error for nan and infs and negative
 values

"""
import numpy as np
import warnings

def checkPriors(data,options):

    if options.logspace :
        data[:,0] = np.log(data[:,0])
    
    """ on threshold 
    values chosen according to standard boarders 
    at the borders it may be 0 -> a little inwards """
    data_min = np.min(data[:,0])
    data_max = np.max(data[:,0])
    dataspread = data_max - data_min
    testValues = np.linspace(data_min - .4*dataspread, data_max + .4*dataspread, 25)
    
    testResult = options.priors[0](testValues)

    testForWarnings(testResult, "the threshold")
    """ on width
    values according to standard priors
    """
    testValues = np.linspace(1.1*np.min(np.diff(np.sort(np.unique(data[:,0])))), 2.9*dataspread, 25)
    testResult = options.priors[1](testValues)
    
    testForWarnings(testResult, "the width")
    
    """ on lambda
    values 0 to .9
    """
    testValues = np.linspace(0.001,.9,25)
    testResult = options.priors[2](testValues)
    
    testForWarnings(testResult, "lambda")
    
    """ on gamma
    values 0 to .9
    """
    testValues = np.linspace(0.0001,.9,25)
    testResult = options.priors[3](testValues)
    
    testForWarnings(testResult, "gamma")
    
    """ on eta
    values 0 to .9
    """
    testValues = np.linspace(0,.9,25)
    testResult = options.priors[4](testValues)
    
    testForWarnings(testResult, "eta")    
   
    
def testForWarnings(testResult, parameter):
    
    assert all(np.isfinite(testResult)), "the prior you provided for %s returns non-finite values" %parameter
    assert all(testResult >= 0), "the prior you provided for %s returns negative values" % parameter

    if any(testResult == 0):
        warnings.warn("the prior you provided for %s returns zeros" % parameter)
    