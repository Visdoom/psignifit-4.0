# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:09:42 2015

sets the standard Priors
function priors = getStandardPriors(data,options)
The priors set here are the ones used if the user does supply own priors.
Thus this functions constitutes a way to change the priors permanently
note here that the priors here are not normalized. Psignifit takes care
of the normalization implicitly.

@author = Wichmann Lab
translated by Sophie Laturnus

"""
import numpy as np

def getStandardPriors(data, options):
    
    priors = np.array(5,1)
    
    return(priors)