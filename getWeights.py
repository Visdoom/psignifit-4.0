# -*- coding: utf-8 -*-
"""
creates the weights for quadrature / numerical integration
function weight=getWeights(X1D)
this function calculates the weights for integration/the volume of the
cuboids given by the 1 dimensional borders in X1D

"""

def getWeights(X1D):
    
    d = len(X1D)
    
    return weight