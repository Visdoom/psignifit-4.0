# -*- coding: utf-8 -*-
"""
creates the weights for quadrature / numerical integration
function weight=getWeights(X1D)
this function calculates the weights for integration/the volume of the
cuboids given by the 1 dimensional borders in X1D

"""
from numpy import diff, tile, reshape, ones, array, convolve
from scipy.ndimage.filters import convolve as convn

def getWeights(X1D):
    
    d = len(X1D)
    
    ''' puts the X values in their respective dimensions to use bsxfun for
    evaluation'''
    Xreshape = []
    Xreshape.append(reshape(X1D[0], [], 1))
    if d >= 2:
        Xreshape.append(reshape(X1D[1], 1,[]))
    
    for idx in range (2,d):
        Xreshape.append(reshape(X1D[idx], [ones(1,idx-1), len(X1D[idx])]))
    
    # to find and handle singleton dimensions
    Xlength = array([len(X1D[i]) for i in range(0,d)])
    
    ''' calculate weights '''
    #1) calculate mass/volume for each cuboid
    weight = 1
    for idx in range(0,d):
        if Xlength[idx] > 1:
            weight = weight*diff(Xreshape[idx]) # TODO check
    
    #2) sum to get weight for each point
    if d > 1:
        dims = tile(2,[1,d])
        dims[Xlength == 1] = 1
        d = sum(Xlength > 1)
        weight = (2**(-d))* convn(weight, ones(dims))
    else:
        weight = (2**(-1))*convolve(weight, np.array([[1],[1]])
        
    return weight

if __name__ == "__main__":
    import sys
    getWeights(sys.argv[1])
