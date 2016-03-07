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
from utils import my_betapdf, my_norminv

def prior1(x, xspread, stimRange):
    
    r = (x >= (stimRange[0]-.5*xspread))*(x<=stimRange[0])*(.5+.5*np.cos(2*np.pi*(stimRange[0]-x)/xspread)) 
    + (x>stimRange[0])*(x<stimRange[1]) + (x>=stimRange[1])*(x<=stimRange[1]+.5*xspread)*(.5+.5*np.cos(2*np.pi*(x-stimRange[1])/xspread))
    return r

def prior2(x, Cfactor, wmin, wmax):
    
    r = ((x*Cfactor)>=wmin)*((x*Cfactor)<=2*wmin)*(.5-.5*np.cos(np.pi*((x*Cfactor)-wmin)/wmin))
    + ((x*Cfactor)>2*wmin)*((x*Cfactor)<wmax)
    + ((x*Cfactor)>=wmax)*((x*Cfactor)<=3*wmax)*(.5+.5*np.cos(np.pi/2*(((x*Cfactor)-wmax)/wmax)))

    return r
def getStandardPriors(data, options):
    priors = []    
    
    """ treat logspace sigmoids """
    if options.logspace:
        data[:,0] = np.log(data[:,0])
        
    ''' of range was not given take it from data '''
    if np.ravel(options.stimulusRange) <= 1:
        options.stimulusRange = np.array([np.min(data[:,0]), np.max(data[:,0])])
        stimRangeSet = False
    else:
        stimRangeSet = True
        if options.logspace:
            options.stimulusRange = np.log(options.stimulusRange)
    
    """ threshold """
    xspread = options.stimulusRange[1]-options.stimulusRange[0]
    ''' we assume the threshold is in the range of the data, for larger or
        smaller values we tapre down to 0 with a raised cosine across half the
        dataspread '''

    priors.append(lambda x: prior1(x,xspread,options.stimulusRange))
    
    """width"""
    # minimum = minimal difference of two stimulus levels
    if len(np.unique(data[:,0])) >1 and not(stimRangeSet):
        widthmin = np.min(np.diff(np.sort(np.unique(data[:,0]))))
    else:
        widthmin =100*np.spacing(options.stimulusRange[1])
    
    widthmax = xspread
    ''' We use the same prior as we previously used... e.g. we use the factor by
        which they differ for the cumulative normal function'''
    Cfactor = (my_norminv(.95,0,1) - my_norminv(.05,0,1))/( my_norminv(1-options.widthalpha,0,1) - my_norminv(options.widthalpha,0,1))
    
    priors.append(lambda x: prior2(x,Cfactor, widthmin, widthmax))
    
    """ asymptotes 
    set asymptote prior to the 1, 10 beta prior, which corresponds to the
    knowledge obtained from 9 correct trials at infinite stimulus level
    """
    
    priors.append(lambda x: my_betapdf(x,1,10))
    priors.append(lambda x: my_betapdf(x,1,10))
    
    """ sigma """
    be = options.betaPrior
    priors.append(lambda x: my_betapdf(x,1,be))
    
    return priors
    
    
    def __call__(self):
        import sys
        
        return getStandardPriors(sys.argv[1], sys.argv[2])
if __name__ == "__main__":
    import sys
    getStandardPriors(sys.argv[1], sys.argv[2])
