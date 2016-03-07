# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:14:59 2016

 automatically set borders on the parameters based on were you sampled.
function Borders=setBorders(data,options)
 this function  sets borders on the parameter values of a given function
 automaically

 It sets: -the threshold to be within the range of the data +/- 50%
          -the width to half the distance of two datapoints up to 10 times
                   the range of the data
          -the lapse rate to 0 to .5
          -the lower asymptote to 0 to .5 or fix to 1/n for nAFC
          -the varscale to the full range from almost 0 to almost 1


@author: root
"""
import numpy as np
from utils import my_norminv 

def setBorders(data,options):

    # lapse fix to 0 - .5    
    lapseB = np.array([0,.5])
    
    if options.expType == 'nAFC':
        gammaB = np.array([1/options.expN, 1/options.expN])
    elif options.expType == 'YesNo':
        gammaB = np.array([0, .5])
    elif options.expType == 'equalAsymptote':
        gammaB = np.array([np.nan, np.nan])
    
    # varscale from 0 to 1, 1 excluded!
    varscaleB = np.array(0, 1-np.exp(-20))  
    
    if options.logspace:
        data[:,0] = np.log(data[:,0])
    
    # if range was not given take from data
    if options.stimulusRange.size <= 1 :
        options.stimulusRange = np.array([min(data[:,0]), max(data[:,0])])
        stimRangeSet = False
    else:
        stimRangeSet= True
        if options.logspace:
            options.stimulusRange = np.log(options.stimulusRange)
    
    '''
     We then assume it is one of the reparameterized functions with
     alpha=threshold and beta= width
     The threshold is assumed to be within the range of the data +/-
     .5 times it's spread
    '''
    dataspread = np.diff(options.stimulusRange)
    alphaB = np.array([options.stimulusRange[0] - .5*dataspread, options.stimulusRange[1] +.5*dataspread])
    
    ''' the width we assume to be between half the minimal distance of
    two points and 5 times the spread of the data '''
    
    if len(np.unique(data[:,0])) > 1 and not(stimRangeSet):
        widthmin = np.min(np.diff(np.sort(np.unique(data[:,0]))))
    else :
        widthmin = 100*np.spacing(options.stimulusRange[1])
    
    ''' We use the same prior as we previously used... e.g. we use the factor by
    which they differ for the cumulative normal function '''
    
    Cfactor = (my_norminv(.95,0,1) - my_norminv(.05, 0,1))/(my_norminv(1- options.widthalpha, 0,1) - my_norminv(options.widthalpha, 0,1))
    betaB  = np.array([widthmin, 3/Cfactor*dataspread])
    
    borders =[[alphaB], [betaB], [lapseB], [gammaB], [varscaleB]]
    borders = np.array(borders).squeeze()
    
    return borders 

if __name__ == "__main__":
    import sys
    setBorders(sys.argv[1], sys.argv[2])
