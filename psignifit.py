# -*- coding: utf-8 -*-
"""
main function for fitting psychometric functions
function result=psignifit(data,options)
This function is the user interface for fitting psychometric functions to data.
    
pass your data in the n x 3 matrix of the form:
[x-value, number correct, number of trials]

options should be a 1x1 struct in which you set the options for your fit.
You can find a full overview over the options in demo002
    
The result of this function is a struct, which contains all information the 
program produced for your fit. You can pass this as whole to all further 
processing function provided with psignifit. Especially to the plot functions.
You can find an explanation for all fields of the result in demo006
    
To get an introduction to basic useage start with demo001
"""
import numpy as np
import warnings

def psignifit(data, options):
   
    #--------------------------------------------------------------------------
    #input parsing
    #--------------------------------------------------------------------------
    # data
    data = np.array(data)
                # percent correct in data
    if all(data[:,1] <= 1 & data[:,1] >= 0) and any(data[:,1] > 0 & data[:,1] < 1):
        
        data[:,1] = round(map( lambda x, y: x*y, data[:,2],data[:,1])) # we try to convert into our notation
        
    # options
        
    if ~(options in locals()): 
        options = lambda:0

    if ~(hasattr(options, 'sigmoidName')):
        options.sigmoidName = 'norm'
    
    if ~(hasattr(options, 'expType')):
        options.expType = 'YesNo'

    if ~(hasattr(options, 'estimateType')):
        options.estimateType = 'MAP'

    if ~(hasattr(options, 'confP')):
        options.confP = [.95, .9, .68]
        
    if ~(hasattr(options, 'instantPlot')):
        options.instantPlot = 0
        
    if ~(hasattr(options, 'setBordersType')):
        options.setBordersType = 0
        
    if ~(hasattr(options, 'maxBorderValue')):
        options.maxBorderValue = .00001
        
    if ~(hasattr(options, 'moveBorders')):
        options.moveBorders = 1
        
    if ~(hasattr(options, 'dynamicGrid')):
        options.dynamicGrid = 0
        
    if ~(hasattr(options, 'widthalpha')):
        options.widthalpha = .05
        
    if ~(hasattr(options, 'threshPC')):
        options.threshPC = .5

    if ~(hasattr(options, 'CImethod')):
        options.CImethod = 'percentiles'

    if ~(hasattr(options, 'gridSetType')):
        options.gridSetType = 'cumDist'
        
    if ~(hasattr(options, 'fixedPars')):
        a = np.empty((5,1))
        a[:] = np.NaN
        options.fixedPars = a
        
    if ~(hasattr(options, 'nblocks')):
        options.nblocks = 25
    
    if ~(hasattr(options, 'useGPU')):
        options.useGPU = 0
    
    if ~(hasattr(options, 'poolMaxGap')):
        options.poolMaxGap = np.inf
    
    if ~(hasattr(options, 'poolMaxLength')):
        options.poolMaxLength = np.inf
    
    if ~(hasattr(options, 'poolxTol')):
        options.poolxTol = 0
    
    if ~(hasattr(options, 'betaPrior')):
        options.betaPrior = 10
    
    if ~(hasattr(options, 'verbose')):
        options.verbose = 0
        
    if ~(hasattr(options, 'stimulusRange')):
        options.stimulusRange = 0
        
    if ~(hasattr(options, 'fastOptim')):
        options.fastOptim = False
        

    if options.expType in ['2AFC', '3AFC', '4AFC']:            
        options.expN = int(float(options.expType[0]))
        options.expType = 'nAFC'

    if options.expType == 'nAFC' and ~hasattr(options,'expN'):
        raise ValueError('For nAFC experiments please also pass the number of alternatives (options.expN)')
    
    if options.expType == 'YesNo':
        if ~(hasattr(options,'stepN')):
            options.stepN = [40,40,20,20,20]
        if ~(hasattr(options, 'mbStepN')):
            options.mbStepN = [25,20, 10,10,15]
    elif options.expType == 'nAFC' or options.expType == 'equalAsymptote':
        if ~(hasattr(options,'stepN')):
            options.stepN = [40,40,20,1,20]
        if ~(hasattr(options, 'mbStepN')):
            options.mbStepN = [30,30,10,1,20]
    else:
        raise ValueError('You specified an illegal experiment type')
    
    assert(max(data[:,0]) > min(data[:,0]), 
    'Your data does not have variance on the x-axis! This makes fitting impossible')
                 
    # check GPU options
    '''TODO!'''
                     
    '''
    log space sigmoids
    we fit these functions with a log transformed physical axis
    This is because it makes the paramterization easier and also the priors
    fit our expectations better then.
    The flag is needed for the setting of the parameter bounds in setBorders
    '''
    
    if options.sigmoidName in ['Weibull','logn','weibull']:
            options.logspace = 1
            assert(min(data[:,0]) > 0, 'The sigmoid you specified is not defined for negative data points!')
    else:
        options.logspace = 0
        
    # add priors
    if options.threshPC != .5 and not(hasattr(options, 'priors')):
        warnings.warn('psignifit:TresholdPCchanged\n You changed the percent correct corresponding to the threshold\n')    
    
    if ~hasattr(options, 'priors'):
        options.priors = getStandardPriors(data, options)
    elif:
        #TODO!
        
    
    
        
                 
    result = 0
    return[result]
