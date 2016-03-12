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
    
To get an introduction to basic usage start with demo001
"""
import numpy as np
import datetime as dt
import warnings

import priors as p
import borders as b
from poolData import poolData 
from getSigmoidHandle import getSigmoidHandle
from psignifitCore import psignifitCore

def psignifit(data, options):
   
    #--------------------------------------------------------------------------
    #input parsing
    #--------------------------------------------------------------------------
    # data
    data = np.array(data)
                # percent correct in data
    if all( np.logical_and(data[:,1] <= 1, data[:,1] >= 0)) and any(np.logical_and(data[:,1] > 0, data[:,1] < 1)):
        
        data[:,1] = round(map( lambda x, y: x*y, data[:,2],data[:,1])) # we try to convert into our notation
        
    # options
        
    if not('options' in locals()): 
        options = lambda:0

    if not(hasattr(options, 'sigmoidName')):
        options.sigmoidName = 'norm'
    
    if not(hasattr(options, 'expType')):
        options.expType = 'YesNo'

    if not(hasattr(options, 'estimateType')):
        options.estimateType = 'MAP'

    if not(hasattr(options, 'confP')):
        options.confP = [.95, .9, .68]
        
    if not(hasattr(options, 'instantPlot')):
        options.instantPlot = 0
        
    if not(hasattr(options, 'setBordersType')):
        options.setBordersType = 0
        
    if not(hasattr(options, 'maxBorderValue')):
        options.maxBorderValue = .00001
        
    if not(hasattr(options, 'moveBorders')):
        options.moveBorders = 1
        
    if not(hasattr(options, 'dynamicGrid')):
        options.dynamicGrid = 0
        
    if not(hasattr(options, 'widthalpha')):
        options.widthalpha = .05
        
    if not(hasattr(options, 'threshPC')):
        options.threshPC = .5

    if not(hasattr(options, 'CImethod')):
        options.CImethod = 'percentiles'

    if not(hasattr(options, 'gridSetType')):
        options.gridSetType = 'cumDist'
        
    if not(hasattr(options, 'fixedPars')):
        a = np.empty((5,1))
        a[:] = np.NaN
        options.fixedPars = a
        
    if not(hasattr(options, 'nblocks')):
        options.nblocks = 25
    
    if not(hasattr(options, 'useGPU')):
        options.useGPU = 0
    
    if not(hasattr(options, 'poolMaxGap')):
        options.poolMaxGap = np.inf
    
    if not(hasattr(options, 'poolMaxLength')):
        options.poolMaxLength = np.inf
    
    if not(hasattr(options, 'poolxTol')):
        options.poolxTol = 0
    
    if not(hasattr(options, 'betaPrior')):
        options.betaPrior = 10
    
    if not(hasattr(options, 'verbose')):
        options.verbose = 0
        
    if not(hasattr(options, 'stimulusRange')):
        options.stimulusRange = 0
        
    if not(hasattr(options, 'fastOptim')):
        options.fastOptim = False
        

    if options.expType in ['2AFC', '3AFC', '4AFC']:            
        options.expN = int(float(options.expType[0]))
        options.expType = 'nAFC'

    if options.expType == 'nAFC' and not(hasattr(options,'expN')):
        raise ValueError('For nAFC experiments please also pass the number of alternatives (options.expN)')
    
    if options.expType == 'YesNo':
        if not(hasattr(options,'stepN')):
            options.stepN = [40,40,20,20,20]
        if not(hasattr(options, 'mbStepN')):
            options.mbStepN = [25,20, 10,10,15]
    elif options.expType == 'nAFC' or options.expType == 'equalAsymptote':
        if not(hasattr(options,'stepN')):
            options.stepN = [40,40,20,1,20]
        if not(hasattr(options, 'mbStepN')):
            options.mbStepN = [30,30,10,1,20]
    else:
        raise ValueError('You specified an illegal experiment type')
    
    assert((max(data[:,0]) - min(data[:,0]) > 0),   
           'Your data does not have variance on the x-axis! This makes fitting impossible')
                 
                     
    '''
    log space sigmoids
    we fit these functions with a log transformed physical axis
    This is because it makes the paramterization easier and also the priors
    fit our expectations better then.
    The flag is needed for the setting of the parameter bounds in setBorders
    '''
    
    if options.sigmoidName in ['Weibull','logn','weibull']:
            options.logspace = 1
            assert min(data[:,0]) > 0, 'The sigmoid you specified is not defined for negative data points!'
    else:
        options.logspace = 0
        
    # add priors
    if options.threshPC != .5 and not(hasattr(options, 'priors')):
        warnings.warn('psignifit:TresholdPCchanged\n'\
            'You changed the percent correct corresponding to the threshold\n')    
    
    if not(hasattr(options, 'priors')):
        options.priors = p.getStandardPriors(data, options)
    else:
        
        priors = p.getStandardPriors(data, options)
        
        for ipar in range(5):
            if not(hasattr(options.priors[ipar], '__call__')):
                options.priors[ipar] = priors[ipar]
                
        p.checkPriors(data, options)
    if options.dynamicGrid and not(hasattr(options, 'GridSetEval')):
        options.GridSetEval = 10000
    if options.dynamicGrid and not(hasattr(options, 'UniformWeight')):
        options.UniformWeight = 1

    '''
    initialize
    '''        
    
    #warning if many blocks were measured
    if (len(np.unique(data[:,0])) >= 25) and (np.ravel(options.stimulusRange).size == 1):
        warnings.warn('psignifit:probablyAdaptive\n'\
            'The data you supplied contained >= 25 stimulus levels.\n'\
            'Did you sample adaptively?\n'\
            'If so please specify a range which contains the whole psychometric function in options.stimulusRange.\n'\
            'This will allow psignifit to choose an appropriate prior.\n'\
            'For now we use the standard heuristic, assuming that the psychometric function is covered by the stimulus levels,\n'\
            'which is frequently invalid for adaptive procedures!')
    
    if all(data[:,2] <= 5) and (np.ravel(options.stimulusRange).size == 1):
        warnings.warn('psignifit:probablyAdaptive\n'\
            'All provided data blocks contain <= 5 trials \n'\
            'Did you sample adaptively?\n'\
            'If so please specify a range which contains the whole psychometric function in options.stimulusRange.\n'\
            'This will allow psignifit to choose an appropriate prior.\n'\
            'For now we use the standard heuristic, assuming that the psychometric function is covered by the stimulus levels,\n'\
            'which is frequently invalid for adaptive procedures!')
    
    #pool data if necessary: more than options.nblocks blocks or only 1 trial per block
    if np.max(data[:,2]) == 1 or len(data) > options.nblocks:
        warnings.warn('psignifit:pooling\n'\
            'We pooled your data, to avoid problems with n=1 blocks or to save time fitting because you have a lot of blocks\n'\
            'You can force acceptence of your blocks by increasing options.nblocks')
        data = poolData(data, options)
    
    options.nblocks = len(data)
    
    # create function handle of sigmoid
    options.sigmoidHandle = getSigmoidHandle(options)
    
    # borders of integration
    if hasattr(options, 'borders'):
        borders = b.setBorders(data, options)
        options.borders[np.isnan(options.borders)] = borders[np.isnan(options.borders)]
    else:
        options.borders = b.setBorders(data,options)
    
    border_idx = np.where(np.isnan(options.fixedPars) == False);
    
    options.borders[border_idx[0]] = options.fixedPars[border_idx[0]]
    options.borders[border_idx[1]] = options.fixedPars[border_idx[1]]
    #options.borders[np.logical_not(np.isnan(options.fixedPars)),0] = options.fixedPars[np.logical_not(np.isnan(options.fixedPars))]
    #options.borders[np.logical_not(np.isnan(options.fixedPars)),1] = options.fixedPars[np.logical_not(np.isnan(options.fixedPars))]        
            
    # normalize priors to first hoice of borders
    options.priors = p.normalizePriors(options)
    if options.moveBorders:
        options.borders = b.moveBorders(data, options)
    
    ''' core '''
    result = psignifitCore(data,options)
        
    ''' after processing '''
    # check that the marginals go to nearly 0 at the borders of the grid
    if options.verbose > -5:
        #TODO check
        if result.marginals[0][0] * result.marginalsW[0][0] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the threshold is not near 0 at the lower border.\n'\
                'This indicates that smaller Thresholds would be possible.')
        if result.marginals[0][-1] * result.marginalsW[0][-1] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the threshold is not near 0 at the upper border.\n'\
                'This indicates that your data is not sufficient to exclude much higher thresholds.\n'\
                'Refer to the paper or the manual for more info on this topic.')
        if result.marginals[1][0] * result.marginalsW[1][0] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the width is not near 0 at the lower border.\n'\
                'This indicates that your data is not sufficient to exclude much lower widths.\n'\
                'Refer to the paper or the manual for more info on this topic.')
        if result.marginals[1][-1] * result.marginalsW[1][-1] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the width is not near 0 at the lower border.\n'\
                'This indicates that your data is not sufficient to exclude much higher widths.\n'\
                'Refer to the paper or the manual for more info on this topic.')
    
    result.timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    
    #if options.instantPlot:
        #plotPsych(result)
        #plotBayes(result) TODO implement and uncomment
    
       
    
    return result
    
if __name__ == "__main__":
    import sys
    psignifit(sys.argv[1], sys.argv[2])

