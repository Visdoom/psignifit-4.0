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
from copy import deepcopy
import scipy

import likelihood as l 
import priors as p
import borders as b

from gridSetting import gridSetting
from getWeights import getWeights
from getConfRegion import getConfRegion
from getSeed import getSeed
from marginalize import marginalize
from poolData import poolData 
from getSigmoidHandle import getSigmoidHandle

from psigniplot import plotPsych

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
        options = {}

    if not('sigmoidName' in options.keys()):
        options['sigmoidName'] = 'norm'
    
    if not('expType' in options.keys()):
        options['expType'] = 'YesNo'

    if not('estimateType' in options.keys()):
        options['estimateType'] = 'MAP'

    if not('confP' in options.keys()):
        options['confP'] = [.95, .9, .68]
        
    if not('instantPlot' in options.keys()):
        options['instantPlot'] = 0
        
    if not('setBordersType' in options.keys()):
        options['setBordersType'] = 0
        
    if not('maxBorderValue' in options.keys()):
        options['maxBorderValue'] = .00001
        
    if not('moveBorders' in options.keys()):
        options['moveBorders'] = 1
        
    if not('dynamicGrid' in options.keys()):
        options['dynamicGrid'] = 0
        
    if not('widthalpha' in options.keys()):
        options['widthalpha'] = .05
        
    if not('threshPC' in options.keys()):
        options['threshPC'] = .5

    if not('CImethod' in options.keys()):
        options['CImethod'] = 'percentiles'

    if not('gridSetType' in options.keys()):
        options['gridSetType'] = 'cumDist'
        
    if not( 'fixedPars' in options.keys()):
        a = np.empty((5,1))
        a[:] = np.NaN
        options['fixedPars'] = a
        
    if not('nblocks' in options.keys()):
        options['nblocks'] = 25
    
    if not('useGPU' in options.keys()):
        options['useGPU'] = 0
    
    if not('poolMaxGap' in options.keys()):
        options['poolMaxGap'] = np.inf
    
    if not('poolMaxLength' in options.keys()):
        options['poolMaxLength'] = np.inf
    
    if not('poolxTol' in options.keys()):
        options['poolxTol'] = 0
    
    if not('betaPrior' in options.keys()):
        options['betaPrior'] = 10
    
    if not('verbose' in options.keys()):
        options['verbose'] = 0
        
    if not('stimulusRange' in options.keys()):
        options['stimulusRange'] = 0
        
    if not('fastOptim' in options.keys()):
        options['fastOptim'] = False
    
    if options['expType'] in ['2AFC', '3AFC', '4AFC']:            
        options['expN'] = int(float(options['expType'][0]))
        options['expType'] = 'nAFC'

    if options['expType'] == 'nAFC' and not('expN' in options.keys()):
        raise ValueError('For nAFC experiments please also pass the number of alternatives (options.expN)')
    
    if options['expType'] == 'YesNo':
        if not('stepN' in options.keys()):
            options['stepN'] = [40,40,20,20,20]
        if not('mbStepN' in options.keys()):
            options['mbStepN'] = [25,30, 10,10,15]
    elif options['expType'] == 'nAFC' or options['expType'] == 'equalAsymptote':
        if not('stepN' in options.keys()):
            options['stepN'] = [40,40,20,1,20]
        if not('mbStepN' in options.keys()):
            options['mbStepN'] = [30,40,10,1,20]
    else:
        raise ValueError('You specified an illegal experiment type')
    
    assert((max(data[:,0]) - min(data[:,0]) > 0), \
                'Your data does not have variance on the x-axis! This makes fitting impossible')
                 
                     
    '''
    log space sigmoids
    we fit these functions with a log transformed physical axis
    This is because it makes the paramterization easier and also the priors
    fit our expectations better then.
    The flag is needed for the setting of the parameter bounds in setBorders
    '''
    
    if options['sigmoidName'] in ['Weibull','logn','weibull']:
            options['logspace'] = 1
            assert min(data[:,0]) > 0, 'The sigmoid you specified is not defined for negative data points!'
    else:
        options['logspace'] = 0
        
    #if range was not given take from data
    if len(np.ravel(options['stimulusRange'])) <=1 :
        if options['logspace']:
            options['stimulusRange'] = np.array(np.log([min(data[:,0]),max(data[:,0])]))
        else :
            options['stimulusRange'] = np.array([min(data[:,0]),max(data[:,0])])

        stimRangeSet = False
    else:
        stimRangeSet = True
        if options['logspace']:
            options['stimulusRange'] = np.log(options['stimulusRange'])
    

    if not('widthmin' in options.keys()):
        if len(np.unique(data[:,0])) >1 and not(stimRangeSet):
            if options['logspace']:
                options['widthmin']  = min(np.diff(np.sort(np.unique(np.log(data[:,0])))))
            else:
                options['widthmin']  = min(np.diff(np.sort(np.unique(data[:,0]))))
        else:
            options['widthmin'] = 100*np.spacing(options['stimulusRange'][1])

    # add priors
    if options['threshPC'] != .5 and not(hasattr(options, 'priors')):
        warnings.warn('psignifit:TresholdPCchanged\n'\
            'You changed the percent correct corresponding to the threshold\n')    
    
    if not('priors' in options.keys()):
        options['priors'] = p.getStandardPriors(data, options)
    else:
        
        priors = p.getStandardPriors(data, options)
        
        for ipar in range(5):
            if not(hasattr(options['priors'][ipar], '__call__')):
                options['priors'][ipar] = priors[ipar]
                
        p.checkPriors(data, options)
    if options['dynamicGrid'] and not('GridSetEval' in options.keys()):
        options['GridSetEval'] = 10000
    if options['dynamicGrid'] and not('UniformWeight' in options.keys()):
        options['UniformWeight'] = 1

    '''
    initialize
    '''        
    
    #warning if many blocks were measured
    if (len(np.unique(data[:,0])) >= 25) and (np.ravel(options['stimulusRange']).size == 1):
        warnings.warn('psignifit:probablyAdaptive\n'\
            'The data you supplied contained >= 25 stimulus levels.\n'\
            'Did you sample adaptively?\n'\
            'If so please specify a range which contains the whole psychometric function in options.stimulusRange.\n'\
            'This will allow psignifit to choose an appropriate prior.\n'\
            'For now we use the standard heuristic, assuming that the psychometric function is covered by the stimulus levels,\n'\
            'which is frequently invalid for adaptive procedures!')
    
    if all(data[:,2] <= 5) and (np.ravel(options['stimulusRange']).size == 1):
        warnings.warn('psignifit:probablyAdaptive\n'\
            'All provided data blocks contain <= 5 trials \n'\
            'Did you sample adaptively?\n'\
            'If so please specify a range which contains the whole psychometric function in options.stimulusRange.\n'\
            'This will allow psignifit to choose an appropriate prior.\n'\
            'For now we use the standard heuristic, assuming that the psychometric function is covered by the stimulus levels,\n'\
            'which is frequently invalid for adaptive procedures!')
    
    #pool data if necessary: more than options.nblocks blocks or only 1 trial per block
    if np.max(data[:,2]) == 1 or len(data) > options['nblocks']:
        warnings.warn('psignifit:pooling\n'\
            'We pooled your data, to avoid problems with n=1 blocks or to save time fitting because you have a lot of blocks\n'\
            'You can force acceptance of your blocks by increasing options.nblocks')
        data = poolData(data, options)
    
    
    # create function handle of sigmoid
    options['sigmoidHandle'] = getSigmoidHandle(options)
    
    # borders of integration
    if 'borders' in options.keys():
        borders = b.setBorders(data, options)
        options['borders'][np.isnan(options['borders'])] = borders[np.isnan(options.borders)]
    else:
        options['borders'] = b.setBorders(data,options)
    
    border_idx = np.where(np.isnan(options['fixedPars']) == False);
    
    options['borders'][border_idx[0]] = options['fixedPars'][border_idx[0]]
    options['borders'][border_idx[1]] = options['fixedPars'][border_idx[1]]
            
    # normalize priors to first choice of borders
    options['priors'] = p.normalizePriors(options)
    if options['moveBorders']:
        options['borders'] = b.moveBorders(data, options)
    
    ''' core '''
    result = psignifitCore(data,options)
        
    ''' after processing '''
    # check that the marginals go to nearly 0 at the borders of the grid
    if options['verbose'] > -5:
    
        if result['marginals'][0][0] * result['marginalsW'][0][0] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the threshold is not near 0 at the lower border.\n'\
                'This indicates that smaller Thresholds would be possible.')
        if result['marginals'][0][-1] * result['marginalsW'][0][-1] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the threshold is not near 0 at the upper border.\n'\
                'This indicates that your data is not sufficient to exclude much higher thresholds.\n'\
                'Refer to the paper or the manual for more info on this topic.')
        if result['marginals'][1][0] * result['marginalsW'][1][0] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the width is not near 0 at the lower border.\n'\
                'This indicates that your data is not sufficient to exclude much lower widths.\n'\
                'Refer to the paper or the manual for more info on this topic.')
        if result['marginals'][1][-1] * result['marginalsW'][1][-1] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the width is not near 0 at the lower border.\n'\
                'This indicates that your data is not sufficient to exclude much higher widths.\n'\
                'Refer to the paper or the manual for more info on this topic.')
    
    result['timestamp'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if options['instantPlot']:
        plotPsych(result)
        #plotBayes(result)  #TODO
    
       
    
    return result
    
def psignifitFast(data,options):
    """
    this uses changed settings for the fit to obtain a fast point estimate to
    your data. 
    The mean estimate with these settings is very crude, the MAP estimate is
    better, but takes a bit of time for the optimization (~100 ms)
    """
    
    warnings.warn('You use the speed optimized version of this program. \n' \
    'This is NOT suitable for the final analysis, but meant for online analysis, adaptive methods etc. \n'  \
    'It has not been tested how good the estimates from this method are!')

    options['stepN']     = [20,20,10,10,1]
    options['mbStepN']  = [20,20,10,10,1]
    options['fixedPars'] = [np.NaN,np.NaN,np.NaN,np.NaN,0]
    options['fastOptim'] = True
    
    res = psignifit(data,options)
    
    return res 
    

def psignifitCore(data, options):
    """
    This is the Core processing of psignifit, call the frontend psignifit!
    function result=psignifitCore(data,options)
    Data nx3 matrix with values [x, percCorrect, NTrials]

    sigmoid should be a handle to a function, which accepts
    X,parameters as inputs and gives back a value in [0,1]. ideally
    parameters(1) should correspond to the threshold and parameters(2) to
    the width (distance containing 95% of the function.
    """
    
    d = len(options['borders'])
    result = {'X1D': [], 'marginals': [], 'marginalsX': [], 'marginalsW': []}
    
    '''Choose grid dynamically from data'''
    if options['dynamicGrid']:
        # get seed from linear regression with logit transform
        Seed = getSeed(data,options)
        
        # further optimize the logliklihood to obtain a good estimate of the MAP
        if options['expType'] == 'YesNo':
            calcSeed = lambda X: -l.logLikelihood(data, options, X[0], X[1], X[2], X[3], X[4])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = Seed)
        elif options['expType'] == 'nAFC':
            calcSeed = lambda X: -l.logLikelihood(data, options, X[0], X[1], X[2], 1/options['expN'], X[3])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = [Seed[0:2], Seed[4]])
            Seed = [Seed[0:2], 1/options['expN'], Seed[3]] #ToDo check whether row or colum vector
        result['X1D'] = gridSetting(data,options, Seed) 
    
    
    else: # for types which do not need a MAP estimate
        if (options['gridSetType'] == 'priorlike' or options['gridSetType'] == 'STD'
            or options['gridSetType'] == 'exp' or options['gridSetType'] == '4power'):
                result['X1D'] = gridSetting(data,options) 
        else: # Use a linear grid
            for idx in range(0,d):
                # If there is an actual Interval
                if options['borders'][idx, 0] < options['borders'][idx,1]: 
                    
                    result['X1D'].append(np.linspace(options['borders'][idx,0], options['borders'][idx,1],
                                    num=options['stepN'][idx]))
                # if parameter was fixed
                else:
                    result['X1D'].append(np.array([options['borders'][idx,0]]))
                    
    '''Evaluate likelihood and form it into a posterior'''
    
    (result['Posterior'], result['logPmax']) = l.likelihood(data, options, result['X1D'])
    result['weight'] = getWeights(result['X1D'])
    integral = np.sum(np.array(result['Posterior'][:])*np.array(result['weight'][:]))
    result['Posterior'] = result['Posterior']/integral
    result['integral'] = integral
    
    '''Compute marginal distributions'''
    
    for idx in range(0,d):
        m, mX, mW = marginalize(result, np.array([idx]))
        result['marginals'].append(m)
        result['marginalsX'].append(mX)
        result['marginalsW'].append(mW) 
    
    result['marginals'] = np.squeeze(result['marginals'])
    result['marginalsX'] = np.squeeze(result['marginalsX'])
    result['marginalsW'] = np.squeeze(result['marginalsW'])
        
    '''Find point estimate'''
    if (options['estimateType'] in ['MAP','MLE']):
        # get MLE estimate
    
        #start at most likely grid point
        index = np.where(result['Posterior'] == np.max(result['Posterior'].ravel()))
      
        Fit = np.zeros([d,1])
        for idx in range(0,d):
            Fit[idx] = result['X1D'][idx][index[idx]] 
        
        if options['expType'] == 'YesNo':
            fun = lambda X, f: -l.logLikelihood(data, options, [X[0],X[1],X[2],X[3],X[4]])
            x0 = deepcopy(Fit)
            a = None
            
        elif options['expType'] == 'nAFC':
            #def func(X,f):
            #    return -l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            #fun = func
            fun = lambda X, f:  -l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            x0 = deepcopy(Fit[0:3]) # Fit[3]  is excluded
            x0 = np.append(x0,deepcopy(Fit[4]))
            a = np.array([1/options['expN']])
            
        elif options['expType'] == 'equalAsymptote':
            fun = lambda X, f: -l.logLikelihood(data,options,[X[0], X[1], X[2], f, X[3]])
            x0 = deepcopy(Fit[0:3])
            x0 = np.append(x0,deepcopy(Fit[4]))
            a =  np.array([np.nan])
           
        else:
            raise ValueError('unknown expType')
            
        if options['fastOptim']:           
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), xtol=0, ftol = 0, maxiter = 100, maxfun=100)
            warnings.warn('changed options for optimization')
        else:            
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), disp = True)
          
        if options['expType'] == 'YesNo':
            result['Fit'] = deepcopy(Fit)
        elif options['expType'] == 'nAFC': 
            fit = deepcopy(Fit[0:3])
            fit = np.append(fit, np.array([1/options['expN']]))
            fit = np.append(fit, deepcopy(Fit[3]))
            result['Fit'] = fit
            
        elif options['expType'] =='equalAsymptote':
            fit = deepcopy(Fit[0:3])
            fit = np.append(fit, Fit[2])
            fit = np.append(fit, Fit[3])
            result['Fit'] = fit
        else:
            raise ValueError('unknown expType')
    
        par_idx = np.where(np.isnan(options['fixedPars']) == False)
        for idx in par_idx:
            result['Fit'][idx] = options['fixedPars'][idx]
            
    elif options['estimateType'] == 'mean':
        # get mean estimate
        Fit = np.zeros([d,1])
        for idx in range[0:d]:
            Fit[idx] = np.sum(result['marginals'][idx]*result['marginalsW'][idx]*result['marginalsX'][idx])
        
        result['Fit'] = deepcopy(Fit)
        Fit = np.empty(Fit.shape)
    '''Include input into result'''
    result['options'] = options # no copies here, because they are not changing
    result['data'] = data
    
    '''Compute confidence intervals'''
    if ~options['fastOptim']:
        result['conf_Intervals'] = getConfRegion(result)
        
    return result

if __name__ == "__main__":
    import sys
    psignifit(sys.argv[1], sys.argv[2])

