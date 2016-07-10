# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:34:08 2016



@author: Ole
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import marginalize
from utils import my_norminv 

def plotPsych(result,
              dataColor      = [0, 105./255, 170./255],
              plotData       = True,
              lineColor      = [0, 0, 0],
              lineWidth      = 2,
              xLabel         = 'Stimulus Level',
              yLabel         = 'Proportion Correct',
              labelSize      = 15,
              fontSize       = 10,
              fontName       = 'Helvetica',
              tufteAxis      = False,
              plotAsymptote  = True,
              plotThresh     = True,
              aspectRatio    = False,
              extrapolLength = .2,
              CIthresh       = False,
              dataSize       = 0,
              axisHandle     = None):
    """
    This function produces a plot of the fitted psychometric function with 
    the data.
    """
    # TODO: plotting options additionally as struct/dict ??
    
    fit = result['Fit']
    data = result['data']
    options = result['options']
    
    if axisHandle == None: axisHandle = plt.gca()
    try:
        plt.axes(axisHandle)
    except TypeError:
        raise ValueError('Invalid axes handle provided to plot in.')
    
    if np.isnan(fit[3]): fit[3] = fit[2]
    if data.size == 0: return
    if dataSize == 0: dataSize = 10000. / np.sum(data[:,2])
    
    if 'nAFC' in options['expType']:
        ymin = 1. / options['expN']
        ymin = min([ymin, min(data[:,1] / data[:,2])])
    else:
        ymin = 0
    
    
    # PLOT DATA
    holdState = plt.ishold()
    if not holdState: plt.cla()
    plt.hold(True)
    xData = data[:,0]
    if plotData:
        yData = data[:,1] / data[:,2]
        markerSize = np.sqrt(dataSize/2 * data[:,2])
        for i in range(len(xData)):
            plt.plot(xData[i], yData[i], '.', ms=markerSize[i], c=dataColor, clip_on=False)
    
    # PLOT FITTED FUNCTION
    if options['logspace']:
        xMin = np.log(min(xData))
        xMax = np.log(max(xData))
        xLength = xMax - xMin
        x       = np.exp(np.linspace(xMin, xMax, num=1000))
        xLow    = np.exp(np.linspace(xMin - extrapolLength*xLength, xMin, num=100))
        xHigh   = np.exp(np.linspace(xMax, xMax + extrapolLength*xLength, num=100))
        axisHandle.set_xscale('log')
    else:
        xMin = min(xData)
        xMax = max(xData)
        xLength = xMax - xMin
        x       = np.linspace(xMin, xMax, num=1000)
        xLow    = np.linspace(xMin - extrapolLength*xLength, xMin, num=100)
        xHigh   = np.linspace(xMax, xMax + extrapolLength*xLength, num=100)
    
    fitValuesLow  = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xLow,  fit[0], fit[1]) + fit[3]
    fitValuesHigh = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xHigh, fit[0], fit[1]) + fit[3]
    fitValues     = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x,     fit[0], fit[1]) + fit[3]
    
    plt.plot(x,     fitValues,           c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xLow,  fitValuesLow,  '--', c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xHigh, fitValuesHigh, '--', c=lineColor, lw=lineWidth, clip_on=False)
    
    # PLOT PARAMETER ILLUSTRATIONS
    # THRESHOLD
    if plotThresh:
        if options['logspace']:
            x = [np.exp(fit[0]), np.exp(fit[0])]
        else:
            x = [fit[0], fit[0]]
        y = [ymin, fit[3] + (1 - fit[2] - fit[3]) * options['threshPC']]
        plt.plot(x, y, '-', c=lineColor)
    # ASYMPTOTES
    if plotAsymptote:
        plt.plot([min(xLow), max(xHigh)], [1-fit[2], 1-fit[2]], ':', c=lineColor, clip_on=False)
        plt.plot([min(xLow), max(xHigh)], [fit[3], fit[3]],     ':', c=lineColor, clip_on=False)
    # CI-THRESHOLD
    if CIthresh:
        CIs = result['confIntervals']
        y = np.array([fit[3] + .5*(1 - fit[2] - fit[3]) for i in range(2)])
        plt.plot(CIs[0,:,0],               y,               c=lineColor)
        plt.plot([CIs[0,0,0], CIs[0,0,0]], y + [-.01, .01], c=lineColor)
        plt.plot([CIs[0,1,0], CIs[0,1,0]], y + [-.01, .01], c=lineColor)
    
    #AXIS SETTINGS
    plt.axis('tight')
    plt.tick_params(labelsize=fontSize)
    plt.xlabel(xLabel, fontname=fontName, fontsize=labelSize)
    plt.ylabel(yLabel, fontname=fontName, fontsize=labelSize)
    if aspectRatio: axisHandle.set_aspect(2/(1 + np.sqrt(5)))
#    if tufteAxis:
# TODO: tufteaxis
#    else:
    plt.ylim([ymin, 1])
    # tried to mimic box('off') in matlab, as box('off') in python works differently
    plt.tick_params(direction='out',right='off',top='off')
    for side in ['top','right']: axisHandle.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    plt.hold(holdState)
    plt.show() 
    return axisHandle

def plotsModelfit(result):
    """
    Plots some standard plots, meant to help you judge whether there are
    systematic deviations from the model. We dropped the statistical tests
    here though.
    
    The left plot shows the psychometric function with the data. 
    
    The central plot shows the Deviance residuals against the stimulus level. 
    Systematic deviations from 0 here would indicate that the measured data
    shows a different shape than the fitted one.
    
    The right plot shows the Deviance residuals against "time", e.g. against
    the order of the passed blocks. A trend in this plot would indicate
    learning/ changes in performance over time. 
    
    These are the same plots as presented in psignifit 2 for this purpose.
    """
    
    fit = result['Fit']
    data = result['data']
    options = result['options']
    
    minStim = min(data[:,0])
    maxStim = max(data[:,0])
    stimRange = [1.1*minStim - .1*maxStim, 1.1*maxStim - .1*minStim]
    
    plt.figure(figsize=(15,5))

    ax = plt.subplot(1,3,1)    
    # the psychometric function
    x = np.linspace(stimRange[0], stimRange[1], 1000)
    y = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](x, fit[0], fit[1])
    
    plt.plot(x, y, 'k', clip_on=False)
    plt.plot(data[:,0], data[:,1]/data[:,2], '.k', ms=10, clip_on=False)
    
    plt.xlim(stimRange)
    if options['expType'] == 'nAFC':
        plt.ylim([min(1./options['expN'], min(data[:,1]/data[:,2])), 1])
    else:
        plt.ylim([0,1])
    plt.xlabel('Stimulus Level',  fontsize=14)
    plt.ylabel('Percent Correct', fontsize=14)
    plt.title('Psychometric Function', fontsize=20)
    plt.tick_params(right='off',top='off')
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))   
    
    ax = plt.subplot(1,3,2)
    # stimulus level vs deviance
    stdModel = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](data[:,0],fit[0],fit[1])
    deviance = data[:,1]/data[:,2] - stdModel
    stdModel = np.sqrt(stdModel * (1-stdModel))
    deviance = deviance / stdModel
    xValues = np.linspace(minStim, maxStim, 1000)
    
    plt.plot(data[:,0], deviance, 'k.', ms=10, clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,1)
    plt.plot(xValues, np.polyval(linefit,xValues),'k-', clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,2)
    plt.plot(xValues, np.polyval(linefit,xValues),'k--', clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,3)
    plt.plot(xValues, np.polyval(linefit,xValues),'k:', clip_on=False)

    plt.xlabel('Stimulus Level',  fontsize=14)
    plt.ylabel('Deviance', fontsize=14)
    plt.title('Shape Check', fontsize=20)
    plt.tick_params(right='off',top='off')
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    ax = plt.subplot(1,3,3)
    # block number vs deviance
    blockN = range(len(deviance))
    xValues = np.linspace(min(blockN), max(blockN), 1000)
    plt.plot(blockN, deviance, 'k.', ms=10, clip_on=False)
    linefit = np.polyfit(blockN,deviance,1)
    plt.plot(xValues, np.polyval(linefit,xValues),'k-', clip_on=False)
    linefit = np.polyfit(blockN,deviance,2)
    plt.plot(xValues, np.polyval(linefit,xValues),'k--', clip_on=False)
    linefit = np.polyfit(blockN,deviance,3)
    plt.plot(xValues, np.polyval(linefit,xValues),'k:', clip_on=False)
    
    plt.xlabel('Block #',  fontsize=14)
    plt.ylabel('Deviance', fontsize=14)
    plt.title('Time Dependence?', fontsize=20)
    plt.tick_params(right='off',top='off')
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    plt.tight_layout()


def plotMarginal(result,
                 dim        = 0,
                 lineColor  = [0, 105/255, 170/255],
                 lineWidth  = 2,
                 xLabel     = '',
                 yLabel     = 'Marginal Density',
                 labelSize  = 15,
                 tufteAxis  = False,
                 prior      = True,
                 priorColor = [.7, .7, .7],
                 CIpatch    = True,
                 plotPE     = True,
                 axisHandle = None):
    """
    Plots the marginal for a single dimension.
    result       should be a result struct from the main psignifit routine
    dim          is the parameter to plot:
                   1=threshold, 2=width, 3=lambda, 4=gamma, 5=sigma
    """
    from utils import strToDim
    if isinstance(dim,str): dim = strToDim(dim)
    #TODO: check result['marginals'] type. assumed np.array(dtype=object)
    if len(result['marginals'][dim]) <= 1:
        raise ValueError('The parameter you wanted to plot was fixed in the analysis!')
    if axisHandle == None: axisHandle = plt.gca()
    try:
        plt.axes(axisHandle)
    except TypeError:
        raise ValueError('Invalid axes handle provided to plot in.')
    if not xLabel:
        if   dim == 0: xLabel = 'Threshold'
        elif dim == 1: xLabel = 'Width'
        elif dim == 2: xLabel = '\lambda'
        elif dim == 3: xLabel = '\gamma'
        elif dim == 4: xLabel = '\sigma'
    
    x        = result['marginalsX'][dim]
    marginal = result['marginals'][dim]
    CI       = result['confIntervals'][dim,:,0]
    Fit      = result['Fit'][dim]
    
    holdState = plt.ishold()
    if not holdState: plt.cla()
    plt.hold(True)
    
    # patch for confidence region
    if CIpatch:
        xCI = np.array([CI[0], CI[1], CI[1], CI[0]])
        xCI = np.insert(xCI, 1, x[np.logical_and(x>=CI[0], x<=CI[1])])
        yCI = np.array([np.interp(CI[0], x, marginal), np.interp(CI[1], x, marginal), 0, 0])
        yCI = np.insert(yCI, 1, marginal[np.logical_and(x>=CI[0], x<=CI[1])])
        from matplotlib.patches import Polygon as patch
        color = .5*np.array(lineColor) + .5* np.array([1,1,1])
        axisHandle.add_patch(patch(np.array([xCI,yCI]).T, fc=color, ec=color))
    
    # plot prior
    if prior:
        xprior = np.linspace(min(x), max(x), 1000)
        plt.plot(xprior, result['options']['priors'][dim](xprior), '--', c=priorColor, clip_on=False)
    
    # posterior
    plt.plot(x, marginal, lw=lineWidth, c=lineColor, clip_on=False)
    # point estimate
    if plotPE:
        plt.plot([Fit,Fit], [0, np.interp(Fit, x, marginal)], 'k', clip_on=False)
    
    #if tufteAxis:
        #TODO: tufteAxis
    #else:
    plt.xlim([min(x), max(x)])
    plt.ylim([0, 1.1*max(marginal)])
    
    plt.xlabel(xLabel, fontsize=labelSize, visible=True)
    # if tufteAxis
    plt.ylabel(yLabel, fontsize=labelSize, visible=True)
    # if tufteAxis
    # else:
    plt.tick_params(direction='out', right='off', top='off')
    for side in ['top','right']: axisHandle.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci', scilimits=(-2,4))
    
    plt.hold(holdState)
    return axisHandle
    

def getColorMap():
    """
       This function returns the standard University of Tuebingen Colormap. 
    """    
    midBlue = np.array([165, 30, 55])/255
    lightBlue = np.array([210, 150, 0])/255
    steps = 200
    
    #m1 = np.array([np.linspace(midBlue[i], lightBlue[i], steps) for i in range(0,3)]).transpose()
    #m2 = np.array([np.linspace(lightBlue[i], 1, steps) for i in range(0,3)]).transpose()
    #m = np.append(m1,m2,0)
    MAP = mcolors.LinearSegmentedColormap.from_list('Tuebingen', \
                    [midBlue, lightBlue, [1,1,1]],N = steps, gamma = 1.0) 
    
    return MAP
    
def plotBayes(result, cmap = getColorMap()):

    plt.clf()
    plt.rc('text', usetex=True)
    plt.set_cmap(cmap)
    
    if result['expType'] == 'equalAsymptote':
        result['X1D'][3] = 0

    for ix in range(0,4):
        for jx in range(ix+1,5):
            
            plt.subplot(4,4,4*(ix-1)+jx-1)
            #marginalize
            marg = np.squeeze(marginalize(result,[ix,jx]))
            e = [result['X1D'][jx][0], result['X1D'][jx][-1], \
                 result['X1D'][ix][0], result['X1D'][ix][-1] ]
            if marg.ndim == 1:
                marg = np.reshape(marg, -1, 1)
                if len(result['X1D'][i]) != 1:
                    plt.imshow(marg, extend = e)    
                else:
                    plt.imshow(marg.transpose(), extend = e)
            else:
                plt.imshow(marg, extend = e)
            
            # axis labels
            if ix == 0:
                plt.ylabel('threshold')
            elif ix == 1:
                plt.ylabel('width')
            elif ix == 2:
                plt.ylabel(r'\lambda')
            elif ix == 3:
                plt.ylabel(r'\gamma')
            
            if jx == 0:
                plt.xlabel('threshold')
            elif jx == 1:
                plt.xlabel('width')
            elif jx == 2:
                plt.xlabel(r'\lambda')
            elif jx == 3:
                plt.xlabel(r'\gamma')
            elif jx == 4:
                plt.xlabel(r'\eta')
                
            #TODO there is a one plot function I don't understand
                
    plt.show()
    
def plotPrior(result):
    
    """
    This function creates the plot illustrating the priors on the different 
    parameters
    """

    # plotting parameter
    lineWidth = 2
    lineColor = np.array([0,105,170])/255
    markerSize = 30
    
    data = result['data']

    if np.size(result['options']['stimulusRange'] <= 1):
        result['options']['stimulusRange'] = np.array([min(data[:,0]), max(data[:,0])])
        stimRangeSet = False
    else:
        stimRangeSet = True
        
    stimRange = result['options']['stimulusRange']
    r = stimRange[1] - stimRange[0]
    
    # get borders for width
    # minimum = minimal difference of two stimulus levels
    
    if len(np.unique(data[:,0])) > 1 and not(stimRangeSet):
        widthmin = min(np.diff(np.sort(np.unique(data[:,0]))))
    else:
        widthmin = 100*np.spacing(stimRange[1])
    # maximum = spread of the data

    # We use the same prior as we previously used... e.g. we use the factor by
    # which they differ for the cumulative normal function
    Cfactor = (my_norminv(.95,0,1) - my_norminv(.05,0,1))/          \
            (my_norminv(1-result['options']['widthalpha'], 0,1) -   \
             my_norminv(result['options']['widthalpha'], 0,1))
    widthmax = r
    
    steps = 10000
    theta = np.empty(5)
    for itheta in range(0,5):
        if itheta == 0:
            x = np.linspace(stimRange[0]-.5*r, stimRange[1]+.5*r, steps)
        elif itheta == 1:
            x = np.linspace(min(result['X1D'][itheta]), max(result['X1D'][1],),steps)
        elif itheta == 2:
            x = np.linspace(0,.5,steps)
        elif itheta == 3:
            x = np.linspace(0,.5,steps)
        elif itheta == 4:                
            x = np.linspace(0,1,steps)
        
        y = result['options']['priors'][itheta](x)
        theta[itheta] = sum(x*y)/sum(y)
        
    if result['options']['expType'] == 'equalAsymptote':
        theta[3] = theta[2]
    if result['options']['expType'] == 'nAFC':
        theta[3] = 1/result['options']['expN']
        
    # get limits for the psychometric function plots
    xLimit = [stimRange[0] - .5*r , stimRange[1] +.5*r]
    
    """ threshold """
    
    xthresh = np.linspace(xLimit[0], xLimit[1], steps )
    ythresh = result['options']['priors'][0](xthresh)
    wthresh = np.convolve(np.diff(xthresh), .5*np.array([1,1])) #TODO is that the right function or rather the one from scipy.signal
    cthresh = np.cumsum(ythresh*wthresh)
    
    plt.subplot(2,3,1)
    plt.plot(xthresh,ythresh, lw = lineWidth, c= lineColor)
    plt.hold(True)
    plt.xlim(xLimit)
    plt.title('Threshold', fontsize = 18)
    plt.ylabel('Density',  fontsize = 18)
    
    plt.subplot(2,3,4)    
    plt.plot(data[:,0], np.zeros(data[:,0].shape), 'k.', ls = None, ms = markerSize*.75 )
    plt.hold(True)
    plt.ylabel('Percent Correct', fontsize = 18)
    plt.xlim(xLimit)
    
    for idot in range(0,5):
        if idot == 0:
            xcurrent = theta[0]
            color = 'k'
        elif idot == 1:
            xcurrent = min(xthresh)
            color = [1,200/255,0]
        elif idot == 2:
            tix = next(ix for ix in cthresh if ix >= .25)
            xcurrent = xthresh[tix]
            color = 'r'
        elif idot == 3:
            tix = next(ix for ix in cthresh if ix >= .75)
            xcurrent = xthresh[tix]
            color = 'b'
        elif idot == 4:
            xcurrent = max(xthresh)
            color = 'g'
        y = 100*(theta[3]+(1-theta[2])-theta[3])*result['options']['sigmoidHandle'](x,xcurrent, theta[1])
        
        plt.subplot(2,3,4)
        plt.plot(x,y, '-', lw=lineWidth,c=color )
        plt.subplot(2,3,1)
        plt.plot(xcurrent, result['options']['priors'][0](xcurrent), '.',c=color, ls = None, ms = markerSize)
    
    """ width"""
    xwidth = np.linspace(widthmin, 3/Cfactor*widthmax, steps)
    ywidth = result['options']['priors'][1](xwidth)
    wwidth = np.convolve(np.diff(xwidth), .5*np.array([1,1]))
    cwidth = np.cumsum(ywidth*wwidth)

    plt.subplot(2,3,2)
    plt.plot(xwidth,ywidth,lw=lineWidth,c=lineColor)
    plt.hold(True)
    plt.xlim([widthmin,3/Cfactor*widthmax])
    plt.title('Width',fontsize=18)

    plt.subplot(2,3,5)
    plt.plot(data[:,0],0,'k.',ls = None,ms =markerSize*.75)
    plt.hold(True)
    plt.xlim(xLimit)
    plt.xlabel('Stimulus Level',fontsize=18)

    x = np.linspace(xLimit[0],xLimit[1],steps)
    for idot in range(0,5):
        if idot == 0:
            xcurrent = theta[1]
            color = 'k'
        elif idot == 1:
            xcurrent = min(xwidth)
            color = [1,200/255,0]
        elif idot == 2:
            wix = next(i for i in cwidth if i >= .25)
            xcurrent = xwidth[wix]
            color = 'r'
        elif idot == 3:
            wix = next(i for i in cwidth if i >= .75)
            xcurrent = xwidth[wix]
            color = 'b'
        elif idot ==4:
            xcurrent = max(xwidth)
            color = 'g'
    
    y = 100*(theta[3]+ (1-theta[2] -theta[3])* result['options']['sigmoidHandle'](x,theta[0],xcurrent))
    plt.subplot(2,3,5)
    plt.plot(x,y,'-',lw = lineWidth, c= color)
    plt.subplot(2,3,2)
    plt.plot(xcurrent,result['options']['priors'][1](xcurrent),'.',c = color,ls =None,ms=markerSize)

    """ lapse """

    xlapse = np.linspace(0,.5,steps)
    ylapse = result['options']['priors'][2](xlapse)
    wlapse = np.convolve(np.diff(xlapse),.5*np.array([1,1]))
    clapse = np.cumsum(ylapse*wlapse)
    plt.subplot(2,3,3)
    plt.plot(xlapse,ylapse,lw=lineWidth,c=lineColor)
    plt.hold(True)
    plt.xlim([0,.5])
    plt.title('\lambda',fontsize=18)

    plt.subplot(2,3,6)
    plt.plot(data[:,0],0,'k.',ls=None,ms=markerSize*.75)
    plt.hold(True)
    plt.xlim(xLimit)


    x = np.linspace(xLimit[0],xLimit[1],steps)
    for idot in range(0,5):
        if idot == 0:
            xcurrent = theta[2]
            color = 'k'
        elif idot == 1:
            xcurrent = 0
            color = [1,200/255,0]
        elif idot == 2:
            lix = next(i for i in clapse if i >= .25)
            xcurrent = xlapse[lix]
            color = 'r'
        elif idot == 3:
            lix = next(i for i in clapse if i >= .75)
            xcurrent = xlapse[lix]
            color = 'b'
        elif idot ==4:
            xcurrent = .5
            color = 'g'
    y = 100*(theta[3]+ (1-xcurrent-theta[3])*result['options']['sigmoidHandle'](x,theta[0],theta[1]))
    plt.subplot(2,3,6)
    plt.plot(x,y,'-',lw=lineWidth,c=color)
    plt.subplot(2,3,3)
    plt.plot(xcurrent,result['options']['priors'][2](xcurrent),'.',c=color,ls=None,ms=markerSize)


    a_handle = plt.gca
    a_handle.set_position([200,300,1000,600])
    fig, ax = plt.subplots()
    
    for item in [fig, ax]:
        item.patch.set_visible(False)

def plot2D(result,par1,par2, 
           colorMap = getColorMap(), 
            labelSize = 15,
            fontSize = 10,
            h = None):
    """ 
    This function constructs a 2 dimensional marginal plot of the posterior
    density. This is the same plot as it is displayed in plotBayes in an
    unmodifyable way.

    The result struct is passed as result.
    par1 and par2 should code the two parameters to plot:
        0 = threshold
        1 = width
        2 = lambda
        3 = gamma
        4 = eta
        
    Further plotting options may be passed.
    """
    def strToDim(string):
        """ Finds the number corresponding to a dim/parameter given as a string. """

        s = string.lower()
        if s in ['threshold','thresh','m','t','alpha']:
            dim = 0
            label = 'Treshold'
        elif s in  ['width','w','beta']:
            dim = 1
            label = 'Width'
        elif s in ['lapse','lambda','lapserate','lapse rate','lapse-rate','upper asymptote','l']:
            dim = 2
            label = '\lambda'
        elif s in ['gamma','guess','guessrate','guess rate','guess-rate','lower asymptote','g']:
            dim = 3
            label = '\gamma'
        elif s in ['sigma','std','s','eta','e']:
            dim = 4
            label = '\eta'
        
        return (dim, label)


    # convert strings to dimension number
    if not(par1.isdigit()):
        par1,label1 = strToDim(par1)
    if not(par2.isdigit()):
        par2,label2 = strToDim(par2)

    assert (isnumeric(par1) & isnumeric(par2) & par1 != par2), 'par1 and par2 must be different numbers to code for the parameters to plot'
    assert (par1 in range(0,5) & par2 in range(0,5)) , 'par1 and par2 must be natural numbers up to 4 for the five parameters'

    if h == None:
        h = plt.gca

    plt.axes(h)

    plt.set_cmap(colorMap)
    
    marg = np.squeeze(marginalize(result, [par1, par2]))
    
    if par1 > par2 :
        marg = marg.T


    if 1 in marg.shape:
        if len(result['X1D'][par1])==1:
            plotMarginal(result,par2)
        else:
            plotMarginal(result,par2)
    else:
        e = [result['X1D'][par2],result['X1D'][par1]] # TODO check
        plt.imshow(marg, extend = e)
        plt.ylabel(label1,fontsize = labelSize)
        plt.xlabel(label2,fontsize = labelSize)
        
        set(gca,'TickDir','out')
        plt.box('off')

    
    
if __name__ == "__main__":
    result = {}
    
    result['Fit'] = np.array([.004651, .004658, 1.7125E-7, .5, 1.0632E-4])
    
    options = {}
    options['expType'] = 'nAFC'
    options['expN'] = 2
    options['logspace'] = False
    options['threshPC'] = .5
    from utils import my_norminv, my_normcdf
    alpha = .05
    C = my_norminv(1-alpha,0,1)-my_norminv(alpha,0,1)
    options['sigmoidHandle'] = lambda X,m,width: my_normcdf(X, (m-my_norminv(.5,0,width/C)), width/C)    
    
    tmp1 = np.array([10,15,20,25,30,35,40,45,50,60,70,80,100], dtype=float)/10000
    tmp2 = np.array([45,50,44,44,52,53,62,64,76,79,88,90,90], dtype=float)
    tmp3 = np.array([90 for i in range(len(tmp1))], dtype=float)
    data = np.array([tmp1,tmp2,tmp3]).T
    result['data'] = data
    
    options['stimulusRange'] = 0
    options['widthalpha'] = .05
    options['betaPrior'] = 10
    options['priors'] = [lambda x: [74.074074196287796 for i in range(len(x))]]
    result['options'] = options
    
    CIs = np.zeros((5,2,3))
    CIs[:,:,0] = [[.0043,.0050],[.0035,.0060],[.0002,.0219],[.5,.5],[.0013,.1196]]
    CIs[:,:,1] = [[.0043,.0049],[.0037,.0058],[.0003,.0181],[.5,.5],[.0026,.1016]]
    CIs[:,:,2] = [[.0045,.0048],[.0041,.0053],[.0011,.0112],[.5,.5],[.0083,.0691]]
    result['confIntervals'] = CIs
    
    m1 = np.array(
    [.0082,.0136,.0229,.0394,.0693,.1252,.2334,.4513,.9106,1.93,4.3147,10.1299,24.5262,
     59.3546,138.382,300.3194,590.1429,1.0289E3,1.5691E3,2.0739E3,2.3629E3,2.3158E3,
     1.9536E3,1.4237E3,902.2289,502.3969,249.541,112.9197,47.8892,19.7137,8.1762,3.5234,
     1.6037,.7722,.3908,.206,.1124,.063,.0362,.0212])
    marg = np.empty((5,),dtype=object)
    marg[0] = m1
    result['marginals'] = marg
    
    m1x = np.array(
    [0.003327586206897,
    0.003391246684350,
    0.003454907161804,
    0.003518567639257,
    0.003582228116711,
    0.003645888594164,
    0.003709549071618,
    0.003773209549072,
    0.003836870026525,
    0.003900530503979,
    0.003964190981432,
    0.004027851458886,
    0.004091511936340,
    0.004155172413793,
    0.004218832891247,
    0.004282493368700,
    0.004346153846154,
    0.004409814323607,
    0.004473474801061,
    0.004537135278515,
    0.004600795755968,
    0.004664456233422,
    0.004728116710875,
    0.004791777188329,
    0.004855437665782,
    0.004919098143236,
    0.004982758620690,
    0.005046419098143,
    0.005110079575597,
    0.005173740053050,
    0.005237400530504,
    0.005301061007958,
    0.005364721485411,
    0.005428381962865,
    0.005492042440318,
    0.005555702917772,
    0.005619363395225,
    0.005683023872679,
    0.005746684350133,
    0.005810344827586])
    marg = np.empty((5,),dtype=object)
    marg[0] = m1x
    result['marginalsX'] = marg
    
    #plotPsych(result,CIthresh=True)
    plotsModelfit(result)
    #plotMarginal(result)