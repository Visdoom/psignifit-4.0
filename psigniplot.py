# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:34:08 2016



@author: Ole
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    # TODO: need back compatibility??
    
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
        y = [fit[3] + .5*(1 - fit[2] - fit[3]) for i in range(2)]
        plt.plot(CIs[1,:,1],               y,               c=lineColor)
        plt.plot([CIs[1,1,1], CIs[1,1,1]], y + [-.01, .01], c=lineColor)
        plt.plot([CIs[1,2,1], CIs[1,2,1]], y + [-.01, .01], c=lineColor)
    
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
    plt.ticklabel_format(style='sci',scilimits=(0,0))
    
    if not holdState: plt.hold(False)

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
    result['options'] = options    
    
    tmp1 = np.array([10,15,20,25,30,35,40,45,50,60,70,80,100], dtype=float)/10000
    tmp2 = np.array([45,50,44,44,52,53,62,64,76,79,88,90,90], dtype=float)
    tmp3 = np.array([90 for i in range(len(tmp1))], dtype=float)
    data = np.array([tmp1,tmp2,tmp3]).T
    result['data'] = data

    plotPsych(result)