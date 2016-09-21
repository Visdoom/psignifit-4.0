# -*- coding: utf-8 -*-
"""
DEMO_005 PLOTTING FUNCTIONS

 Here the basic plot functions which come with the toolbox are explained. 
 Most of the functions return the handle of the axis they plotted in 
 to enable you to plot further details and change axis properties after the plot. 

 To have something to plot we use the example data as provided in
 demo_001:

"""

from numpy import array

import matplotlib.pyplot as plt
from psignifit import psignifit

from psigniplot import *

data = array([[0.0010,   45.0000,   90.0000],
                 [0.0015,   50.0000,   90.0000],
                 [0.0020,   44.0000,   90.0000],
                 [0.0025,   44.0000,   90.0000],
                 [0.0030,   52.0000,   90.0000],
                 [0.0035,   53.0000,   90.0000],
                 [0.0040,   62.0000,   90.0000],
                 [0.0045,   64.0000,   90.0000],
                 [0.0050,   76.0000,   90.0000],
                 [0.0060,   79.0000,   90.0000],
                 [0.0070,   88.0000,   90.0000],
                 [0.0080,   90.0000,   90.0000],
                 [0.0100,   90.0000,   90.0000]])
                 




options = dict()
options['expType'] = '2AFC'
options['sigmoidName'] = 'norm'

res = psignifit(data, options)

""" plotPsych """
'''This funciton plots the fitted psychometric function with the measured data. 
 It takes the result dict you want to plot. You can also set plotting options.'''
 
plotOptions = {'dataColor': [0,round(105/255,3),round(170/255,3)],
                   'plotData':    True, 
                   'lineColor': [0,0,0],
                   'lineWidth': 2,
                   'xLabel' : 'Stimulus Level', 
                   'yLabel' : 'PercentCorrect', 
                   'labelSize' : 15, 
                   'fontSize' : 10, 
                   'fontName' : 'Helvetica', 
                   'tufteAxis' : False,
                   'plotPar' : True, 
                   'aspectRatio': False, 
                   'extrapolLength': .2, 
                   'CIthresh': False} 

#plt.figure()
plotPsych(res,plotOptions)

""" plotMarginal """
'''This function plots the marginal posterior density for a single parameter. 
As input it requires a results dictionary, the parameter to plot and optionally 
plotting options and a handle to an axis to plot in. 
(As usual 1 = threshold, 2 = width, 3 = lambda, 4 = gamma, 5 = eta)'''

#plt.figure()
plotMarginal(res)

'''The gray shadow corresponds to the chosen confidence interval and the black 
line shows the point estimate for the plotted parameter. 
The prior is also included in the plot as a gray dashed line.'''

'''You may set the following options again with their
 respective default values assigned to change the behaviour of the plot:'''
plotOptions['dim'] = 0
plotOptions['lineColor'] = [0,round(105/255,3),round(170/255,3)]      # color of the density
plotOptions['lineWidth']      = 2                   # width of the plotline
plotOptions['xLabel']         = '[parameter name] '   # X-Axis label
plotOptions['yLabel']         = 'Marginal Density'  # Y-Axis label
plotOptions['labelSize']      = 15                  # Font size for the label
plotOptions['tufteAxis']      = False               # custom axis drawing enabled
plotOptions['prior']          = True;               # include the prior as a dashed weak line
plotOptions['priorColor']     = [.7,.7,.7]          # color of the prior distibution
plotOptions['CIpatch']        = True                # draw the patch for the confidence interval
plotOptions['plotPE']         = True                # plot the point estimate?


""" plot2D """
''' This plots 2 dimensional posterior marginals. 
As input this function expects the result dict, two numbers for the two parameters 
to plot against each other and optionally a handle h to the axis to plot in 
and plotting options. '''

#plt.figure()
plot2D(res,0,1)

'''As options the following fields in plotOptions can be set: '''

plotOptions['axisHandle']  = plt.gca()    # axes handle to plot in
plotOptions['colorMap']  = getColorMap()         # A colormap for the posterior
plotOptions['labelSize'] = 15                   # FontSize for the labels
plotOptions['fontSize']  = 10                   # FontSize for the ticks
plotOptions['label1']    = '[parameter name]'   # label for the first parameter
plotOptions['label2']    = '[parameter name]'   # label for the second parameter

""" plotBayes """
''' This function is a tool to look at the posterior density of the parameter. 
 It plots a grid of all 2 paramter combinations of marginals. 
 If a parameter is fixed in the analysis you will see a 1 dimensional plot 
 in the overview.'''
#plt.figure()
plotBayes(res)

'''You may provide a few additional plotting options. '''


""" plotPrior """
'''As a tool this function plots the actually used priors of the provided 
result dictionary. '''
#plt.figure()
plotPrior(res)
 
 
 
 
 


