# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:17:05 2016

@author: Ole
"""

import scipy.io as sio
from psignifit import psignifit
import numpy as np

testSet = '2AFC_Gauss'
testsMat = sio.loadmat('SophieTests.mat')['R_' + testSet]

n = len(testsMat['data'])
differences = {'X1D':        np.zeros((n,5), dtype=float),
               'logPmax':    np.zeros((n, ), dtype=float),
               'integral':   np.zeros((n, ), dtype=float),
               'marginals':  np.zeros((n,5), dtype=float),
               'marginalsX': np.zeros((n,5), dtype=float),
               'marginalsW': np.zeros((n,5), dtype=float),
               'Fit':        np.zeros((n,5), dtype=float)}

options = {}
keys = ['sigmoidName', 'expType','expN','estimateType','stepN','mbStepN','confP',
        'CImethod', 'betaPrior', 'useGPU','nblocks','poolMaxGap','poolMaxLength',
        'poolxTol', 'instantPlot', 'borders', 'setBordersType', 'maxBorderValue',
        'moveBorders', 'dynamicGrid', 'GridSetEval', 'UniformWeight', 'widthalpha',
        'logspace']

for i in range(n):
    data = testsMat['data'][i][0]
    
    # parse options struct to proper dict
    tmpOptions = testsMat['options'][i][0][0,0]
    for key in keys:
        try:
            if 'str' in tmpOptions[key].dtype.name:
                options[key] = tmpOptions[key][0]
            else:
                options[key] = tmpOptions[key]
        except:
            print(key + ' does not exist')    
    
    # run fitting
    results = psignifit(data,options)
    
    # calculate differences to matlab results (max difference for the latter ones)
    for param in ['logPmax','integral']:
        differences[param][i] = testsMat[param][i][0][0][0][0] - results[param]
    
    for param in ['X1D','marginals','marginalsX','marginalsW','Fit']:
        tmp = testsMat[param][i][0][0]
        for j in range(5):
            differences[param][i,j] = max(tmp[j] - results[param])