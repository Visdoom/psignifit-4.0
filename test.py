# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:21:33 2015

@author: sophie
"""

#from collection import namedtuple
import numpy as np
import scipy.io as importer
import psignifit as ps
from getSigmoidHandle import getSigmoidHandle 
import priors as p
from utils import my_norminv, my_normcdf
#from utils import my_betapdf


data = np.array([
    [0.0010,   45.0000,   90.0000],
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




options = {}

options['sigmoidName'] = 'norm'   # choose a cumulative Gauss as the sigmoid
options['expType']     = 'nAFC' 
options['stepN']       = np.array([40,40,20,1,20])
options['betaPrior']   = 10
options['expN']        = 2
options['fixedPars']   = np.array([[np.nan], [np.nan], [np.nan],[np.nan], [np.nan]])
options['poolxTol']    = 0
options['poolMaxLength'] = np.inf
options['poolMaxGap']  = np.inf
options['estimateType'] = 'MAP'
options['confP']       = np.array([0.95,0.9,0.68]) 
options['instantPlot'] = 0
options['setBordersType'] = 0
options['maxBorderValue'] = 1.000000000000000e-05
options['moveBorders'] = 1    
options['dynamicGrid'] = 0
options['widthalpha']  = 0.05 
options['threshPC']    = 0.5
options['CImethod']    = 'percentiles'
options['gridSetType'] = 'cumDist'
options['nblocks']     = 25
options['verbose']     = 0 
#options['stimulusRange'] = 0
options['fastOptim']   = 0
options['mbStepN']     = np.array([30,40,10,1,20])
options['logspace']    = 0

temp_data= importer.loadmat('variables.mat', struct_as_record=True,matlab_compatible=True)
temp_options = importer.loadmat('options.mat', struct_as_record=False, squeeze_me=True)

alpha = np.array([-0.0035    , -0.00287931, -0.00225862, -0.00163793, -0.00101724,
       -0.00039655,  0.00022414,  0.00084483,  0.00146552,  0.00208621,
        0.0027069 ,  0.00332759,  0.00394828,  0.00456897,  0.00518966,
        0.00581034,  0.00643103,  0.00705172,  0.00767241,  0.0082931 ,
        0.00891379,  0.00953448,  0.01015517,  0.01077586,  0.01139655,
        0.01201724,  0.01263793,  0.01325862,  0.01387931,  0.0145    ])

beta = np.array([[  1.73472348e-16,   6.92307692e-04,   1.38461538e-03,
          2.07692308e-03,   2.76923077e-03,   3.46153846e-03,
          4.15384615e-03,   4.84615385e-03,   5.53846154e-03,
          6.23076923e-03,   6.92307692e-03,   7.61538462e-03,
          8.30769231e-03,   9.00000000e-03,   9.69230769e-03,
          1.03846154e-02,   1.10769231e-02,   1.17692308e-02,
          1.24615385e-02,   1.31538462e-02,   1.38461538e-02,
          1.45384615e-02,   1.52307692e-02,   1.59230769e-02,
          1.66153846e-02,   1.73076923e-02,   1.80000000e-02,
          1.86923077e-02,   1.93846154e-02,   2.00769231e-02,
          2.07692308e-02,   2.14615385e-02,   2.21538462e-02,
          2.28461538e-02,   2.35384615e-02,   2.42307692e-02,
          2.49230769e-02,   2.56153846e-02,   2.63076923e-02,
          2.70000000e-02]])


res = ps.psignifit(data, options)

strarray = np.chararray(4,1)
strarray[0] = 'a'
strarray[1] = 'b'
strarray[2] = 'c'
strarray[3] = 'd'

geez = lambda x,y,width: (x*2 + 3, y+width)

t = 'I just want to change something'
