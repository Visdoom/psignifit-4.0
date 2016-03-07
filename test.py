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
from utils import my_betapdf


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




options = lambda:0

options.sigmoidName = 'norm'   # choose a cumulative Gauss as the sigmoid
options.expType     = 'nAFC' 
options.stepN       = np.array([40,40,20,1,20])
options.betaPrior   = 10
options.expN        = 2
options.fixedPars   = np.array([[np.nan], [np.nan], [np.nan],[np.nan], [np.nan]])
options.poolxTol    = 0
options.poolMaxLength = np.inf
options.poolMaxGap  = np.inf
options.estimateType = 'MAP'
options.confP       = np.array([0.95,0.9,0.68]) 
options.instantPlot = 0
options.setBordersType = 0
options.maxBorderValue = 1.000000000000000e-05
options.moveBorders = 1    
options.dynamicGrid = 0
options.widthalpha  = 0.05 
options.CImethod    = 'percentiles'
options.gridSetType = 'cumDist'
options.nblocks     = 25
options.verbose     = 0 
options.stimulusRange = 0
options.fastOptim   = 0
options.mbStepN     = np.array([30,40,10,1,20])
options.logspace    = 0
options.priors      = p.getStandardPriors(data,options)

x = np.linspace(0.001,.9,25)
my_betapdf(x,1,10)
p.checkPriors(data,options)
#options.borders
options.sigmoidHandle = getSigmoidHandle(options)

temp_data= importer.loadmat('variables.mat', struct_as_record=True,matlab_compatible=True)
temp_options = importer.loadmat('options.mat', struct_as_record=False, squeeze_me=True)





res = ps.psignifit(data, options)

strarray = np.chararray(4,1)
strarray[0] = 'a'
strarray[1] = 'b'
strarray[2] = 'c'
strarray[3] = 'd'

geez = lambda x,y,width: (x*2 + 3, y+width)

t = 'I just want to change something'
