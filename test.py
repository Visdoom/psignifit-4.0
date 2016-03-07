# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:21:33 2015

@author: sophie
"""

#from collection import namedtuple
import numpy as np
import scipy.io as importer
import psignifit as ps
options = lambda:0

options.sigmoidName = 'norm'   # choose a cumulative Gauss as the sigmoid
options.expType     = '2AFC' 


temp_data= importer.loadmat('variables.mat', struct_as_record=True,matlab_compatible=True)
temp_options = importer.loadmat('options.mat', struct_as_record=False, squeeze_me=True)

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
    
    

res = ps.psignifit(data, options)

strarray = np.chararray(4,1)
strarray[0] = 'a'
strarray[1] = 'b'
strarray[2] = 'c'
strarray[3] = 'd'

geez = lambda x,y,width: (x*2 + 3, y+width)

t = 'I just want to change something'
