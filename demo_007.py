# -*- coding: utf-8 -*-
"""
DEMO_007 POOLING UTILITY
 This demo illustrates the use of psignifits automatic pooling utility.

 For this illustration we use the following, not so handy dataset of a
 Quest run with 400 trials:

"""
from numpy import array, inf
from psignifit import psignifit
from psigniplot import plotPsych

data = array([[1.3262  ,  1.0000  ,  1.0000], 
                [0.8534  ,   1.0000  ,   1.0000], 
                [0.4717  ,   0  ,  1.0000],
                [1.3091  ,  1.0000  ,  1.0000],
                [1.0467  ,  1.0000  ,  1.0000],
                [0.8816  ,  1.0000  ,  1.0000],
                [0.7765  ,       0  ,  1.0000],
                [1.0535  ,       0  ,  1.0000],
                [1.6860  ,  1.0000  ,  1.0000],
                [1.3763  ,  1.0000  ,  1.0000],
                [1.2443  ,  1.0000  ,  1.0000],
                [1.1757  ,  1.0000  ,  1.0000],
                [1.1286  ,  1.0000  ,  1.0000],
                [1.0922  ,  1.0000  ,  1.0000],
                [1.0623  ,  1.0000  ,  1.0000],
                [1.0367  ,       0  ,  1.0000],
                [1.1065  ,  1.0000  ,  1.0000],
                [1.0838  ,  1.0000  ,  1.0000],
                [1.0651  ,  1.0000  ,  1.0000],
                [1.0490  ,  1.0000  ,  1.0000],
                [1.0352  ,       0  ,  1.0000],
                [1.0739  ,  1.0000  ,  1.0000],
                [1.0603  ,       0  ,  1.0000],
                [1.0964  ,  1.0000  ,  1.0000],
                [1.0826  ,  1.0000  ,  1.0000],
                [1.0708  ,  1.0000  ,  1.0000],
                [1.0606  ,  1.0000  ,  1.0000],
                [1.0515  ,  1.0000  ,  1.0000],
                [1.0434  ,       0  ,  1.0000],
                [1.0649  ,  1.0000  ,  1.0000],
                [1.0567  ,  1.0000  ,  1.0000],
                [1.0492  ,  1.0000  ,  1.0000],
                [1.0424  ,  1.0000  ,  1.0000],
                [1.0362  ,  1.0000  ,  1.0000],
                [1.0303  ,  1.0000  ,  1.0000],
                [1.0249  ,  1.0000  ,  1.0000],
                [1.0197  ,  1.0000  ,  1.0000],
                [1.0150  ,       0  ,  1.0000],
                [1.0288  ,  1.0000  ,  1.0000],
                [1.0240  ,  1.0000  ,  1.0000],
                [1.0194  ,  1.0000  ,  1.0000],
                [1.0151  ,  1.0000  ,  1.0000],
                [1.0109  ,       0  ,  1.0000],
                [1.0222  ,  1.0000  ,  1.0000],
                [1.0181  ,  1.0000  ,  1.0000],
                [1.0142  ,  1.0000  ,  1.0000],
                [1.0104  ,  1.0000  ,  1.0000],
                [1.0069  ,       0  ,  1.0000],
                [1.0170  ,       0  ,  1.0000],
                [1.0275  ,       0  ,  1.0000],
                [1.0386  ,       0  ,  1.0000],
                [1.0503  ,  1.0000  ,  1.0000],
                [1.0461  ,  1.0000  ,  1.0000],
                [1.0420  ,       0  ,  1.0000],
                [1.0529  ,  1.0000  ,  1.0000],
                [1.0488  ,       0  ,  1.0000],
                [1.0601  ,  1.0000  ,  1.0000],
                [1.0560  ,       0  ,  1.0000],
                [1.0679  ,  1.0000  ,  1.0000],
                [1.0636  ,  1.0000  ,  1.0000],
                [1.0595  ,  1.0000  ,  1.0000],
                [1.0557  ,  1.0000  ,  1.0000],
                [1.0521  ,       0  ,  1.0000],
                [1.0618  ,       0  ,  1.0000],
                [1.0722  ,       0  ,  1.0000],
                [1.0836  ,  1.0000  ,  1.0000],
                [1.0792  ,  1.0000  ,  1.0000],
                [1.0752  ,  1.0000  ,  1.0000],
                [1.0713  ,       0  ,  1.0000],
                [1.0816  ,  1.0000  ,  1.0000],
                [1.0777  ,  1.0000  ,  1.0000],
                [1.0740  ,  1.0000  ,  1.0000],
                [1.0705  ,  1.0000  ,  1.0000],
                [1.0673  ,  1.0000  ,  1.0000],
                [1.0642  ,  1.0000  ,  1.0000],
                [1.0611  ,       0  ,  1.0000],
                [1.0692  ,  1.0000  ,  1.0000],
                [1.0663  ,  1.0000  ,  1.0000],
                [1.0634  ,  1.0000  ,  1.0000],
                [1.0605  ,       0  ,  1.0000],
                [1.0681  ,  1.0000  ,  1.0000],
                [1.0653  ,  1.0000  ,  1.0000],
                [1.0626  ,       0  ,  1.0000],
                [1.0697  ,  1.0000  ,  1.0000],
                [1.0671  ,  1.0000  ,  1.0000],
                [1.0645  ,  1.0000  ,  1.0000],
                [1.0620  ,  1.0000  ,  1.0000],
                [1.0595  ,  1.0000  ,  1.0000],
                [1.0572  ,  1.0000  ,  1.0000],
                [1.0550  ,  1.0000  ,  1.0000],
                [1.0528  ,  1.0000  ,  1.0000],
                [1.0506  ,      0   , 1.0000],
                [1.0565  ,      0   , 1.0000],
                [1.0627  ,  1.0000  ,  1.0000],
                [1.0604  ,  1.0000  ,  1.0000],
                [1.0583  ,  1.0000  ,  1.0000],
                [1.0562  ,  1.0000 ,   1.0000],
                [1.0542  ,  1.0000 ,   1.0000],
                [1.0522  ,  1.0000 ,   1.0000],
                [1.0502  ,  1.0000 ,   1.0000],
                [1.0484  ,  1.0000 ,   1.0000],
                [1.0467  ,  1.0000 ,   1.0000],
                [1.0450  ,  1.0000 ,   1.0000],
                [1.0432  ,  1.0000 ,   1.0000],
                [1.0415  ,       0 ,   1.0000],
                [1.0461  ,  1.0000 ,   1.0000],
                [1.0444  ,  1.0000 ,   1.0000],
                [1.0428  ,  1.0000 ,   1.0000],
                [1.0411  ,  1.0000 ,   1.0000],
                [1.0395  ,  1.0000 ,   1.0000],
                [1.0380  ,       0 ,   1.0000],
                [1.0424  ,  1.0000 ,   1.0000],
                [1.0408  ,  1.0000 ,   1.0000],
                [1.0392  ,       0 ,   1.0000],
                [1.0436  ,  1.0000 ,   1.0000],
                [1.0420  ,  1.0000 ,   1.0000],
                [1.0404  ,  1.0000 ,   1.0000],
                [1.0390  ,  1.0000 ,   1.0000],
                [1.0376  ,       0 ,   1.0000],
                [1.0416  ,  1.0000 ,   1.0000],
                [1.0401  ,  1.0000 ,   1.0000],
                [1.0387  ,  1.0000 ,   1.0000],
                [1.0374  ,  1.0000 ,   1.0000],
                [1.0361  ,       0 ,   1.0000],
                [1.0398  ,       0 ,   1.0000],
                [1.0438  ,  1.0000 ,   1.0000],
                [1.0424  ,  1.0000 ,   1.0000],
                [1.0410  ,  1.0000 ,   1.0000],
                [1.0396  ,  1.0000 ,   1.0000],
                [1.0383  ,  1.0000 ,   1.0000],
                [1.0371  ,       0 ,   1.0000],
                [1.0406  ,  1.0000 ,   1.0000],
                [1.0393  ,  1.0000 ,   1.0000],
                [1.0381  ,       0 ,   1.0000],
                [1.0417  ,  1.0000 ,   1.0000],
                [1.0403  ,  1.0000 ,   1.0000],
                [1.0391  ,  1.0000 ,   1.0000],
                [1.0379  ,  1.0000 ,   1.0000],
                [1.0367  ,  1.0000 ,   1.0000],
                [1.0356  ,  1.0000 ,   1.0000],
                [1.0344  ,  1.0000 ,   1.0000],
                [1.0332  ,  1.0000 ,   1.0000],
                [1.0321  ,  1.0000 ,   1.0000],
                [1.0309  ,  1.0000 ,   1.0000],
                [1.0297  ,  1.0000 ,   1.0000],
                [1.0287  ,  1.0000 ,   1.0000],
                [1.0277  ,  1.0000 ,   1.0000],
                [1.0267  ,  1.0000 ,   1.0000],
                [1.0256  ,  1.0000 ,   1.0000],
                [1.0246  ,  1.0000 ,   1.0000],
                [1.0236  ,       0 ,   1.0000],
                [1.0263  ,  1.0000 ,   1.0000],
                [1.0253  ,       0 ,   1.0000],
                [1.0281  ,  1.0000 ,   1.0000],
                [1.0271  ,  1.0000 ,   1.0000],
                [1.0262  ,       0 ,   1.0000],
                [1.0289  ,  1.0000 ,   1.0000],
                [1.0280  ,       0 ,   1.0000],
                [1.0308  ,       0 ,   1.0000],
                [1.0337  ,  1.0000 ,   1.0000],
                [1.0326  ,  1.0000 ,   1.0000],
                [1.0316  ,  1.0000 ,   1.0000],
                [1.0305  ,  1.0000 ,   1.0000],
                [1.0295  ,  1.0000 ,   1.0000],
                [1.0286  ,       0 ,   1.0000],
                [1.0313  ,  1.0000 ,   1.0000],
                [1.0303  ,  1.0000 ,   1.0000],
                [1.0293  ,       0 ,   1.0000],
                [1.0321  ,  1.0000 ,   1.0000],
                [1.0311  ,  1.0000 ,   1.0000],
                [1.0300  ,  1.0000 ,   1.0000],
                [1.0291  ,  1.0000 ,   1.0000],
                [1.0282  ,  1.0000 ,   1.0000],
                [1.0274  ,  1.0000 ,   1.0000],
                [1.0265  ,  1.0000 ,   1.0000],
                [1.0256  ,       0 ,   1.0000],
                [1.0281  ,  1.0000 ,   1.0000],
                [1.0272  ,       0 ,   1.0000],
                [1.0297  ,  1.0000 ,   1.0000],
                [1.0288  ,  1.0000 ,   1.0000],
                [1.0280  ,  1.0000 ,   1.0000],
                [1.0271  ,  1.0000 ,   1.0000],
                [1.0263  ,  1.0000 ,   1.0000],
                [1.0255  ,  1.0000 ,   1.0000],
                [1.0246  ,       0 ,   1.0000],
                [1.0268  ,  1.0000 ,   1.0000],
                [1.0260  ,  1.0000 ,   1.0000],
                [1.0252  ,  1.0000 ,   1.0000],
                [1.0244  ,  1.0000 ,   1.0000],
                [1.0235  ,  1.0000 ,   1.0000],
                [1.0227  ,  1.0000 ,   1.0000],
                [1.0219  ,  1.0000 ,   1.0000],
                [1.0211  ,  1.0000 ,   1.0000],
                [1.0202  ,  1.0000 ,   1.0000],
                [1.0194  ,       0 ,   1.0000],
                [1.0217  ,  1.0000 ,   1.0000],
                [1.0209  ,  1.0000 ,   1.0000],
                [1.0200  ,  1.0000 ,   1.0000],
                [1.0193  ,       0 ,   1.0000],
                [1.0215  ,       0 ,   1.0000],
                [1.0236  ,  1.0000 ,   1.0000],
                [1.0229  ,       0 ,   1.0000],
                [1.0249  ,  1.0000 ,   1.0000],
                [1.0242  ,  1.0000 ,   1.0000],
                [1.0234  ,  1.0000 ,   1.0000],
                [1.0227  ,  1.0000 ,   1.0000],
                [1.0219  ,  1.0000 ,   1.0000],
                [1.0211  ,  1.0000 ,   1.0000],
                [1.0203  ,  1.0000 ,   1.0000],
                [1.0195  ,  1.0000 ,   1.0000],
                [1.0189  ,  1.0000 ,   1.0000],
                [1.0182  ,  1.0000 ,   1.0000],
                [1.0175  ,       0 ,   1.0000],
                [1.0194  ,  1.0000 ,   1.0000],
                [1.0187  ,  1.0000 ,   1.0000],
                [1.0181  ,  1.0000 ,   1.0000],
                [1.0174  ,  1.0000 ,   1.0000],
                [1.0167  ,  1.0000 ,   1.0000],
                [1.0161  ,  1.0000 ,   1.0000],
                [1.0154  ,  1.0000 ,   1.0000],
                [1.0148  ,  1.0000 ,   1.0000],
                [1.0141  ,  1.0000 ,   1.0000],
                [1.0135  ,       0 ,   1.0000],
                [1.0152  ,       0 ,   1.0000],
                [1.0171  ,       0 ,   1.0000],
                [1.0189  ,  1.0000 ,   1.0000],
                [1.0182  ,  1.0000 ,   1.0000],
                [1.0176  ,  1.0000 ,   1.0000],
                [1.0170  ,  1.0000 ,   1.0000],
                [1.0163  ,  1.0000 ,   1.0000],
                [1.0157  ,  1.0000 ,   1.0000],
                [1.0151  ,  1.0000 ,   1.0000],
                [1.0145  ,       0 ,   1.0000],
                [1.0161  ,  1.0000 ,   1.0000],
                [1.0155  ,  1.0000 ,   1.0000],
                [1.0149  ,  1.0000 ,   1.0000],
                [1.0143  ,  1.0000 ,   1.0000],
                [1.0136  ,  1.0000 ,   1.0000],
                [1.0130  ,  1.0000 ,   1.0000],
                [1.0124  ,  1.0000 ,   1.0000],
                [1.0118  ,  1.0000 ,   1.0000],
                [1.0111  ,  1.0000 ,   1.0000],
                [1.0105  ,       0 ,   1.0000],
                [1.0122  ,  1.0000 ,   1.0000],
                [1.0116  ,  1.0000 ,   1.0000],
                [1.0110  ,  1.0000 ,   1.0000],
                [1.0103  ,  1.0000 ,   1.0000],
                [1.0097  ,  1.0000 ,   1.0000],
                [1.0091  ,  1.0000 ,   1.0000],
                [1.0086  ,  1.0000 ,   1.0000],
                [1.0081  ,  1.0000 ,   1.0000],
                [1.0075  ,       0 ,   1.0000],
                [1.0090  ,  1.0000 ,   1.0000],
                [1.0085  ,  1.0000 ,   1.0000],
                [1.0080  ,  1.0000 ,   1.0000],
                [1.0074  ,  1.0000 ,   1.0000],
                [1.0069  ,  1.0000 ,   1.0000],
                [1.0064  ,       0 ,   1.0000],
                [1.0079  ,       0 ,   1.0000],
                [1.0093  ,  1.0000 ,   1.0000],
                [1.0088  ,  1.0000 ,   1.0000],
                [1.0083  ,  1.0000 ,   1.0000],
                [1.0078  ,  1.0000 ,   1.0000],
                [1.0073  ,  1.0000 ,   1.0000],
                [1.0067  ,  1.0000 ,   1.0000],
                [1.0062  ,       0 ,   1.0000],
                [1.0077  ,       0 ,   1.0000],
                [1.0091  ,       0 ,   1.0000],
                [1.0107  ,  1.0000 ,   1.0000],
                [1.0101  ,  1.0000 ,   1.0000],
                [1.0095  ,  1.0000 ,   1.0000],
                [1.0090  ,       0 ,   1.0000],
                [1.0105  ,  1.0000 ,   1.0000],
                [1.0099  ,  1.0000 ,   1.0000],
                [1.0094  ,  1.0000 ,   1.0000],
                [1.0089  ,  1.0000 ,   1.0000],
                [1.0084  ,  1.0000 ,   1.0000],
                [1.0079  ,       0 ,   1.0000],
                [1.0093  ,       0 ,   1.0000],
                [1.0108  ,  1.0000 ,   1.0000],
                [1.0102  ,  1.0000 ,   1.0000],
                [1.0097  ,  1.0000 ,   1.0000],
                [1.0092  ,  1.0000 ,   1.0000],
                [1.0087  ,  1.0000 ,   1.0000],
                [1.0082  ,  1.0000 ,   1.0000],
                [1.0077  ,  1.0000 ,   1.0000],
                [1.0073  ,       0 ,   1.0000],
                [1.0086  ,       0 ,   1.0000],
                [1.0100  ,  1.0000 ,   1.0000],
                [1.0095  ,  1.0000 ,   1.0000],
                [1.0090  ,  1.0000 ,   1.0000],
                [1.0085  ,  1.0000 ,   1.0000],
                [1.0080  ,       0 ,   1.0000],
                [1.0094  ,  1.0000 ,   1.0000],
                [1.0089  ,  1.0000 ,   1.0000],
                [1.0084  ,       0 ,   1.0000],
                [1.0097  ,  1.0000 ,   1.0000],
                [1.0093  ,  1.0000 ,   1.0000],
                [1.0088  ,       0 ,   1.0000],
                [1.0101  ,       0 ,   1.0000],
                [1.0117  ,  1.0000 ,   1.0000],
                [1.0111  ,  1.0000 ,   1.0000],
                [1.0106  ,  1.0000 ,   1.0000],
                [1.0100  ,  1.0000 ,   1.0000],
                [1.0095  ,  1.0000 ,   1.0000],
                [1.0091  ,  1.0000 ,   1.0000],
                [1.0086  ,  1.0000 ,   1.0000],
                [1.0082  ,  1.0000 ,   1.0000],
                [1.0077  ,  1.0000 ,   1.0000],
                [1.0073  ,  1.0000 ,   1.0000],
                [1.0068  ,       0 ,   1.0000],
                [1.0081  ,  1.0000 ,   1.0000],
                [1.0076  ,  1.0000 ,   1.0000],
                [1.0072  ,  1.0000 ,   1.0000],
                [1.0067  ,  1.0000 ,   1.0000],
                [1.0063  ,  1.0000 ,   1.0000],
                [1.0059  ,  1.0000 ,   1.0000],
                [1.0054  ,  1.0000 ,   1.0000],
                [1.0050  ,  1.0000 ,   1.0000],
                [1.0046  ,  1.0000 ,   1.0000],
                [1.0041  ,  1.0000 ,   1.0000],
                [1.0037  ,  1.0000 ,   1.0000],
                [1.0033  ,  1.0000 ,   1.0000],
                [1.0028  ,  1.0000 ,   1.0000],
                [1.0024  ,  1.0000 ,   1.0000],
                [1.0019  ,  1.0000 ,   1.0000],
                [1.0014  ,  1.0000 ,   1.0000],
                [1.0009  ,       0 ,   1.0000],
                [1.0022  ,  1.0000 ,   1.0000],
                [1.0018  ,  1.0000 ,   1.0000],
                [1.0013  ,  1.0000 ,   1.0000],
                [1.0008  ,  1.0000 ,   1.0000],
                [1.0003  ,       0 ,   1.0000],
                [1.0016  ,  1.0000 ,   1.0000],
                [1.0012  ,       0 ,   1.0000],
                [1.0024  ,  1.0000 ,   1.0000],
                [1.0020  ,  1.0000 ,   1.0000],
                [1.0015  ,  1.0000 ,   1.0000],
                [1.0011  ,  1.0000 ,   1.0000],
                [1.0006  ,       0 ,   1.0000],
                [1.0019  ,  1.0000 ,   1.0000],
                [1.0014  ,  1.0000 ,   1.0000],
                [1.0009  ,  1.0000 ,   1.0000],
                [1.0005  ,  1.0000 ,   1.0000],
                [1.0000  ,  1.0000 ,   1.0000],
                [0.9996  ,  1.0000 ,   1.0000],
                [0.9992  ,  1.0000 ,   1.0000],
                [0.9988  ,  1.0000 ,   1.0000],
                [0.9984  ,       0 ,   1.0000],
                [0.9995  ,  1.0000 ,   1.0000],
                [0.9991  ,  1.0000 ,   1.0000],
                [0.9987  ,  1.0000 ,   1.0000],
                [0.9984  ,  1.0000 ,   1.0000],
                [0.9980  ,  1.0000 ,   1.0000],
                [0.9976  ,  1.0000 ,   1.0000],
                [0.9972  ,  1.0000 ,   1.0000],
                [0.9969  ,  1.0000 ,   1.0000],
                [0.9965  ,       0 ,   1.0000],
                [0.9975  ,  1.0000 ,   1.0000],
                [0.9972  ,  1.0000 ,   1.0000],
                [0.9968  ,  1.0000 ,   1.0000],
                [0.9965  ,  1.0000 ,   1.0000],
                [0.9961  ,  1.0000 ,   1.0000],
                [0.9957  ,       0 ,   1.0000],
                [0.9968  ,  1.0000 ,   1.0000],
                [0.9964  ,  1.0000 ,   1.0000],
                [0.9961  ,  1.0000 ,   1.0000],
                [0.9957  ,  1.0000 ,   1.0000],
                [0.9953  ,  1.0000 ,   1.0000],
                [0.9950  ,  1.0000 ,   1.0000],
                [0.9946  ,  1.0000 ,   1.0000],
                [0.9943  ,       0 ,   1.0000],
                [0.9952  ,  1.0000 ,   1.0000],
                [0.9949  ,  1.0000 ,   1.0000],
                [0.9945  ,  1.0000 ,   1.0000],
                [0.9942  ,  1.0000 ,   1.0000],
                [0.9938  ,  1.0000 ,   1.0000],
                [0.9935  ,  1.0000 ,   1.0000],
                [0.9931  ,  1.0000 ,   1.0000],
                [0.9927  ,  1.0000 ,   1.0000],
                [0.9923  ,  1.0000 ,   1.0000],
                [0.9920  ,       0 ,   1.0000],
                [0.9930  ,       0 ,   1.0000],
                [0.9939  ,  1.0000 ,   1.0000],
                [0.9936  ,      0  ,  1.0000],
                [0.9945  ,       0 ,   1.0000],
                [0.9954  ,       0 ,   1.0000],
                [0.9964  ,  1.0000 ,   1.0000],
                [0.9961  ,  1.0000 ,   1.0000],
                [0.9957  ,       0 ,   1.0000],
                [0.9967  ,  1.0000 ,   1.0000],
                [0.9964  ,  1.0000 ,   1.0000],
                [0.9960  ,  1.0000 ,   1.0000],
                [0.9957  ,       0 ,   1.0000],
                [0.9966  ,  1.0000 ,   1.0000],
                [0.9963  ,       0 ,   1.0000],
                [0.9973  ,  1.0000 ,   1.0000],
                [0.9969  ,  1.0000 ,   1.0000],
                [0.9966  ,       0 ,   1.0000],
                [0.9975  , 1.0000  ,  1.0000]])

''' Now we fit this dataset using the settings for a 2AFC experiment and a
 normal distribution. See demo_004 for arguments why we pass a stimulus 
 range here. '''

options = dict()
options['expType'] = '2AFC'
options['sigmoidName'] = 'norm'
options['stimulusRange'] = array([.25,1.75])
res = psignifit(data,options)

''' Note that this took a bit longer than usual.
 Let's have a look at what psignifit did automatically here:
'''

plotPsych(res)

''' Each block contains only very few trials  Thus the beta-binomial model 
 cannot help much to correct for overdispersion. Furthermore the many 
 lines of data slow psignifit down. '''

''' Psignifit only pools if you collected more than options.nblocks blocks of
 data and then sets options.nblocks to be the number of blocks after
 pooling.
 If psignifit pools, it starts at the beginning and pools stimulus levels,
 which differ by maximally options.poolxTol in stimulus level and are
 separated by maximally options.poolMaxGap trials of other stimulus
 levels. Furthermore it stops collecting trials for a block once the block
 contains options.poolMaxLength trials. '''

''' The default settings for the pooling behaviour are: '''
print(res['options']['poolxTol'])
print(res['options']['poolMaxGap'])
print(res['options']['poolMaxLength'])
print(res['options']['nblocks'])

''' This means that by default psignifit pooled only trials collected 
 at the exact same stimulus level, but all of them.

 We can allow psignifit to pool trials which differ by up to 0.01 to
 reach more pooling: '''

options['poolxTol'] = 0.01

resPool1 = psignifit(data,options)

''' Lets have a look at this function: '''
plotPsych(resPool1)

''' Now we have some quite strong pooling.'''

''' The other two options allow us to restric which trials should be pooled
 again. For example we could restrict the number of trials to 25 per
 block: '''
options['poolMaxLength'] = 25

resPool2 = psignifit(data,options)

plotPsych(resPool2)

''' This brakes the large blocks up again allowing us to notice if there was
 more variability over time than expected. '''

''' The last option gives us a different rule to achieve something in a 
 similar direction: We can enforce that a block must be collected en bloc
 like this: '''
options['poolMaxLength'] = inf
options['poolMaxGap'] = 0

resPool3 = psignifit(data,options)

plotPsych(resPool3)

''' Values between 0 and infinity will allow "gaps" of maximally 
 options.poolMaxGap trials which are not included into the block (because
 their stimulus level differs to much). '''

''' Of course all pooling options can be combined. '''



