# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:33:31 2016

@author: root
"""
from numpy import log, exp, sqrt, tan
from scipy.special import betainc,erfcinv, erfc

def my_norminv(p,mu,sigma):
    
    x0 = -sqrt(2)*erfcinv(2*p)
    x = sigma*x0 + mu
    
    return x

def my_normcdf(x,mu,sigma):
    z = (x-mu) /sigma
    p = .5*erfc(-z/sqrt(2))
    
    return p

    
def my_t1cdf(x):
    '''    
    cumulative distribution function of a t-dist. with 1 degree of freedom
    function p=my_t1cdf(x)
    input
          x = point
          output
          p = cumulative probability

    see also: tcdf 
    '''
    xsq=x*x;
    p = betainc(1 / (1 + xsq), 1/2, 1/2) / 2
    p[x>0]=1-p[x>0]
    
    return p    

def my_t1icdf(p):
    x = tan(pi * (p - 0.5));
    return x    
    
def getSigmoidHandle(options):
    '''
    creates a function handle to a specific sigmoid
    function Handle=getSigmoidHandle(options)
    This function creates a function handle to the sigmoid specified by its
    name in options.sigmoidName. 
    Additional parameter is the options.widthalpha which specifies the
    scaling of the width by
    width= psi^(-1)(1-alpha) - psi^(-1)(alpha)
    where psi^(-1) is the inverse of the sigmoid function.
    '''
    
    # TODO isstruct check 
    if hasattr(options, 'widthalpha'):
        options.widthalpha = .05
    
    alpha = options.widthalpha
    sigmoid = options.sigmoidName
    PC = options.threshPC
    
    if isinstance(sigmoid,str):
        sig = sigmoid.lower()
        if sig in ['gauss ', 'norm']:   # cumulative normal distribution
            C = my_norminv(1-alpha, 0,1) - my_norminv(alpha,0,1)
            handle = lambda X,m,width: my_normcdf(X, (m-my_norminv(PC,0,width/C)), width/C)
        elif sig == 'logistic':         # logistic function
            handle = lambda X,m, width: 1/(1 + exp(-2 * log(1/alpha -1) / width *(X-m) + log(1/PC -1)))
        elif sig == 'gumbel':           #gumbel
            C = log(-log(alpha)) - log(-log(1-alpha))
            handle = lambda X, m, width: 1 - exp(-exp(C / width * (X-m) + log(-log(1-PC))))
        elif sig == 'rgumbel':
            C = log(-log(1-alpha)) - log(-log(alpha))
            handle = lambda X, m, width: exp(-exp( C / width * (X-m) + log(-log(PC))))
        elif sig == 'logn':
            C = my_norminv(1-alpha,0,1) - my_norminv(alpha,0,1)
            handle = lambda X, m, width: my_normcdf(log(X), m-my_norminv(PC,0,width/C), width / C)
        elif sig == 'weibull':
            C = log(-log(alpha)) - log(-log(1-alpha))
            handle = lambda X, m, width: 1 - exp(-exp(C/ width * (log(X)-m) + log(-log(1-PC))))
        elif sig in ['tdist', 'student', 'heavytail']:
            C = (my_t1icdf(1-alpha) - my_t1icdf(alpha))
            handle = lambda X, m, width: my_t1cdf(C*(X-m)/ width + my_t1icdf(PC))       
        else:
            raise ValueError('unknown sigmoid function')
    elif hasattr(sigmoid, '__call__'):
        handle = sigmoid
        
    return handle