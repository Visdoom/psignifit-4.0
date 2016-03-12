# -*- coding: utf-8 -*-
"""
get the marginal posterior distributions from your result
function [marginal,x,weight]=marginalize(result,dimension)
this function gets the marginal distribution for result
It returns 3 variables:

  marginal - the values of the marginal posterior
  x        - the gridpoints at which the posterior is sampled
  weight   - the integration weight for this point

This also works for more than one dimension in dimension to obtain
multidimensional marginals, x then becomes a cell array of the 1D ticks
of the grid.

"""
import numpy as np
def marginalize(result, dimension):
        
      #
      assert(dimension.all() in range(0,5), 'the dimensions you want to marginalize to  should be given as a vector of numbers 0 to 4')
        
      d = len(result['X1D'])
        
      if len(dimension) == 1 and ('marginals' in result.keys()) and len(result['marginals']) >= dimension:
          marginal = result['marginals'][dimension]
          weight = result['marginalsW'][dimension]
          x = result['marginalsX'][dimension]
      else:
          if not('Posterior' in result.keys()):
              raise ValueError('marginals cannot be computed anymore because posterior was dropped')
          else:
              assert(np.shape(result['Posterior']) == np.shape(result['weight']), 'dimensions mismatch in marginalization')
                
              if len(dimension) == 1:
                  x = result['X1D'][dimension][:]
              else:
                  x = np.nan
              
              # calculate mass at each grid point
              marginal = result['weight']*result['Posterior']
                
              weight = result['weight']
               
              for i in range(0,d):
                  if not(any(i == dimension)) and marginal.shape[i] > 1:
                      marginal = np.sum(marginal,i)
                      weight = np.sum(weight,i)/(np.max(result.X1D[i])-np.min(result.X1D[i]))
              marginal = marginal/weight
              
              if len(dimension) == 1:
                  marginal = np.reshape(marginal, [],1)
                  weight = np.reshape(weight, [],1)
                  x = np.reshape(x,[],1)
              else:
                  marginal = np.array(marginal)
                  weight = np.array(weight)
                  x = result['X1D'][np.sort(dimension)] #TODO why the sort here?
                
      return (marginal, x, weight)

if __name__ == "__main__":
    import sys
    marginalize(sys.argv[1], sys.argv[2])

