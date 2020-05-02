'''
BSD 3-Clause License

Copyright (c) 2020, Lawrence Livermore National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

'''
https://github.com/LLNL/fastcam

A toolkit for efficent computation of saliency maps for explainable 
AI attribution.

This work was performed under the auspices of the U.S. Department of Energy 
by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 
and was supported by the LLNL-LDRD Program under Project 18-ERD-021 and 
Project 17-SI-003. 

Software released as LLNL-CODE-802426.

See also: https://arxiv.org/abs/1911.11293
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussNorm2D(nn.Module):
    r'''
        This will normalize a saliency map to range from 0 to 1 via normal cumulative distribution function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any real valued number (supported by hardware)
        Output will range from 0 to 1
        
        Notes:
        
        (1) GammaNorm2D will produce slightly better results
            The sum ROAR/KAR will improve from 1.44 to 1.45 for FastCAM using GradCAM.
        (2) This method is a bit less expensive than GammaNorm2D.
    '''
    
    def __init__(self, const_mean=None, const_std=None):
        
        super(GaussNorm2D, self).__init__()   
        
        assert isinstance(const_mean,float)    or const_mean is None
        assert isinstance(const_std,float)     or const_std is None
        
        self.const_mean = const_mean
        self.const_std  = const_std
        
    def forward(self, x):
        r'''
            Input: 
                x:     A Torch Tensor image with shape [batch size x height x width] e.g. [64,7,7]
            Return:
                x:     x Normalized by computing mean and std over each individual batch item and squashed with a 
                       Normal/Gaussian CDF.  
        '''
        assert torch.is_tensor(x), "Input must be a Torch Tensor"
        assert len(x.size()) == 3, "Input should be sizes [batch size x height x width]"
        
        s0      = x.size()[0]
        s1      = x.size()[1]
        s2      = x.size()[2]

        x       = x.reshape(s0,s1*s2) 
        
        r'''
            Compute Mean
        '''
        if self.const_mean is None:
            m       = x.mean(dim=1)
            m       = m.reshape(m.size()[0],1)
        else:
            m       = self.const_mean
            
        r'''
            Compute Standard Deviation
        '''
        if self.const_std is None:
            s       = x.std(dim=1)
            s       = s.reshape(s.size()[0],1)
        else:
            s       = self.const_std
        
        r'''
            The normal cumulative distribution function is used to squash the values from within the range of 0 to 1
        '''
        x       = 0.5*(1.0 + torch.erf((x-m)/(s*torch.sqrt(torch.tensor(2.0)))))
                
        x       = x.reshape(s0,s1,s2)
            
        return x  
    
# *******************************************************************************************************************  
# ******************************************************************************************************************* 
class GammaNorm2D(nn.Module):
    r'''
        This will normalize a saliency map to range from 0 to 1 via gamma cumulative distribution function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any positive real valued number (supported by hardware)
        Output will range from 0 to 1
        
        Notes:
        
        (1) When applied to each saliency map, this method produces slightly better results than GaussNorm2D.
            The sum ROAR/KAR will improve from 1.44 to 1.45 for FastCAM using GradCAM.
        (2) This method is a bit more expensive than GaussNorm2D.
    '''
    
    def __init__(self):
        
        super(GammaNorm2D, self).__init__()   
        
        r'''
            Chebyshev polynomials for Gamma Function
        '''
        self.cheb = torch.tensor([676.5203681218851,
                                  -1259.1392167224028,
                                  771.32342877765313,
                                  -176.61502916214059,
                                  12.507343278686905,
                                  -0.13857109526572012,
                                  9.9843695780195716e-6,
                                  1.5056327351493116e-7
                                  ])
        
        self.two_pi = torch.tensor(math.sqrt(2.0*3.141592653589793))
        
    def _gamma(self,z):
        r'''
            Gamma Function:
        
            http://mathworld.wolfram.com/GammaFunction.html
            
            https://en.wikipedia.org/wiki/Gamma_function#Weierstrass's_definition
            
            https://en.wikipedia.org/wiki/Lanczos_approximation#Simple_implementation
            
                gives us gamma(z + 1)
                Our version makes some slight changes and is more stable. 
            
            Notes: 
            
            (1) gamma(z) = gamma(z+1)/z
            (2) The gamma function is essentially a factorial function that supports real numbers
                so it grows very quickly. If z = 18 the result is 355687428096000.0
            
            Input is an array of positive real values. Zero is undefined. 
            Output is an array of real postive values. 
        ''' 
        
        x = torch.ones_like(z) * 0.99999999999980993
        
        for i in range(8):
            i1  = torch.tensor(i + 1.0)
            x   = x + self.cheb[i] / (z + i1)
            
        t = z + 8.0 - 0.5
        y = self.two_pi * t.pow(z+0.5) * torch.exp(-t) * x
        
        y = y / z
        
        return y   
    
    def _lower_incl_gamma(self,s,x, iter=8):
        r'''
            Lower Incomplete Gamma Function:
            
            This has been optimized to call _gamma and pow only once
            The gamma function is very expensive to call over all pixels, as we might do here. 
        
            See: https://en.wikipedia.org/wiki/Incomplete_gamma_function#Holomorphic_extension
        '''
        iter    = iter - 2
        
        gs      = self._gamma(s)
        
        L       = x.pow(s) * gs * torch.exp(-x)
        
        r'''
            For the gamma function: f(x + 1) = x * f(x)
        '''
        gs      *= s                  # Gamma(s + 1)
        R       = torch.reciprocal(gs) * torch.ones_like(x)
        X       = x                   # x.pow(1)
        
        for k in range(iter):
            gs      *= s + k + 1.0    # Gamma(s + k + 2)
            R       += X / gs 
            X       = X*x             # x.pow(k+1)
        
        gs      *= s + iter + 1.0     # Gamma(s + iter + 2)
        R       += X / gs
        
        return  L * R
    
    def _trigamma(self,x):
        r''' 
            Trigamma function:
            
            https://en.wikipedia.org/wiki/Trigamma_function
            
            We need the first line since recursion is not good for x < 1.0
            Note that we take + torch.reciprocal(x.pow(2)) at the end because:
            
            trigamma(z) = trigamma(z + 1) + 1/z^2
        '''
        
        z   = x + 1.0
        
        zz  = z.pow(2.0)
        a   = 0.2 - torch.reciprocal(7.0*zz)
        b   = 1.0 - a/zz 
        c   = 1.0 + b/(3.0 * z)
        d   = 1.0 + c/(2.0 * z)
        e   = d/z 
        
        e   = e + torch.reciprocal(x.pow(2.0))
     
        return e

    def _k_update(self,k,s):
        
        nm = torch.log(k) - torch.digamma(k) - s
        dn = torch.reciprocal(k) - self._trigamma(k)
        k2 = k - nm/dn
        
        return k2
            
    def _compute_ml_est(self, x, i=10, eps=0.0000001):
        r'''
            Compute k and th parameters for the Gamma Probability Distribution. 
            
            This uses maximum likelihood estimation per Choi, S. C.; Wette, R. (1969)
            
            See: https://en.wikipedia.org/wiki/Gamma_distribution#Parameter_estimation
            
            Input is an array of real positive values. Zero is undefined, but we handle it. 
            Output is a single value (per image) for k and th
        '''
        
        r'''
            avoid log(0)
        '''
        x  = x + eps
        
        r'''
            Calculate s
            This is somewhat akin to computing a log standard deviation. 
        '''
        s  = torch.log(torch.mean(x,dim=1)) - torch.mean(torch.log(x),dim=1)
        
        r'''
            Get estimate of k to within 1.5%
        
            NOTE: K gets smaller as log variance s increases.
        '''
        s3 = s - 3.0
        rt = torch.sqrt(s3.pow(2.0) + 24.0 * s)
        nm = 3.0 - s + rt
        dn = 12.0 * s
        k  = nm / dn + eps

        r'''
            Do i Newton-Raphson steps to get closer than 1.5%
            For i=5 gets us within 4 or 5 decimal places
        '''
        for _ in range(i):
            k =  self._k_update(k,s)
        
        r'''
            prevent gamma(k) from being silly big or zero
            With k=18, gamma(k) is still 355687428096000.0
            This is because the Gamma function is a factorial function with support 
            for positive natural numbers (here we only support reals).
        '''
        k   = torch.clamp(k, eps, 18.0)
        
        th  = torch.reciprocal(k) * torch.mean(x,dim=1)
        
        return k, th
     
    def forward(self, x):
        r'''
            Input: 
                x:     A Torch Tensor image with shape [batch size x height x width] e.g. [64,7,7]
                       All values should be real positive (i.e. >= 0).  
            Return:
                x:     x Normalized by computing shape and scale over each individual batch item and squashed with a 
                       Gamma CDF.  
        '''
        
        assert torch.is_tensor(x), "Input must be a Torch Tensor"
        assert len(x.size()) == 3, "Input should be sizes [batch size x height x width]"
        
        s0      = x.size()[0]
        s1      = x.size()[1]
        s2      = x.size()[2]

        x       = x.reshape(s0,s1*s2) 
        
        r'''
            offset from just a little more than 0, keeps k sane
        '''
        x       = x - torch.min(x,dim=1)[0] + 0.0000001
        
        k,th    = self._compute_ml_est(x)
        
        '''
            Squash with a Gamma CDF for range within 0 to 1.
        '''
        x       = (1.0/self._gamma(k)) * self._lower_incl_gamma(k, x/th)
    
        r'''
            There are weird edge cases (e.g. all numbers are equal), prevent NaN
        '''
        x       = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
                
        x       = x.reshape(s0,s1,s2)
            
        return x  
    
# *******************************************************************************************************************                   
# ******************************************************************************************************************* 
class RangeNorm2D(nn.Module):
    r'''
        This will normalize a saliency map to range from 0 to 1 via linear range function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any real valued number (supported by hardware)
        Output will range from 0 to 1
        
        Parameters:
            full_norm:     This forces the values to range completely from 0 to 1. 
    '''
    
    def __init__(self, full_norm=True, eps=10e-10):
        
        super(RangeNorm2D, self).__init__()   
        self.full_norm  = full_norm
        self.eps        = eps
                
    def forward(self, x):
        r'''
            Input: 
                x:     A Torch Tensor image with shape [batch size x height x width] e.g. [64,7,7]
                       All values should be real positive (i.e. >= 0).  
            Return:
                x:     x Normalized by dividing by either the min value or the range between max and min. 
                       Each max/min is computed for each batch item.  
        '''
        assert torch.is_tensor(x), "Input must be a Torch Tensor"
        assert len(x.size()) == 3, "Input should be sizes [batch size x height x width]"
        
        s0      = x.size()[0]
        s1      = x.size()[1]
        s2      = x.size()[2]

        x       = x.reshape(s0,s1*s2) 
        
        xmax    = x.max(dim=1)[0].reshape(s0,1)
                
        if self.full_norm:
            xmin    = x.min(dim=1)[0].reshape(s0,1)
        
            nval    = x     - xmin
            range   = xmax  - xmin
        else:
            nval    = x
            range   = xmax
        
        r'''
            prevent divide by zero by setting zero to a small number
            
            Simply adding eps does not work will in this case. So we use torch.where to set a minimum value. 
        '''
        eps_mat = torch.zeros_like(range) + self.eps
        range   = torch.where(range > self.eps, range, eps_mat)
        
        x       = nval / range
        
        x       = x.reshape(s0,s1,s2)
        
        return x
