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
import torch
import torch.nn as nn
import torch.nn.functional as F

import maps
import misc

# *******************************************************************************************************************
class SMOEScaleMap(nn.Module):
    r'''
        Compute SMOE Scale on a 4D tensor. This acts as a standard PyTorch layer. 
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) A tensor with only positive values. (After a ReLU)
        
        Output is a 3D tensor of size [batch x height x width] 
    '''
    def __init__(self, run_relu=False):
        
        super(SMOEScaleMap, self).__init__()
        
        r'''
            SMOE Scale must take in values > 0. Optionally, we can run a ReLU to do that.
        '''
        if run_relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = None
               
    def forward(self, x):

        assert(torch.is_tensor(x))
        assert(len(x.size()) > 2)
        
        
        r'''
            If we do not have a convenient ReLU to pluck from, we can do it here
        '''
        if self.relu is not None:
            x = self.relu(x)
                               
        r'''
            avoid log(0)
        '''
        x   = x + 0.0000001
        
        r'''
            This is one form. We can also use the log only form.
        '''
        m   = torch.mean(x,dim=1)
        k   = torch.log2(m) - torch.mean(torch.log2(x), dim=1)
        
        th  = k * m
        
        return th

# *******************************************************************************************************************
class StdMap(nn.Module):
    r'''
        Compute vanilla standard deviation on a 4D tensor. This acts as a standard PyTorch layer. 
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) Recommend a tensor with only positive values. (After a ReLU)
        
        Output is a 3D tensor of size [batch x height x width]
    '''
    def __init__(self):
        
        super(StdMap, self).__init__()
        
    def forward(self, x):
        
        assert(torch.is_tensor(x))
        assert(len(x.size()) > 2)
        
        x = torch.std(x,dim=1)
        
        return x

# *******************************************************************************************************************
class MeanMap(nn.Module):
    r'''
        Compute vanilla mean on a 4D tensor. This acts as a standard PyTorch layer. 
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) Recommend a tensor with only positive values. (After a ReLU)
        
        Output is a 3D tensor of size [batch x height x width]
    '''
    def __init__(self):
        
        super(MeanMap, self).__init__()
        
    def forward(self, x):
        
        assert(torch.is_tensor(x))
        assert(len(x.size()) > 2)
        
        x = torch.mean(x,dim=1)
        
        return x
    
# *******************************************************************************************************************
class MaxMap(nn.Module):
    r'''
        Compute vanilla mean on a 4D tensor. This acts as a standard PyTorch layer. 
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) Recommend a tensor with only positive values. (After a ReLU)
        
        Output is a 3D tensor of size [batch x height x width]
    '''
    def __init__(self):
        
        super(MaxMap, self).__init__()
        
    def forward(self, x):
        
        assert(torch.is_tensor(x))
        assert(len(x.size()) > 2)
        
        x = torch.max(x,dim=1)[0]
        
        return x   
    
# *******************************************************************************************************************
class TruncNormalEntMap(nn.Module):
    r'''
        Compute truncated normal entropy on a 4D tensor. This acts as a standard PyTorch layer. 
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) This should come BEFORE a ReLU and can range over any real value
        
        Output is a 3D tensor of size [batch x height x width]
    '''
    def __init__(self):
        
        super(TruncNormalEntMap, self).__init__()
        
        self.c1 = torch.tensor(0.3989422804014327)  # 1.0/math.sqrt(2.0*math.pi)
        self.c2 = torch.tensor(1.4142135623730951)  # math.sqrt(2.0)
        self.c3 = torch.tensor(4.1327313541224930)  # math.sqrt(2.0*math.pi*math.exp(1))
    
    def _compute_alpha(self, mean, std, a=0):
        
        alpha = (a - mean)/std
        
        return alpha
        
    def _compute_pdf(self, eta):
        
        pdf = self.c1 * torch.exp(-0.5*eta.pow(2.0))
        
        return pdf
        
    def _compute_cdf(self, eta):
        
        e   = torch.erf(eta/self.c2)
        cdf = 0.5 * (1.0 + e)
        
        return cdf
    
    def forward(self, x):
        
        assert(torch.is_tensor(x))
        assert(len(x.size()) > 2)
 
        m   = torch.mean(x,   dim=1)
        s   = torch.std(x,    dim=1)
        a   = self._compute_alpha(m, s)
        pdf = self._compute_pdf(a)  
        cdf = self._compute_cdf(a) + 0.0000001  # Prevent log AND division by zero by adding a very small number
        Z   = 1.0 - cdf 
        T1  = torch.log(self.c3*s*Z)
        T2  = (a*pdf)/(2.0*Z)
        ent = T1 + T2

        return ent
   
# *******************************************************************************************************************                   
# ******************************************************************************************************************* 
class Normalize2D(nn.Module):
    r'''
        This will normalize a saliency map to range from 0 to 1 via normal cumulative distribution function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any real valued number (supported by hardware)
        Output will range from 0 to 1
    '''
    
    def __init__(self, const_mean=None, const_std=None):
        
        super(Normalize2D, self).__init__()   
        
        assert(isinstance(const_mean,float)    or const_mean is None)
        assert(isinstance(const_std,float)     or const_std is None)
        
        self.const_mean = const_mean
        self.const_std  = const_std
        
    def forward(self, x):
        r'''
            Original shape is something like [64,7,7] i.e. [batch size x height x width]
        '''
        assert(torch.is_tensor(x))
        assert(len(x.size()) == 3) 
        
        s0      = x.size()[0]
        s1      = x.size()[1]
        s2      = x.size()[2]

        x       = x.reshape(s0,s1*s2) 
        
        if self.const_mean is None:
            m       = x.mean(dim=1)
            m       = m.reshape(m.size()[0],1)
        else:
            m       = self.const_mean
            
        if self.const_std is None:
            s       = x.std(dim=1)
            s       = s.reshape(s.size()[0],1)
        else:
            s       = seld.const_std
        
        r'''
            The normal cumulative distribution function is used to squash the values from 0 to 1
        '''
        x       = 0.5*(1.0 + torch.erf((x-m)/(s*torch.sqrt(torch.tensor(2.0)))))
                
        x       = x.reshape(s0,s1,s2)
            
        return x   
    
# ******************************************************************************************************************* 
class GammaNorm2D(nn.Module):
    r'''
        This will normalize a saliency map to range from 0 to 1 via normal cumulative distribution function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any real valued number (supported by hardware)
        Output will range from 0 to 1
    '''
    
    def __init__(self):
        
        super(GammaNorm2D, self).__init__()   
        
    def _gamma(self,z):
    
        # Chebyshev polynomials
        p = torch.tensor([676.5203681218851,
                          -1259.1392167224028,
                          771.32342877765313,
                          -176.61502916214059,
                          12.507343278686905,
                          -0.13857109526572012,
                          9.9843695780195716e-6,
                          1.5056327351493116e-7
                          ])
        
        # Removed to get gamma(z + 1)
        #z = z - 1.0
        
        x = torch.ones_like(z) * 0.99999999999980993
        
        for i in range(8):
            i1  = torch.tensor(i + 1.0)
            x   = x + p[i] / (z + i1)
            
        t = z + 8.0 - 0.5
        y = math.sqrt(2.0*3.141592653589793) * t.pow(z+0.5) * torch.exp(-t) * x
        
        # Added to get gamma(z)
        y = y / z
        
        return y   
    
    def _lower_incl_gamma(self,s,x):
        
        L = x.pow(s) * self._gamma(s) * torch.exp(-x)
        
        R = torch.zeros_like(L)
        
        for k in range(8):
            
            R += x.pow(k) / self._gamma(s + k + 1)
            
        return L * R
    
    # UNUSED
    def _log_gamma(self,x):
    
        # We need this line since recursion is not good for x < 1.0
        # Note that we take -log(x) at the end to remove this because:
        #
        #     log_gamma(z) = log_gamma(z + 1) - log(z)
        #
        
        z   = x + 1.0
        
        # First three lines give a very good approximation if z < 1.0
        T1  = z * torch.log(z)
        T2  = z.clone()
        T3  = 0.5 * torch.log(z/(2.0*3.141592653589793))
        
        T4  = torch.reciprocal(12.0*z)
        z3  = z.pow(3)
        T5  = torch.reciprocal(360.0*z3)
        z5  = z.pow(5)
        T6  = torch.reciprocal(1260.0*z5)
        z7  = z.pow(7)
        T7  = (1.0/30.0)/(8.0*7.0*z7)
    
        l   = T1 - T2 - T3 + T4 - T5 + T6 - T7 - torch.log(x)
        
        return l 
    
    def _trigamma(self,x):
    
        # We need this line since recursion is not good for x < 1.0
        # Note that we take + torch.reciprocal(x.pow(2)) at the end because:
        #
        #     trigamma(z) = trigamma(z + 1) + 1/z^2
        #
        
        z   = x + 1.0
        
        zz  = z.pow(2)
        a   = 0.2 - torch.reciprocal(7.0*zz)
        b   = 1.0 - a/zz 
        c   = 1.0 + b/(3.0 * z)
        d   = 1.0 + c/(2.0 * z)
        e   = d/z 
        
        e   = e + torch.reciprocal(x.pow(2))
     
        return e
    
    def _k_update(self,k,s):
        
        nm = torch.log(k) - torch.digamma(k) - s
        dn = torch.reciprocal(k) - self._trigamma(k)
        k2 = k - nm/dn
        
        return k2
        
    def _compute_ml_est(self, x, i=10):
        
        # avoid log(0)
        x  = x + 0.0000001
        
        #Calculate s
        # If x has been normalized, the first number is negative, the second number is positive (larger?)
        s  = torch.log(torch.mean(x,dim=1)) - torch.mean(torch.log(x),dim=1)
        
        # Get estimate of k to within 1.5%
        #
        # NOTE: K gets smaller as log variance s increases
        #
        s3 = s - 3.0
        rt = torch.sqrt(s3.pow(2) + 24.0 * s)
        nm = 3.0 - s + rt
        dn = 12.0 * s
        k  = nm / dn + 0.0000001

        # Do i Newton-Raphson steps to get closer than 1.5%
        # For i=5 gets us within 4 or 5 decimal places
        for _ in range(i):
            k =  self._k_update(k,s)
            
        th  = torch.reciprocal(k) * torch.mean(x,dim=1)
        
        return k, th
    
    def forward(self, x):
        r'''
            Original shape is something like [64,7,7] i.e. [batch size x height x width]
        '''
        assert(torch.is_tensor(x))
        assert(len(x.size()) == 3) 
        
        s0      = x.size()[0]
        s1      = x.size()[1]
        s2      = x.size()[2]

        x       = x.reshape(s0,s1*s2) 
        
        k,th    = self._compute_ml_est(x)
        
        # CHECK THIS TO MAKE SURE DIMS ARE CORRECT
        x       = (1.0/self._gamma(k)) * self._lower_incl_gamma(k, x/th)
        
        x       = x.reshape(s0,s1,s2)
            
        return x  
    
# *******************************************************************************************************************     
# *******************************************************************************************************************
class CombineSaliencyMaps(nn.Module): 
    r'''
        This will combine saliency maps into a single weighted saliency map. 
        
        Input is a list of 3D tensors or various sizes. 
        Output is a 3D tensor of size output_size
        
        num_maps specifies how many maps we will combine
        weights is an optional list of weights for each layer e.g. [1, 2, 3, 4, 5]
    '''
    
    def __init__(self, output_size=[224,224], map_num=5, weights=None, resize_mode='bilinear', magnitude=False, do_relu=False):
        
        super(CombineSaliencyMaps, self).__init__()
        
        assert(isinstance(output_size,list))
        assert(isinstance(map_num,int))
        assert(isinstance(resize_mode,str))    
        assert(len(output_size) == 2)
        assert(output_size[0] > 0)
        assert(output_size[1] > 0)
        assert(map_num > 0)
        
        r'''
            We support weights being None, a scaler or a list. 
            
            Depending on which one, we create a list or just point to one.
        '''
        if weights is None:
            self.weights = [1.0 for _ in range(map_num)]
        elif len(weights) == 1:
            assert(weights > 0)
            self.weights = [weights for _ in range(map_num)]   
        else:
            assert(len(weights) == map_num)        
            self.weights = weights
        
        self.weight_sum = 0
        
        for w in self.weights:
            self.weight_sum += w  
        
        self.map_num        = map_num
        self.output_size    = output_size
        self.resize_mode    = resize_mode
        self.magnitude      = magnitude
        self.do_relu        = do_relu
        
    def forward(self, smaps):
        
        r'''
            Input shapes are something like [64,7,7] i.e. [batch size x layer_height x layer_width]
            Output shape is something like [64,224,244] i.e. [batch size x image_height x image_width]
        '''

        assert(isinstance(smaps,list))
        assert(len(smaps) == self.map_num)
        assert(len(smaps[0].size()) == 3)   
        
        bn  = smaps[0].size()[0]
        cm  = torch.zeros((bn, 1, self.output_size[0], self.output_size[1]), dtype=smaps[0].dtype, device=smaps[0].device)
        ww  = []
        
        r'''
            Now get each saliency map and resize it. Then store it and also create a combined saliency map.
        '''
        if not self.magnitude:
            for i in range(len(smaps)):
                assert(torch.is_tensor(smaps[i]))
                wsz = smaps[i].size()
                w   = smaps[i].reshape(wsz[0], 1, wsz[1], wsz[2])
                w   = nn.functional.interpolate(w, size=self.output_size, mode=self.resize_mode, align_corners=False) 
                ww.append(w)
                cm  += (w * self.weights[i])
        else:
            for i in range(len(smaps)):
                assert(torch.is_tensor(smaps[i]))
                wsz = smaps[i].size()
                w   = smaps[i].reshape(wsz[0], 1, wsz[1], wsz[2])
                w   = nn.functional.interpolate(w, size=self.output_size, mode=self.resize_mode, align_corners=False) 
                w   = w*w 
                ww.append(w)
                cm  += (w * self.weights[i])
            
        cm  = cm / self.weight_sum
        cm  = cm.reshape(bn, self.output_size[0], self.output_size[1])
        
        ww  = torch.stack(ww,dim=1)
        ww  = ww.reshape(bn, self.map_num, self.output_size[0], self.output_size[1])
        
        if self.do_relu:
            cm = F.relu(cm)
            ww = F.relu(ww)
        
        return cm, ww 
            
# *******************************************************************************************************************
# ******************************************************************************************************************* 
class ScoreMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scores):
        
        ctx.save_for_backward(scores)
        return torch.tensor(1)
    
    @staticmethod
    def backward(ctx, grad):
                
        saved        = ctx.saved_tensors
        g_scores     = torch.ones_like(saved[0])
        
        return g_scores

# *******************************************************************************************************************         
class FastCAM(object):
    r"""
    Calculate SMOE Scale GradCAM salinecy map.
    
    This code is derived from pytorch-gradcam
    """
    def __init__(self, layers, model, method='V1', maps_method=maps.SMOEScaleMap, 
                 use_gpp=False, grad_pooling=None, pos_neg_map=False):
        
        assert(isinstance(layers, list))
        assert(isinstance(method, str))
        assert(isinstance(grad_pooling, str) or grad_pooling is None)
        assert(callable(maps_method))
        
        self.getSmap        = maps_method()
        self.layers         = layers
        self.model          = model
        self.method         = method
        self.use_gpp        = use_gpp
        self.grad_pooling   = grad_pooling
        self.pos_neg_map    = pos_neg_map
        self.forward        = True
        self.backward       = True
        
        if not pos_neg_map:
            self.getNorm        = maps.Normalize2D()
        else:
            self.getNorm        = maps.Normalize2D()#const_mean=0.0)
        
        self.activation_hooks   = []
        self.gradient_hooks     = []
        
        if self.method == 'V0':
            self.backward = False
        if self.method == 'V5':
            self.forward = False
            
        if self.forward:
            for i,l in enumerate(layers):
                h   = misc.CaptureLayerOutput(post_process=None)
                _   = self.model._modules[l].register_forward_hook(h)
                self.activation_hooks.append(h)

        if self.backward:    
            for i,l in enumerate(layers):
                if self.method=='V6' or self.method=='V5' or self.method=='V4' or self.method=='V3':
                    h   = misc.CaptureGradInput(post_process=None) # The gradient information leaving the layer
                else:
                    h   = misc.CaptureGradOutput(post_process=None) # The gradient information entering the layer

                _   = self.model._modules[l].register_backward_hook(h)
                self.gradient_hooks.append(h)
    
    def _forward_bsize_1(self, class_idx, logit, retain_graph):
        
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        if self.backward:
            self.model.zero_grad()
            score.backward(retain_graph=retain_graph)
    
    def _forward_bsize_N(self):
        
        pass    
    
    def _magnitude_pool2d(self, x, kernel_size=2, stride=2, padding=0, pos_max=False):
        
        # pick the max magnitude gradient in the pool, the one with the highest absolute value

        b, k, u, v          = x.size()

        p1  = F.max_pool2d(x,    kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=True)
        p2  = F.max_pool2d(-1*x, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=True) * -1

        d   = p1 + p2

        if pos_max:
            m   = torch.where(d >= 0.0, p1, torch.zeros_like(d))
        else:
            m   = torch.where(d >= 0.0, p1, p2)

        m   = nn.functional.interpolate(m, size=[u,v], mode='nearest')        

        return m 

    def __call__(self, input, class_idx=None, retain_graph=False, invert=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """

        # Don't compute grads if we do not need them
        with torch.set_grad_enabled(self.backward):

            b, c, h, w      = input.size()
    
            self.model.eval()
    
            logit           = self.model(input)
            
            if b == 1:
                self._forward_bsize_1(class_idx, logit, retain_graph)
            else:
                self._forward_bsize_N(class_idx, logit, retain_graph)
            
            saliency_maps = []
            
            for i,l in enumerate(self.layers):
            
                if self.backward:
                    gradients           = self.gradient_hooks[i].data
                    b, k, u, v          = gradients.size()
                    
                    # Make sure gradients are bounded on -1 to 1
                    #if self.method=='V5' or self.method=='V4':
                    #    sgrad               = torch.std(gradients) + 0.000001
                    #    gradients           = gradients/(sgrad*(3.14159/2.0))
                    #    gradients           = torch.tanh(gradients)
                    
                    if self.grad_pooling == 'avg':
                        gradients           = F.avg_pool2d(gradients, kernel_size=2, stride=2, padding=0, ceil_mode=True)
                        gradients           = nn.functional.interpolate(gradients, size=[u,v], mode='nearest') 
                    elif self.grad_pooling == 'max':
                        # this helps make gradients more selective for larger things when using V1, V2 or V3 
                        # Should also remove spurioous negative gradients
                        gradients           = F.max_pool2d(gradients, kernel_size=2, stride=2, padding=0, ceil_mode=True)
                        gradients           = nn.functional.interpolate(gradients, size=[u,v], mode='nearest') 
                    elif self.grad_pooling == 'mag':
                        gradients           = self._magnitude_pool2d(gradients, kernel_size=2, stride=2, padding=0)
                    
                if self.forward:
                    activations         = self.activation_hooks[i].data
                    b, k, u, v          = activations.size()
                    
                    #sact                = torch.std(activations) + 0.000001
                    #activations         = activations/sact
                    #activations         = torch.tanh(activations)
                    
                    # Just to be sure, we may not be following a ReLU layer
                    activations         = F.relu(activations)
                
                if invert:
                    gradients *= -1
                
                # Do GradCAM Plus Plus preprocess on gradients?
                # This biases for layers with less activation
                # Only makes a noticable difference for V2 and V4
                if self.use_gpp and self.backward and self.forward:
                    alpha_num           = gradients.pow(2)
                    alpha_denom         = gradients.pow(2).mul(2) + \
                                            activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
                    # NOT NEEDED: since we already add 1e-7
                    #alpha_denom         = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            
                    alpha               = alpha_num.div(alpha_denom+1e-7)
                    # OPTIONAL: Adding in the score is optional since we normalize the results any way
                    #positive_gradients  = F.relu(self.score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
                    positive_gradients  = F.relu(gradients)
                    gradients           = alpha*positive_gradients
                
                if self.method=='V0':
                    # Without Gradients, original SMOE Scale
                    saliency_map        = activations
                    
                elif self.method=='V1':
                    # V1 SIGN over layer means
                    alpha               = gradients.view(b, k, -1).mean(2)
                    weights             = alpha.view(b, k, 1, 1)
                    weights             = torch.sign(weights) 
                    saliency_map        = weights*activations
                    
                elif self.method=='V2':
                    # V2 Original method
                    alpha               = gradients.view(b, k, -1).mean(2)
                    weights             = alpha.view(b, k, 1, 1)
                    saliency_map        = weights*activations
                    
                elif self.method=='V3':
                    # V3 SIGN over all values
                    weights             = torch.sign(gradients) 
                    saliency_map        = weights*activations
                    
                elif self.method=='V4':
                    # V4 multiply over all values
                    saliency_map        = gradients*activations
                elif self.method=='V5':
                    # V5 Gradients Only
                    saliency_map        = gradients    
                elif self.method=='V6':
                    # V6 Sign of Gradients Only
                    saliency_map        = torch.sign(gradients)

                '''    
                elif self.method=='V6':
                    # Conditional entropy between postive and negative gradients 
                    alpha               = gradients.view(b, k, -1).mean(2)
                    weights_a           = F.relu(activations*alpha.view(b, k, 1, 1))    
                    weights_b           = F.relu(activations*alpha.view(b, k, 1, 1) * -1)
                    weights_a           = self.getSmap(weights_a)
                    weights_b           = self.getSmap(weights_b)
                    saliency_map        = self.getNorm(weights_a * torch.log(weights_a/weights_b))
                    
                    saliency_maps.append(saliency_map)
                    
                    continue
                '''
                    
                if not self.pos_neg_map:
                    # Only take the positive gradient values
                    saliency_map        = F.relu(saliency_map)
                    saliency_map        = self.getNorm(self.getSmap(saliency_map)).view(b, u, v)
                else:
                    smap1               = F.relu(saliency_map)
                    smap2               = F.relu(saliency_map * -1)
    
                    smap1               = self.getSmap(smap1)
                    smap2               = self.getSmap(smap2) * -1
                    
                    saliency_map        = smap1 + smap2
                    saliency_map        = self.getNorm(saliency_map).view(b, u, v)
                    
                saliency_maps.append(saliency_map)
            
        return saliency_maps, logit  


