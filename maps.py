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
class MeanMap(nn.Module):
    r'''
        Compute vanilla standard deviation on a 4D tensor. This acts as a standard PyTorch layer. 
    
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
    def __init__(self, layers, model, method='V1', maps_method=maps.SMOEScaleMap):
        
        assert(isinstance(layers, list))
        assert(isinstance(method,str))
        assert(callable(maps_method))
        
        self.getSmap    = maps_method()
        self.getNorm    = maps.Normalize2D()
        self.layers     = layers
        self.model      = model
        self.method     = method
        self.forward    = True
        self.backward   = True
        
        self.activation_hooks   = []
        self.gradient_hooks     = []
        
        if self.method=='V0':
            self.backward = False
        if self.method=='V5':
            self.forward = False
            
        if self.forward:
            for i,l in enumerate(layers):
                h   = misc.CaptureLayerOutput(post_process=None)
                _   = self.model._modules[l].register_forward_hook(h)
                self.activation_hooks.append(h)

        if self.backward:    
            for i,l in enumerate(layers):
                h   = misc.CaptureGradOutput(post_process=None) # The gradient information entering the layer
                #h   = misc.CaptureGradInput(post_process=None) # The gradient information exiting the layer
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
                    
                if self.forward:
                    activations         = self.activation_hooks[i].data
                    b, k, u, v          = activations.size()
                
                if invert:
                    gradients *= -1
                
                if self.method=='V0':
                    # Without Gradients, original SMOE Scale
                    weights             = torch.ones_like(activations)
                elif self.method=='V1':
                    # V1 SIGN over layer means
                    alpha               = gradients.view(b, k, -1).mean(2)
                    weights             = alpha.view(b, k, 1, 1)
                    weights             = torch.sign(weights) 
                elif self.method=='V2':
                    # V2 Original method
                    alpha               = gradients.view(b, k, -1).mean(2)
                    weights             = alpha.view(b, k, 1, 1)
                elif self.method=='V3':
                    # V3 SIGN over all values
                    weights             = torch.sign(gradients) 
                elif self.method=='V4':
                    # V4 SIGN over all values
                    weights             = gradients # Just take the raw gradients
                elif self.method=='V5':
                    # V5 Gradients Only
                    weights             = gradients # Just take the raw gradients
                    activations         = torch.ones_like(gradients)
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
                    
                saliency_map        = weights*activations
                saliency_map        = F.relu(saliency_map)
                saliency_map        = self.getNorm(self.getSmap(saliency_map)).view(b, u, v)
    
                saliency_maps.append(saliency_map)
            
        return saliency_maps, logit  


# *******************************************************************************************************************                   
# ******************************************************************************************************************* 
class Normalize2D(nn.Module):
    r'''
        This will normalize a saliency map to range from 0 to 1 via normal cumulative distribution function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any real valued number (supported by hardware)
        Output will range from 0 to 1
    '''
    
    def __init__(self):
        
        super(Normalize2D, self).__init__()   
        
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
        
        m       = x.mean(dim=1)
        m       = m.reshape(m.size()[0],1)
        s       = x.std(dim=1)
        s       = s.reshape(s.size()[0],1)
        
        r'''
            The normal cumulative distribution function is used to squash the values from 0 to 1
        '''
        x       = 0.5*(1.0 + torch.erf((x-m)/(s*torch.sqrt(torch.tensor(2.0)))))
                
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
    
    def __init__(self, output_size=[224,224], map_num=5, weights=None, resize_mode='bilinear', magnitude=False):
        
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
        
        return cm, ww 
        
