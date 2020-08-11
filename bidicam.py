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

from collections import OrderedDict
import math
import numpy as np

try:
    from . import maps
    from . import misc
    from . import norm
    from . import misc
    from . import resnet
except ImportError:
    import maps
    import misc
    import norm
    import misc
    import resnet

# *******************************************************************************************************************       
class ScoreMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scores):
        
        ctx.save_for_backward(scores)
        return torch.tensor(1.0)
    
    @staticmethod
    def backward(ctx, grad):
                
        saved        = ctx.saved_tensors
        g_scores     = torch.ones_like(saved[0])
        
        return g_scores

# *******************************************************************************************************************         
class BiDiCAM(nn.Module):
    r"""
    Bi Di Bi Di Bi Di Nice shootin' Buck!
    """
    def __init__(self, model, layers=None, actv_method=maps.SMOEScaleMap, grad_method=maps.SMOEScaleMap, grad_pooling='mag', interp_pooling='nearest',
                 use_GradCAM=False, do_first_forward=False, num_classes=1000):
        
        super(BiDiCAM, self).__init__()
        
        assert isinstance(layers, list) or layers is None 
        assert isinstance(grad_pooling, str) or grad_pooling is None 
        assert callable(actv_method)
        assert callable(grad_method) 
        #assert not(use_GradCAM and do_first_forward)
        
        
        self.getActvSmap        = actv_method()
        self.getGradSmap        = grad_method()
        self.layers             = layers
        self.model              = model
        self.grad_pooling       = grad_pooling
        self.num_classes        = num_classes
        self.use_GradCAM        = use_GradCAM
        self.interp_pooling     = interp_pooling
        self.do_first_forward   = do_first_forward
        self.auto_layer         = 'BatchNorm2d'
        self.getNorm            = norm.GaussNorm2D() #norm.GammaNorm2D()
        
        if layers is None:
            self.auto_layers    = True
        else:
            self.auto_layers    = False
        
        if self.auto_layers:
            self.layers = []
            
            for m in self.model.modules():
                if self.auto_layer in str(type(m)):
                    self.layers.append(None)
        
    def _forward(self, class_idx, logit, retain_graph):
        
        if class_idx is None:
    
            sz      = logit.size()
            
            lm      = logit.max(1)[1]  
            r'''
                This gets the logits into a form usable when we run a batch. This seems suboptimal.
                Open to ideas about how to make this better/faster.
            '''
            lm      = torch.stack([i*self.num_classes + v for i,v in enumerate(lm)])
                
            logit   = logit.reshape(sz[0]*sz[1])
            
            score   = logit[lm]
                
            logit   = logit.reshape(sz[0],sz[1])    
            score   = score.reshape(sz[0],1,1,1)
        else:
            score   = logit[:, class_idx].squeeze()
            
        r'''
            Pass through layer to make auto grad happy
        '''
        score_end   = ScoreMap.apply(score)

        r'''
            Zero out grads and then run backwards. 
        '''
        self.model.zero_grad()
        score_end.backward(retain_graph=retain_graph)
            
    def _magnitude_pool2d(self, x, kernel_size=2, stride=2, padding=0, pos_max=False):
        r'''
            Pick the max magnitude gradient in the pool, the one with the highest absolute value. 
            
            This is an optional method. 
        '''

        b, k, u, v          = x.size()

        p1  = F.max_pool2d(x,    kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=True)
        p2  = F.max_pool2d(-1*x, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=True) * -1

        d   = p1 + p2

        if pos_max:
            m   = torch.where(d >= 0.0, p1, torch.zeros_like(d))
        else:
            m   = torch.where(d >= 0.0, p1, p2)

        m   = nn.functional.interpolate(m, size=[u,v], mode=self.interp_pooling)        

        return m 
    
    def _proc_salmap(self, saliency_map, map_method, b, u, v):
        r'''
            Derive the saliency map from the input layer and then normalize it.
        '''
        
        saliency_map            = F.relu(saliency_map)
        saliency_map            = self.getNorm(map_method(saliency_map)).view(b, u, v)
        
        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating CAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        
        self.activation_hooks   = []
        self.gradient_hooks     = []
        
        if self.auto_layers:
            r'''
                Auto defined layers. Here we will process all layers of a certain type as defined by the use.
                This might commonly be all ReLUs or all Conv layers.
            '''
            for i,m in enumerate(self.model.modules()): 
                if self.auto_layer in str(type(m)):    
                    m._forward_hooks    = OrderedDict()  # PyTorch bug work around, patch is available, but not everyone may be patched
                    m._backward_hooks   = OrderedDict()
                    
                    if self.do_first_forward and len(self.activation_hooks) > 0 and not self.use_GradCAM:
                        pass
                    else:
                        h   = misc.CaptureLayerOutput(post_process=None, device=input.device)
                        _   = m.register_forward_hook(h)
                        self.activation_hooks.append(h)
                    
                    h   = misc.CaptureGradInput(post_process=None, device=input.device) # The gradient information leaving the layer
                    _   = m.register_backward_hook(h)
                    self.gradient_hooks.append(h) 
        else:            
            for i,l in enumerate(self.layers):
                self.model._modules[l]._forward_hooks    = OrderedDict()
                self.model._modules[l]._backward_hooks   = OrderedDict()
                
                if self.do_first_forward and i>0 and not self.use_GradCAM:
                    pass
                else:
                    h   = misc.CaptureLayerOutput(post_process=None, device=input.device)
                    _   = self.model._modules[l].register_forward_hook(h)
                    self.activation_hooks.append(h)

                h   = misc.CaptureGradInput(post_process=None, device=input.device) # The gradient information leaving the layer
                _   = self.model._modules[l].register_backward_hook(h)
                self.gradient_hooks.append(h)

        r'''
            Force to compute grads since we always need them. 
        '''
        with torch.set_grad_enabled(True):

            b, c, h, w      = input.size()
    
            self.model.eval()
    
            logit           = self.model(input)
            
            self._forward(class_idx, logit, retain_graph)

            backward_saliency_maps  = []
            forward_saliency_maps   = []
            
            r'''
                For each layer, get its activation and gradients. We might pool the gradient layers.
                
                Finally, processes the activations and return.
            '''
            for i,l in enumerate(self.layers):
            
                gradients           = self.gradient_hooks[i].data
                gb, gk, gu, gv      = gradients.size()
                
                if self.do_first_forward and i>0 and not self.use_GradCAM:
                    pass
                else:
                    activations         = self.activation_hooks[i].data
                    ab, ak, au, av      = activations.size()
                
                if self.use_GradCAM:
                    alpha               = gradients.view(gb, gk, -1).mean(2)
                    weights             = alpha.view(gb, gk, 1, 1)
                    cam_map             = self.getNorm((weights*activations).sum(1, keepdim=True).squeeze(0))
                
                if self.grad_pooling == 'avg':
                    gradients           = F.avg_pool2d(gradients, kernel_size=2, stride=2, padding=0, ceil_mode=True)
                    gradients           = nn.functional.interpolate(gradients, size=[gu,gv], mode=self.interp_pooling) 
                elif self.grad_pooling == 'max':
                    gradients           = F.max_pool2d(gradients, kernel_size=2, stride=2, padding=0, ceil_mode=True)
                    gradients           = nn.functional.interpolate(gradients, size=[gu,gv], mode=self.interp_pooling) 
                elif self.grad_pooling == 'mag':
                    gradients           = self._magnitude_pool2d(gradients, kernel_size=2, stride=2, padding=0)
                elif self.grad_pooling is None:
                    gradients           = gradients
                                
                r'''
                    Optionally, we can meld with GradCAM Method.
                '''
                if self.use_GradCAM:
                    l                   = float(len(self.layers)) - float(i)
                    n                   = math.log2(l)
                    d                   = math.log2(float(len(self.layers)))  
                    ratio               = 1.0 - n/d         
                    grad_map            = self._proc_salmap(gradients, self.getGradSmap, gb, gu, gv)
                    gradients           = ratio*cam_map + (1.0 - ratio)*grad_map
                else:
                    gradients           = self._proc_salmap(gradients, self.getGradSmap, gb, gu, gv)
                
                backward_saliency_maps.append(gradients)
                
                if self.do_first_forward and i>0:
                    pass
                else:
                    forward_saliency_maps.append(self._proc_salmap(activations, self.getActvSmap, ab, au, av))
                
        return forward_saliency_maps, backward_saliency_maps, logit  
    
 # *******************************************************************************************************************         
class BiDiCAMModel(nn.Module):
    r"""
    Bi Di Bi Di Bi Di Nice shootin' Buck!
    """
    def __init__(self, model, layers, output_size=[224,224], weights=None, resize_mode='bilinear', do_relu=False, do_first_forward=False, **kwargs):
        
        super(BiDiCAMModel, self).__init__()   
        
        self.do_first_forward   = do_first_forward
        
        self.bidicam            = BiDiCAM(model, layers, do_first_forward=do_first_forward, **kwargs)
        
        if self.do_first_forward:
            self.combine_maps_act       = maps.CombineSaliencyMaps(output_size=output_size, map_num=1, weights=weights, 
                                                               resize_mode=resize_mode, do_relu=do_relu)
        else:
            self.combine_maps_act       = maps.CombineSaliencyMaps(output_size=output_size, map_num=len(self.bidicam.layers), weights=weights, 
                                                               resize_mode=resize_mode, do_relu=do_relu)
            
        self.combine_maps_grad  = maps.CombineSaliencyMaps(output_size=output_size, map_num=len(self.bidicam.layers), weights=weights, 
                                                           resize_mode=resize_mode, do_relu=do_relu)

        
    
    def __call__(self, input, **kwargs):
        
            
        forward_saliency_maps, backward_saliency_maps, logit    = self.bidicam(input, **kwargs)
        
        with torch.set_grad_enabled(False):

            forward_combined_map, _     = self.combine_maps_act(forward_saliency_maps)    
            backward_combined_map, _    = self.combine_maps_grad(backward_saliency_maps)
            
            backward_combined_map       = misc.RangeNormalize(backward_combined_map)
            
            combined_map                = forward_combined_map*backward_combined_map
                    
            saliency_maps               = torch.ones_like(combined_map)
        
        return combined_map, saliency_maps, logit
    
    
    