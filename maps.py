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

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import norm
import misc
import resnet
import math

# *******************************************************************************************************************
class SMOEScaleMap(nn.Module):
    r'''
        Compute SMOE Scale on a 4D tensor. This acts as a standard PyTorch layer. 
        
        SMOE Scale is computed independantly for each batch item at each location x,y
    
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

        assert torch.is_tensor(x), "input must be a Torch Tensor"
        assert len(x.size()) > 2, "input must have at least three dims"
        
        
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
        
        Standard Deviation is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) Recommend a tensor with only positive values. (After a ReLU)
            Any real value will work. 
        
        Output is a 3D tensor of size [batch x height x width]
    '''
    def __init__(self):
        
        super(StdMap, self).__init__()
        
    def forward(self, x):
        
        assert torch.is_tensor(x), "input must be a Torch Tensor"
        assert len(x.size()) > 2, "input must have at least three dims"
        
        x = torch.std(x,dim=1)
        
        return x

# *******************************************************************************************************************
class MeanMap(nn.Module):
    r'''
        Compute vanilla mean on a 4D tensor. This acts as a standard PyTorch layer. 
        
        The Mean is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) Recommend a tensor with only positive values. (After a ReLU)
            Any real value will work. 
        
        Output is a 3D tensor of size [batch x height x width]
    '''
    def __init__(self):
        
        super(MeanMap, self).__init__()
        
    def forward(self, x):
        
        assert torch.is_tensor(x), "input must be a Torch Tensor"
        assert len(x.size()) > 2, "input must have at least three dims"
        
        x = torch.mean(x,dim=1)
        
        return x
    
# *******************************************************************************************************************
class MaxMap(nn.Module):
    r'''
        Compute vanilla mean on a 4D tensor. This acts as a standard PyTorch layer. 
        
        The Max is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) Recommend a tensor with only positive values. (After a ReLU)
            Any real value will work. 
        
        Output is a 3D tensor of size [batch x height x width]
    '''
    def __init__(self):
        
        super(MaxMap, self).__init__()
        
    def forward(self, x):
        
        assert torch.is_tensor(x), "input must be a Torch Tensor"
        assert len(x.size()) > 2, "input must have at least three dims"
        
        x = torch.max(x,dim=1)[0]
        
        return x   
    
# *******************************************************************************************************************
class TruncNormalEntMap(nn.Module):
    r'''
        Compute truncated normal entropy on a 4D tensor. This acts as a standard PyTorch layer. 
        
        Truncated Normal Entropy is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) This should come BEFORE a ReLU and can range over any real value. 
            Ideally it should have both positive and negative values. 
        
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
        
        assert torch.is_tensor(x), "input must be a Torch Tensor"
        assert len(x.size()) > 2, "input must have at least three dims"
 
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
class GammaScaleMap(nn.Module):
    r'''
        Compute Gamma Scale on a 4D tensor (The hard way). This acts as a standard PyTorch layer. 
        
        Gamma Scale is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) A tensor with only positive values. (After a ReLU)
        
        Output is a 3D tensor of size [batch x height x width] 
    '''
    def __init__(self, run_relu=False):
        
        super(GammaScaleMap, self).__init__()
        
        r'''
            SMOE Scale must take in values > 0. Optionally, we can run a ReLU to do that.
        '''
        if run_relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = None
    
    def _trigamma(self, x):
    
        r'''
            We need this line since recursion is not good for x < 1.0
            Note that we take + torch.reciprocal(x.pow(2)) at the end because:
        
                trigamma(z) = trigamma(z + 1) + 1/z^2
        
        '''
        z   = x + 1.0
        
        zz  = z.pow(2)
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
        
    def _compute_k_est(self, x, i=10, dim=1):
                
        r'''
            Calculate s
        '''
        s  = torch.log(torch.mean(x,dim=dim)) - torch.mean(torch.log(x),dim=dim)
        
        r'''
            Get estimate of k to within 1.5%
        
            NOTE: K gets smaller as log variance s increases
        '''
        s3 = s - 3.0
        rt = torch.sqrt(s3.pow(2) + 24.0 * s)
        nm = 3.0 - s + rt
        dn = 12.0 * s
        k  = nm / dn + 0.0000001

        r'''
            Do i Newton-Raphson steps to get closer than 1.5%
            For i=5 gets us within 4 or 5 decimal places
        '''
        for _ in range(i):
            k =  self._k_update(k,s)
        
        return k
                 
    def forward(self, x):

        assert torch.is_tensor(x), "input must be a Torch Tensor"
        assert len(x.size()) > 2, "input must have at least three dims"
        
        r'''
            If we do not have a convenient ReLU to pluck from, we can do it here
        '''
        if self.relu is not None:
            x = self.relu(x)
                               
        r'''
            avoid log(0)
        '''
        x   = x + 0.0000001
        
        k   = self._compute_k_est(x)
                
        th  = torch.reciprocal(k) * torch.mean(x,dim=1)
                
        return th
   
# *******************************************************************************************************************     
# *******************************************************************************************************************
class CombineSaliencyMaps(nn.Module): 
    r'''
        This will combine saliency maps into a single weighted saliency map. 
        
        Input will be a list of 3D tensors or various sizes. 
        Output is a 3D tensor of size batch size x output_size. We also return the individual saliency maps resized. to output_size
        
        Parameters:
        
            output_size:    A list that contains the height and width of the output saliency maps.
            num_maps:       Specifies how many maps we will combine.
            weights:        Is an optional list of weights for each layer e.g. [1, 2, 3, 4, 5].
            resize_mode:    Is given to Torch nn.functional.interpolate. Whatever it supports will work here. 
            do_relu:        Should we do a final clamp on values to set all negative values to 0?
            
        Will Return:
        
            cm:     The combined saliency map over all layers sized batch size x output_size
            ww:     Each individual saliency maps sized output_size. Note that we do not weight these outputs. 
    '''
    
    def __init__(self, output_size=[224,224], map_num=5, weights=None, resize_mode='bilinear', do_relu=False):
        
        super(CombineSaliencyMaps, self).__init__()
        
        assert isinstance(output_size,list), "Output size should be a list (e.g. [224,224])."
        assert isinstance(map_num,int), "Number of maps should be a positive integer > 0"
        assert isinstance(resize_mode,str), "Resize mode is a string recognized by Torch nn.functional.interpolate (e.g. 'bilinear')."    
        assert len(output_size) == 2, "Output size should be a list (e.g. [224,224])."
        assert output_size[0] > 0, "Output size should be a list (e.g. [224,224])."
        assert output_size[1] > 0, "Output size should be a list (e.g. [224,224])."
        assert map_num > 0, "Number of maps should be a positive integer > 0"
        
        r'''
            We support weights being None, a scaler or a list. 
            
            Depending on which one, we create a list or just point to one.
        '''
        if weights is None:
            self.weights = [1.0 for _ in range(map_num)]
        elif len(weights) == 1:
            assert weights > 0
            self.weights = [weights for _ in range(map_num)]   
        else:
            assert len(weights) == map_num        
            self.weights = weights
        
        self.weight_sum = 0
        
        for w in self.weights:
            self.weight_sum += w  
        
        self.map_num        = map_num
        self.output_size    = output_size
        self.resize_mode    = resize_mode
        self.do_relu        = do_relu
        
    def forward(self, smaps):
        
        r'''
            Input shapes are something like [64,7,7] i.e. [batch size x layer_height x layer_width]
            Output shape is something like [64,224,244] i.e. [batch size x image_height x image_width]
        '''

        assert isinstance(smaps,list), "Saliency maps must be in a list"
        assert len(smaps) == self.map_num, "List length is not the same as predefined length"
        assert len(smaps[0].size()) == 3, "Each saliency map must be 3D, [batch size x layer_height x layer_width]"
        
        bn  = smaps[0].size()[0]
        cm  = torch.zeros((bn, 1, self.output_size[0], self.output_size[1]), dtype=smaps[0].dtype, device=smaps[0].device)
        ww  = []
        
        r'''
            Now get each saliency map and resize it. Then store it and also create a combined saliency map.
        '''
        for i in range(len(smaps)):
            assert torch.is_tensor(smaps[i]), "Each saliency map must be a Torch Tensor."
            wsz = smaps[i].size()
            w   = smaps[i].reshape(wsz[0], 1, wsz[1], wsz[2])
            w   = nn.functional.interpolate(w, size=self.output_size, mode=self.resize_mode, align_corners=False) 
            ww.append(w)        # should we weight the raw maps ... hmmm
            cm  += (w * self.weights[i])

        r'''
            Finish the combined saliency map to make it a weighted average.
        '''
        cm  = cm / self.weight_sum
        cm  = cm.reshape(bn, self.output_size[0], self.output_size[1])
        
        ww  = torch.stack(ww,dim=1)
        ww  = ww.reshape(bn, self.map_num, self.output_size[0], self.output_size[1])
        
        if self.do_relu:
            cm = F.relu(cm)
            ww = F.relu(ww)
        
        return cm, ww 

# *******************************************************************************************************************         
class SaliencyMap(object):
    r'''
        Given an input model and parameters, run the neural network and compute saliency maps for given images.
        
        input:             input image with shape of (batch size, 3, H, W)
        
        Parameters:
        
            model:          This should be a valid Torch neural network such as a ResNet.
            layers:         A list of layers you wish to process given by name. If none, we can auto compute a selection.
            maps_method:    How do we compute saliency for each activation map? Default: SMOEScaleMap
            norm_method:    How do we post process normalize each saliency map? Default: norm.GaussNorm2D 
                            This can also be norm.GammaNorm2D or norm.RangeNorm2D. 
            output_size:    This is the standard 2D size for the saliency maps. Torch nn.functional.interpolate
                            will be used to make each saliency map this size. Default [224,224]
            weights:        The weight for each layer in the combined saliency map's weighted average.
                            It should either be a list of floats or None.
            resize_mode:    Is given to Torch nn.functional.interpolate. Whatever it supports will work here. 
            do_relu:        Should we do a final clamp on values to set all negative values to 0?
            
        Will Return:
        
            combined_map:   [Batch x output height x output width] set of 2D saliency maps combined from each layer 
                            we compute from and combined with a CAM if we computed one. 
            saliency_maps:  A list [number of layers size] containing each saliency map [Batch x output height x output width].
                            These will have been resized from their orginal layer size.  
            logit:          The output neural network logits. 
        
    '''
    def __init__(self, model, layers, maps_method=SMOEScaleMap, norm_method=norm.GaussNorm2D,
                 output_size=[224,224], weights=None, resize_mode='bilinear', do_relu=False, cam_method='gradcam',
                 module_layer=None, expl_do_fast_cam=False, do_nonclass_map=False, cam_each_map=False):
                
        assert isinstance(layers, list) or layers is None, "Layers must be a list of layers or None"
        assert callable(maps_method), "Saliency map method must be a callable function or method."
        assert callable(norm_method), "Normalization method must be a callable function or method."
        
        self.get_smap           = maps_method()
        self.get_norm           = norm_method()
        self.model              = model
        
        r'''
            This gives us access to more complex network modules than a standard ResNet if we need it.
        '''
        self.module_layer       = model if module_layer is None else module_layer
        
        self.activation_hooks   = []
                
        r'''
            Optionally, we can either define the layers we want or we can 
            automatically pick all the ReLU layers.
        '''
        if layers is None:
            assert weights is None, "If we auto select layers, we should auto compute weights too."
            r'''
                Pick all the ReLU layers. Set weights to 1 since the number of ReLUs is proportional
                to how high up we are in the resulotion pyramid.
                
                For each we attach a hook to get the layer activations back after the 
                network runs the data.
                
                NOTE: This is quantitativly untested. There are no ROAR/KARR scores yet.  
            '''
            self.layers     = []
            weights         = []
            for m in self.module_layer.modules():
                if isinstance(m, nn.ReLU):          # Maybe allow a user defined layer (e.g. nn.Conv)
                    h   = misc.CaptureLayerOutput(post_process=None)
                    _   = m.register_forward_hook(h)
                    self.activation_hooks.append(h)
                    weights.append(1.0)             # Maybe replace with a weight function
                    self.layers.append(None)
        else:
            r'''
                User defined layers. 
                
                For each we attach a hook to get the layer activations back after the 
                network runs the data.
            '''
            self.layers = layers
            for i,l in enumerate(layers):
                h   = misc.CaptureLayerOutput(post_process=None)
                _   = self.module_layer._modules[l].register_forward_hook(h)
                self.activation_hooks.append(h)
        
        r'''     
            This object will be used to combine all the saliency maps together after we compute them.
        ''' 
        self.combine_maps = CombineSaliencyMaps(output_size=output_size, map_num=len(weights), weights=weights, 
                                                resize_mode=resize_mode, do_relu=do_relu)
        
        r'''
            Are we also computing the CAM map?
        '''
        if isinstance(model,resnet.ResNet_FastCAM) or expl_do_fast_cam:
            self.do_fast_cam        = True
            self.do_nonclass_map    = do_nonclass_map
            self.cam_method         = cam_method
            self.cam_each_map       = cam_each_map
        else:
            self.do_fast_cam        = False
            self.do_nonclass_map    = None
            self.cam_method         = None
            self.cam_each_map       = None
    
    def __call__(self, input, grad_enabled=False):
        """
        Args:
            input:         input image with shape of (B, 3, H, W)
            grad_enabled:  Set this to true if you need to compute grads when running the network. For instance, while training.    
            
        Return:
            combined_map:   [Batch x output height x output width] set of 2D saliency maps combined from each layer 
                            we compute from and combined with a CAM if we computed one. 
            saliency_maps:  A list [number of layers size] containing each saliency map [Batch x output height x output width].
                            These will have been resized from their orginal layer size.  
            logit:          The output neural network logits. 
        """

        r'''
            Don't compute grads if we do not need them. Cuts network compute time way down.
        '''
        with torch.set_grad_enabled(grad_enabled):

            r'''
                Get the size, but we support lists here for certain special cases.
            '''
            b, c, h, w      = input[0].size() if isinstance(input,list) else input.size()
            
            
            self.model.eval()
            
            if self.do_fast_cam:
                logit,cam_map   = self.model(input,method=self.cam_method)
            else:
                logit           = self.model(input)
            
            saliency_maps   = []
            
            r'''
                Get the activation for each layer in our list. Then compute saliency and normalize.
            '''
            for i,l in enumerate(self.layers):
            
                activations         = self.activation_hooks[i].data
                b, k, u, v          = activations.size()
                activations         = F.relu(activations)
                saliency_map        = self.get_norm(self.get_smap(activations)).view(b, u, v)
                                    
                saliency_maps.append(saliency_map)
        
        r'''
            Combine each saliency map together into a single 2D saliency map.
        '''
        combined_map, saliency_maps = self.combine_maps(saliency_maps)
        
        r'''
            If we computed a CAM, combine it with the forward only saliency map.
        '''
        if self.do_fast_cam:
            if self.do_nonclass_map:
                combined_map = combined_map*(1.0 - cam_map)
                if self.cam_each_map:
                    saliency_maps = saliency_maps.squeeze(0)
                    saliency_maps = saliency_maps*(1.0 - cam_map)
                    saliency_maps = saliency_maps.unsqueeze(0)
            else:                
                combined_map = combined_map * cam_map
                
                if self.cam_each_map:
                    saliency_maps = saliency_maps.squeeze(0)
                    saliency_maps = saliency_maps*cam_map
                    saliency_maps = saliency_maps.unsqueeze(0)
            
            
        return combined_map, saliency_maps, logit      

# *******************************************************************************************************************     
# *******************************************************************************************************************         
class SaliencyVector(SaliencyMap):
    r'''
        Given an input model and parameters, run the neural network and compute saliency maps for given images.
        
        input:             input image with shape of (batch size, 3, H, W)
        
        Parameters:
        
            model:          This should be a valid Torch neural network such as a ResNet.
            layers:         A list of layers you wish to process given by name. If none, we can auto compute a selection.
            maps_method:    How do we compute saliency for each activation map? Default: SMOEScaleMap
            norm_method:    How do we post process normalize each saliency map? Default: norm.GaussNorm2D 
                            This can also be norm.GammaNorm2D or norm.RangeNorm2D. 
            output_size:    This is the standard 2D size for the saliency maps. Torch nn.functional.interpolate
                            will be used to make each saliency map this size. Default [224,224]
            weights:        The weight for each layer in the combined saliency map's weighted average.
                            It should either be a list of floats or None.
            resize_mode:    Is given to Torch nn.functional.interpolate. Whatever it supports will work here. 
            do_relu:        Should we do a final clamp on values to set all negative values to 0?
            
        Will Return:
        
            combined_map:   [Batch x output height x output width] set of 2D saliency maps combined from each layer 
                            we compute from and combined with a CAM if we computed one. 
            saliency_maps:  A list [number of layers size] containing each saliency map [Batch x output height x output width].
                            These will have been resized from their orginal layer size.  
            logit:          The output neural network logits. 
            sal_location:   A tuple of x,y locations which are the most salienct in each image.
            feature_vecs:   List of salient feature vectors. Each list item is assocaited with each layer in the layers argument.
        
    '''
    def __init__(self, model, layers, **kwargs):
                
        super(SaliencyVector, self).__init__(model, layers, **kwargs)
    
    def __call__(self, input, grad_enabled=False):
        
        """
        Args:
            input:         input image with shape of (B, 3, H, W)
            grad_enabled:  Set this to true if you need to compute grads when running the network. For instance, while training.    
            
        Return:
            combined_map:   [Batch x output height x output width] set of 2D saliency maps combined from each layer 
                            we compute from and combined with a CAM if we computed one. 
            saliency_maps:  A list [number of layers size] containing each saliency map [Batch x output height x output width].
                            These will have been resized from their orginal layer size.  
            logit:          The output neural network logits. 
            sal_location:   A tuple of x,y locations which are the most salienct in each image.
        feature_vecs:   List of salient feature vectors. Each list item is assocaited with each layer in the layers argument.asssssssssssssssssssssssssssssssssssssQQQQQQQQQQQQQQQQQQ
        """

        r'''
            Call the base __call__  method from the base class first to get saliency maps.
        '''
        combined_map, saliency_maps, logit  = super(SaliencyVector, self).__call__(input, grad_enabled)
        
        sz              = combined_map.size()
        
        combined_map    = combined_map.reshape(sz[0],sz[1]*sz[2])
        
        r'''
            Get the location x,y expressed as one vector. 
        '''
        sal_loc         = torch.argmax(combined_map,dim=1)
        
        r'''
            Get the actual location by offseting the y place size.
        '''
        sal_y           = sal_loc//sz[1]
        sal_x           = sal_loc%sz[1]
        
        r'''
            Get each activation layer again from the layer hooks. 
        '''
        feature_vecs = []
        for i,l in enumerate(self.layers):
                    
            activations       = self.activation_hooks[i].data
            b, k, v, u        = activations.size()              # Note: v->y and u->x
            
            r'''
                Compute new x,y location based on the layers size.
            '''
            loc_x = math.floor((v/sz[2])*float(sal_x))    
            loc_y = math.floor((u/sz[1])*float(sal_y))   
            loc   = loc_y*u + loc_x
            
            r'''
                Get feature vectors k at location loc from all batches b.
            '''
            feature_vecs.append(activations.permute(0,2,3,1).reshape(b,v*u,k)[:,loc,:])
        
        combined_map    = combined_map.reshape(sz[0],sz[1],sz[2])
        sal_location    = (sal_x,sal_y)
        
        return combined_map, saliency_maps, logit, sal_location, feature_vecs       
      
# *******************************************************************************************************************         
# *******************************************************************************************************************         
class SaliencyModel(nn.Module):
    r'''
        Given an input model and parameters, run the neural network and compute saliency maps for given images.
        
        This version will run as a regular batch on a mutli-GPU machine. It will eventually replace SaliencyMap. 
        
        input:             input image with shape of (batch size, 3, H, W)
        
        Parameters:
        
            model:          This should be a valid Torch neural network such as a ResNet.
            layers:         A list of layers you wish to process given by name. If none, we can auto compute a selection.
            maps_method:    How do we compute saliency for each activation map? Default: SMOEScaleMap
            norm_method:    How do we post process normalize each saliency map? Default: norm.GaussNorm2D 
                            This can also be norm.GammaNorm2D or norm.RangeNorm2D. 
            output_size:    This is the standard 2D size for the saliency maps. Torch nn.functional.interpolate
                            will be used to make each saliency map this size. Default [224,224]
            resize_mode:    Is given to Torch nn.functional.interpolate. Whatever it supports will work here. 
            do_relu:        Should we do a final clamp on values to set all negative values to 0?
            
        Will Return:
        
            combined_map:   [Batch x output height x output width] set of 2D saliency maps combined from each layer 
                            we compute from and combined with a CAM if we computed one. 
            saliency_maps:  A list [number of layers size] containing each saliency map [Batch x output height x output width].
                            These will have been resized from their orginal layer size.  
            logit:          The output neural network logits. 
        
    '''
    def __init__(self, model, layers=None, maps_method=SMOEScaleMap, norm_method=norm.GammaNorm2D,
                 output_size=[224,224], weights=None, auto_layer=nn.ReLU, resize_mode='bilinear', 
                 do_relu=False, cam_method='gradcam', module_layer=None, expl_do_fast_cam=False, 
                 do_nonclass_map=False, cam_each_map=False):
                
        assert isinstance(model, nn.Module), "model must be a valid PyTorch module"
        assert isinstance(layers, list) or layers is None, "Layers must be a list of layers or None"        
        assert callable(maps_method), "Saliency map method must be a callable function or method."
        assert callable(norm_method), "Normalization method must be a callable function or method."
        assert isinstance(auto_layer(), nn.Module), "Auto layer if used must be a type for nn.Module such as nn.ReLU."
            
        super(SaliencyModel, self).__init__()
        
        self.get_smap           = maps_method()
        self.get_norm           = norm_method()
        self.model              = model
        self.layers             = layers
        self.auto_layer         = auto_layer
        
        r'''
            If we are auto selecting layers, count how many we have and create an empty layer list of the right size.
            Later, this will make us compatible with enumerate(self.layers)
        '''
        if self.layers is None:
            self.auto_layers    = True
            map_num             = 0
            weights             = None
            self.layers         = []
            for m in self.model.modules():
                if isinstance(m, self.auto_layer):          
                    map_num += 1
                    self.layers.append(None)
        else:
            map_num             = len(self.layers)
            self.auto_layers    = False
                    
        r'''     
            This object will be used to combine all the saliency maps together after we compute them.
        ''' 
        self.combine_maps       = CombineSaliencyMaps(output_size=output_size, map_num=map_num, weights=None, 
                                                      resize_mode=resize_mode, do_relu=do_relu)
        
        r'''
            Are we also computing the CAM map?
        '''
        if isinstance(model, resnet.ResNet_FastCAM) or expl_do_fast_cam:
            self.do_fast_cam        = True
            self.do_nonclass_map    = do_nonclass_map
            self.cam_method         = cam_method
            self.cam_each_map       = cam_each_map
        else:
            self.do_fast_cam        = False
            self.do_nonclass_map    = None
            self.cam_method         = None
            self.cam_each_map       = None
    
    def __call__(self, input, grad_enabled=False):
        """
        Args:
            input:         input image with shape of (B, 3, H, W)
            grad_enabled:  Set this to true if you need to compute grads when running the network. For instance, while training.    
            
        Return:
            combined_map:   [Batch x output height x output width] set of 2D saliency maps combined from each layer 
                            we compute from and combined with a CAM if we computed one. 
            saliency_maps:  A list [number of layers size] containing each saliency map [Batch x output height x output width].
                            These will have been resized from their orginal layer size.  
            logit:          The output neural network logits. 
        """
        
        r'''
            We set up the hooks each iteration. This is needed when running in a multi GPU version where this module is split out
            post __init__. 
        '''
        self.activation_hooks       = []

        if self.auto_layers:
            r'''
                Auto defined layers. Here we will process all layers of a certain type as defined by the use.
                This might commonly be all ReLUs or all Conv layers.
            '''
            for m in self.model.modules():
                if isinstance(m, self.auto_layer):      # Maybe allow a user defined layer (e.g. nn.Conv)
                    m._forward_hooks   = OrderedDict()  # PyTorch bug work around, patch is avialable, but not everyone may be patched
                    h   = misc.CaptureLayerOutput(post_process=None, device=input.device)
                    _   = m.register_forward_hook(h)
                    self.activation_hooks.append(h)
                    
        else:
            r'''
                User defined layers. 
                
                For each we attach a hook to get the layer activations back after the 
                network runs the data.
            '''
            for i,l in enumerate(self.layers):
                self.model._modules[l]._forward_hooks   = OrderedDict()  # PyTorch bug work around, patch is aviable, but not everyone may be patched
                h   = misc.CaptureLayerOutput(post_process=None, device=input.device)
                _   = self.model._modules[l].register_forward_hook(h)
                self.activation_hooks.append(h)
                    
        r'''
            Don't compute grads if we do not need them. Cuts network compute time way down.
        '''
        with torch.set_grad_enabled(grad_enabled):

            r'''
                Get the size, but we support lists here for certain special cases.
            '''
            b, c, h, w      = input[0].size() if isinstance(input,list) else input.size()
            
            self.model.eval()
            
            if self.do_fast_cam:
                logit,cam_map   = self.model(input, method=self.cam_method)
            else:
                logit           = self.model(input)
            
            saliency_maps   = []
            
            r'''
                Get the activation for each layer in our list. Then compute saliency and normalize.
            '''
            for i,l in enumerate(self.layers):
            
                activations         = self.activation_hooks[i].data
                b, k, u, v          = activations.size()
                activations         = F.relu(activations)
                saliency_map        = self.get_norm(self.get_smap(activations)).view(b, u, v)
                                    
                saliency_maps.append(saliency_map)
        
        r'''
            Combine each saliency map together into a single 2D saliency map. This is outside the 
            set_grad_enabled loop since it might need grads if doing FastCAM.  
        '''
        combined_map, saliency_maps = self.combine_maps(saliency_maps)
        
        r'''
            If we computed a CAM, combine it with the forward only saliency map.
        '''
        if self.do_fast_cam:
            if self.do_nonclass_map:
                combined_map = combined_map*(1.0 - cam_map)
                if self.cam_each_map:
                    saliency_maps = saliency_maps.squeeze(0)
                    saliency_maps = saliency_maps*(1.0 - cam_map)
                    saliency_maps = saliency_maps.unsqueeze(0)
            else:                
                combined_map = combined_map * cam_map
                
                if self.cam_each_map:
                    saliency_maps = saliency_maps.squeeze(0)
                    saliency_maps = saliency_maps*cam_map
                    saliency_maps = saliency_maps.unsqueeze(0)
            
            
        return combined_map, saliency_maps, logit