from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import maps

class ConditionalSaliencyMaps(maps.CombineSaliencyMaps): 
    r'''
        This will combine saliency maps into a single weighted saliency map. 
        
        Input is a list of 3D tensors or various sizes. 
        Output is a 3D tensor of size output_size
        
        num_maps specifies how many maps we will combine
        weights is an optional list of weights for each layer e.g. [1, 2, 3, 4, 5]
    '''
    
    def __init__(self, **kwargs):
        
        super(ConditionalSaliencyMaps, self).__init__(**kwargs)
        
    def forward(self, xmap, ymaps, reverse=False):
        
        r'''
            Input shapes are something like [64,7,7] i.e. [batch size x layer_height x layer_width]
            Output shape is something like [64,224,244] i.e. [batch size x image_height x image_width]
        '''

        assert(isinstance(xmap,list))
        assert(len(xmap) == self.map_num)
        assert(len(xmap[0].size()) == 3)   
                
        bn  = xmap[0].size()[0]
        cm  = torch.zeros((bn, 1, self.output_size[0], self.output_size[1]), dtype=xmap[0].dtype, device=xmap[0].device)
        ww  = []
        
        r'''
            Now get each saliency map and resize it. Then store it and also create a combined saliency map.
        '''
        for i in range(len(xmap)):
            assert(torch.is_tensor(xmap[i]))
            wsz = xmap[i].size()
            wx  = xmap[i].reshape(wsz[0], 1, wsz[1], wsz[2]) + 0.0000001
            w   = torch.zeros_like(wx)
            
            if reverse:
                for j in range(len(ymaps)):
                    wy = ymaps[j][i].reshape(wsz[0], 1, wsz[1], wsz[2]) + 0.0000001
                
                    w -= wx*torch.log2(wx/wy)
            else:
                for j in range(len(ymaps)):
                    wy = ymaps[j][i].reshape(wsz[0], 1, wsz[1], wsz[2]) + 0.0000001
                
                    w -= wy*torch.log2(wy/wx)
                        
            w   = torch.clamp(w,0.0000001,1)
            w   = nn.functional.interpolate(w, size=self.output_size, mode=self.resize_mode, align_corners=False) 
            
            ww.append(w)
            cm  += (w * self.weights[i])
                        
        cm  = cm / self.weight_sum
        cm  = cm.reshape(bn, self.output_size[0], self.output_size[1])
        
        ww  = torch.stack(ww,dim=1)
        ww  = ww.reshape(bn, self.map_num, self.output_size[0], self.output_size[1])
        
        return cm, ww 
    
    