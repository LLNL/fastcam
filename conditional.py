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
    
    