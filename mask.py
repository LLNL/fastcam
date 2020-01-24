from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

# ******************************************************************************************************************* 
class DropMap(torch.autograd.Function):
    r'''
        When we created this, the torch.gt function did not seem to propagate gradients. 
        It might now, but we have not checked. This autograd function provides that. 
    '''
    
    @staticmethod
    def forward(ctx, p_map, k):
    
        drop_map_byte   = torch.gt(p_map,k) 
        drop_map        = torch.as_tensor(drop_map_byte, dtype=p_map.dtype, device=p_map.device)
        ctx.save_for_backward(drop_map)
        return drop_map
    
    @staticmethod
    def backward(ctx, grad_output):
                
        drop_map        = ctx.saved_tensors
        g_pmap          = grad_output * drop_map[0]
        
        # Just return empty since we don't have use for this gradient.
        sz              = g_pmap.size()
        g_k             = torch.empty((sz[0],1), dtype=g_pmap.dtype, device=g_pmap.device)
        
        return g_pmap, g_k
    
# ******************************************************************************************************************* 
class SaliencyMaskDropout(nn.Module):
    r'''
        This will mask out an input tensor that can have arbitrary channels. It can also return the 
        binary mask it created from the saliency map. If it is used inline in a network, scale_map
        should be set to True. 
        
        Parameters
       
           keep_percent:         A scalar from 0 to 1. This represents what percent of the image to keep. 
           return_layer_only:    Tells us to just return the masked tensor only. Useful for putting layer into an nn.sequental. 
           scale_map:            Scale the output like we would a dropout layer?
           
        Will return
        
            (1) The maksed tensor.
            (2) The mask by itself.
    '''
    
    def __init__(self, keep_percent = 0.1, return_layer_only=False, scale_map=True):
        
        super(SaliencyMaskDropout, self).__init__()
        
        assert(isinstance(keep_percent,float))
        assert(keep_percent > 0)
        assert(keep_percent <= 1.0)
        
        self.keep_percent       = keep_percent
        if scale_map:
            self.scale              = 1.0/keep_percent
        else:
            self.scale              = 1.0
        self.drop_percent       = 1.0-self.keep_percent
        self.return_layer_only  = return_layer_only
                
    def forward(self, x, sal_map):
        
        assert(torch.is_tensor(x))
        assert(torch.is_tensor(sal_map))
        
        sal_map_size    = sal_map.size()
        x_size          = x.size()
        
        assert(len(x.size())        == 4)
        assert(len(sal_map.size())  == 3)
        
        assert(x_size[0] == sal_map_size[0])
        assert(x_size[2] == sal_map_size[1])
        assert(x_size[3] == sal_map_size[2])
        
        sal_map         = sal_map.reshape(sal_map_size[0], sal_map_size[1]*sal_map_size[2])
        
        r'''
            Using basically the same method we would to find the median, we find what value is 
            at n% in each saliency map. 
        '''
        num_samples     = int((sal_map_size[1]*sal_map_size[2])*self.drop_percent)
        s               = torch.sort(sal_map, dim=1)[0]
        k               = s[:,num_samples]
        k               = k.reshape(sal_map_size[0], 1)
        
        r'''
            We will create the saliency mask but we use torch.autograd so that we can optionally
            propagate the gradients backwards through the mask. k is assumed to be a dead-end, so 
            no gradients go to it. 
        '''
        drop_map        = DropMap.apply(sal_map, k) 
            
        drop_map        = drop_map.reshape(sal_map_size[0], 1, sal_map_size[1]*sal_map_size[2])
        x               = x.reshape(x_size[0], x_size[1], x_size[2]*x_size[3])

        r'''
            Multiply the input by the mask, but optionally scale it like we would a dropout layer.
        '''
        x               = x*drop_map*self.scale
        
        x               = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3])
        
        if self.return_layer_only:
            return x
        else:
            return x, drop_map.reshape(sal_map_size[0], sal_map_size[1], sal_map_size[2])
        
        
        