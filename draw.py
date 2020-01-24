from __future__ import print_function, division, absolute_import

import cv2
import numpy as np
import math

# *******************************************************************************************************************
class _Draw(object):
    r'''
        This is a base class for drawing different representations of saliency maps.
    '''
    
    def __init__(self, shape, weights, color):
        
        assert(isinstance(shape,tuple) or isinstance(shape,list))
        assert(len(shape) == 3)
        assert(isinstance(color,int))   # How openCV treats this        
        
        self.height  = shape[0]
        self.width   = shape[1]
        self.chans   = shape[2]
        
        self.color   = color 
                
        assert(self.height   > 0)
        assert(self.width    > 0)
        assert(self.chans    > 0)
        
        if weights is None:
            self.weights = np.array([1.0 for _ in range(self.chans)]).astype(np.float32)
        elif len(weights) == 1:
            assert(weights[0] > 0)
            self.weights = np.array([weights[0] for _ in range(self.chans)]).astype(np.float32)  
        else:
            assert(len(weights) == self.chans)        
            self.weights = np.array(weights).astype(np.float32)
                
        self.fc             = float(self.chans)
        self.frac           = 1.0/self.fc
        self.HSV_img        = np.empty((self.height,self.width,3), dtype=np.float32)
        
        self.sum_weights    = np.sum(self.weights)
        
    def _range_normalize(self, data):
        
        norm = data.max() - data.min()
        if norm != 0:
            data = (data - data.min()) / norm
            
        return data

# *******************************************************************************************************************        
class HeatMap(_Draw):        
    r'''
        This will create a heat map from a stack of saliency maps. in a H x W x C numpy array. 
        
        Parameters
       
           shape:    This is a list (H,W,C) of the expected size of the saliency map.
           weights:  This is a list of length C of weight for each channel.
           color:    The color conversion method to use.  
           
        Returns:
        
            An openCV compatible numpy array sized H x W x 3 
    ''' 
        
    def __init__(self, shape, weights=None, color=cv2.COLOR_HSV2BGR):
        
        super(HeatMap, self).__init__(shape=shape, weights=weights, color=color)
        
        self.Cexp        = 1.0/math.exp(1.0)
        
    def make(self, input_patches):

        r'''
            Input patches should be sized [height x width x channels]. Here channels is each saliency map.
        '''
        assert(isinstance(input_patches, np.ndarray))
        assert(len(input_patches.shape) == 3)
        assert(input_patches.shape[0] == self.height)
        assert(input_patches.shape[1] == self.width)
        assert(input_patches.shape[2] == self.chans)
        
        patches         = self._range_normalize(input_patches.astype(np.float32)) * self.weights
        
        # Set intensity to be the weighted average
        V               = np.sum(patches, 2) / self.sum_weights
        V               /= V.max()    
        
        # Use the standard integral method for saturation, but give it a boost.
        if self.frac != 1.0:
            S               = 1.0 - (np.sum(patches,2)/(self.fc*np.amax(patches,2)) - self.frac)*(1.0/(1.0 - self.frac))
        else:
            S               = V
            
        S               = pow(S,self.Cexp)
        
        # Set H,S and V in that order. 
        self.HSV_img[:,:,0] = (1.0 - V) * 240.0 
        self.HSV_img[:,:,1] = S
        self.HSV_img[:,:,2] = V
        
        return cv2.cvtColor(self.HSV_img,  self.color)

# *******************************************************************************************************************
class LOVI(_Draw):        
    r'''
        This will create a LOVI map from a stack of saliency maps. in a H x W x C numpy array. 
        
        Parameters
       
           shape:    This is a list (H,W,C) of the expected size of the saliency map.
           weights:  This is a list of length C of weight for each channel.
           color:    The color conversion method to use.  
           
        Returns:
        
            An openCV compatible numpy array sized H x W x 3 
    ''' 
    def __init__(self, shape, weights=None, color=cv2.COLOR_HSV2BGR):
        
        super(LOVI, self).__init__(shape=shape, weights=weights, color=color)
        
        self.pos_img    = np.empty((self.height,self.width,self.chans),     dtype=np.float32)
        
        y = 1.0/((self.fc - 1.0)/self.fc)
        
        for c_i in range(self.chans): 
            self.pos_img[:,:,c_i]   = 1.0 - (float(c_i)/(self.fc))*y    
        
    def make(self, input_patches):
        r'''
            Input patches should be sized [height x width x channels]. Here channels is each saliency map.
        '''
        assert(isinstance(input_patches, np.ndarray))
        assert(len(input_patches.shape) == 3)
        assert(input_patches.shape[0] == self.height)
        assert(input_patches.shape[1] == self.width)
        assert(input_patches.shape[2] == self.chans)
        
        patches         = self._range_normalize(input_patches.astype(np.float32)) * self.weights
        
        # Compute position
        pos             = patches * self.pos_img
        # Get Mean
        m               = np.sum(pos,2) / np.sum(patches,2)

        # Set H,S and V in that order.        
        self.HSV_img[:,:,0] = m*300
        self.HSV_img[:,:,1] = 1.0 - (np.sum(patches,2)/(self.fc*np.amax(patches,2)) - self.frac)*(1.0/(1.0 - self.frac))
        self.HSV_img[:,:,2] = np.amax(patches, 2)
          
        return cv2.cvtColor(self.HSV_img,  self.color)

        


