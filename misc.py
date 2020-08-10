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

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from . import maps
from torchvision import models, transforms
from statistics import stdev # Built-in
from gradcam import GradCAM
from gradcam.utils import visualize_cam
from matplotlib import pyplot as plt

# ******************************************************************************************************************* 
def from_gpu(data):
  
    return data.cpu().detach().numpy()

# ******************************************************************************************************************* 
def detach(data):
    
    return data.detach()

# ******************************************************************************************************************* 
def no_proc(data):
 
    return data

# *******************************************************************************************************************
# ******************************************************************************************************************* 
class CaptureLayerData(object):
    r"""
        This is a helper class to get layer data such as network activations from
        the network. PyTorch hides this away. To get it, we attach on object of this 
        type to a layer. This tells PyTorch to give us a copy when it runs a forward 
        pass.
    """

    def __init__(self, device, post_process=no_proc):
        self.data           = None
        self.post_process   = post_process
        
        if device is not None:
            self.device         = torch.device(device)
        else:
            self.device         = None
        
        assert(callable(self.post_process) or self.post_process is None)
             
# *******************************************************************************************************************            
class CaptureLayerOutput(CaptureLayerData):
    
    def __init__(self, device=None, post_process=detach):
        
        super(CaptureLayerOutput, self).__init__(device, post_process)      
        
    def __call__(self, m, i, o):
        
        if self.device is None or self.device == o.device:
            
            if self.post_process is None:
                self.data = o.data
            else:
                self.data = self.post_process(o.data) 

# *******************************************************************************************************************            
class CaptureGradOutput(CaptureLayerData):
    
    def __init__(self, device=None, post_process=detach):
        
        super(CaptureGradOutput, self).__init__(device, post_process)      
        
    def __call__(self, m, i, o):
        
        if self.device is None or self.device == o[0].device:
            
            # o seems to usualy be size 1
            
            if self.post_process is None:
                self.data = o[0]
            else:
                self.data = self.post_process(o[0]) 
            
# *******************************************************************************************************************            
class CaptureLayerInput(CaptureLayerData):
    
    def __init__(self, device=None, array_item=None):
        
        assert(isinstance(array_item, int) or array_item is None)
        
        if isinstance(array_item, int):
            assert array_item >= 0
        
        self.array_item = array_item
        
        super(CaptureLayerInput, self).__init__(device, post_process=None)     
        
    def __call__(self, m, i, o):
        
        if self.device is None or self.device == o.device:
            
            if self.array_item is None:
                self.data = [n.data for n in i]
            else:
                self.data = i.data[self.array_item]

# *******************************************************************************************************************            
class CaptureLayerPreInput(CaptureLayerData):
    
    def __init__(self, device=None, array_item=None):
        
        assert(isinstance(array_item, int) or array_item is None)
        
        if isinstance(array_item, int):
            assert array_item >= 0
        
        self.array_item = array_item
        
        super(CaptureLayerPreInput, self).__init__(device, post_process=None)     
        
    def __call__(self, m, i):
        
        if self.device is None or self.device == o.device:
            
            if self.array_item is None:
                self.data = [n.data for n in i]
            else:
                self.data = i.data[self.array_item]
                
# *******************************************************************************************************************            
class CaptureGradInput(CaptureLayerData):
    
    def __init__(self, device=None, post_process=detach):
        
        super(CaptureGradInput, self).__init__(device, post_process)      
        
    def __call__(self, m, i, o):
        
        if self.device is None or self.device == i[0].device:
            
             # i seems to usualy be size 1
            
            if self.post_process is None:
                self.data = i[0]
            else:
                self.data = self.post_process(i[0])     
                
# *******************************************************************************************************************
# *******************************************************************************************************************             
def LoadImageToTensor(file_name, device, norm=True, conv=cv2.COLOR_BGR2RGB, 
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                      cv_process_list=[],pt_process_list=[]):
    
    assert isinstance(file_name,str)
    assert isinstance(device,torch.device)
    
    toTensor    = transforms.ToTensor()
    toNorm      = transforms.Normalize(mean,std)
    
    cv_im       = cv2.imread(file_name) 
    assert(cv_im is not None)
    
    for l in cv_process_list:
        cv_im = l(cv_im)

    cv_im       = (cv2.cvtColor(cv_im,  conv) / 255.0).astype(np.float32)  # Put in a float image and range from 0 to 1
    pt_im       = toTensor(cv_im)
    
    for l in pt_process_list:
        pt_im = l(pt_im)
    
    if norm:
        pt_im       = toNorm(pt_im)                                           # Do mean subtraction and divide by std. Then convert to Tensor object.
    
    pt_im       = pt_im.reshape(1, 3, pt_im.size()[1], pt_im.size()[2])                 # Add an extra dimension so now its 4D
    pt_im       = pt_im.to(device)                                                      # Send to the GPU
    
    return pt_im    
                                               
# *******************************************************************************************************************             
def SaveGrayTensorToImage(tens, file_name):

    assert isinstance(file_name,str)
    assert torch.is_tensor(tens)
    assert len(tens.size()) == 3 or len(tens.size()) == 4
    
    if len(tens.size()) == 4:
        sz         = tens.size()
        assert(sz[0] == 1)
        assert(sz[1] == 1)
        tens       = tens.reshape(sz[1],sz[2],sz[3])

    np_tens    = tens.cpu().detach().numpy()                            # Put the tensor into a cpu numpy
    np_tens    = np_tens.transpose(1, 2, 0)                             # Transpose to [height x width x channels]
    np_tens    = cv2.cvtColor(np_tens, cv2.COLOR_GRAY2BGR)              # Convert gray to BGR color 
    np_tens    = (np_tens*255.0).astype(np.uint8)                       # Make it range from 0 to 255 and convert to byte

    cv2.imwrite(file_name,np_tens)

# *******************************************************************************************************************             
def SaveColorTensorToImage(tens, file_name):

    assert isinstance(file_name,str)
    assert torch.is_tensor(tens)
    assert len(tens.size()) == 3 or len(tens.size()) == 4

    if len(tens.size()) == 4:
        sz         = tens.size()
        assert sz[0] == 1
        assert sz[1] == 3
        tens       = tens.reshape(sz[1],sz[2],sz[3])

    np_tens    = tens.cpu().detach().numpy()                            # Put the tensor into a cpu numpy
    np_tens    = np_tens.transpose(1, 2, 0)                             # Transpose to [height x width x channels]
    np_tens    = cv2.cvtColor(np_tens, cv2.COLOR_RGB2BGR)               # Convert gray to BGR color 
    np_tens    = (np_tens*255.0).astype(np.uint8)                       # Make it range from 0 to 255 and convert to byte

    cv2.imwrite(file_name,np_tens)
        
# *******************************************************************************************************************             
def SaveGrayNumpyToImage(np_im, file_name):
    
    assert isinstance(file_name,str)
    assert isinstance(np_im, np.ndarray)
    assert len(np_im.shape) == 2
    
    np_im    = cv2.cvtColor(np_im,  cv2.COLOR_GRAY2BGR)                 # Convert gray to BGR color
    np_im    = (np_im*255.0).astype(np.uint8)                           # Make it range from 0 to 255 and convert to byte
    
    cv2.imwrite(file_name, np_im)                                       # Save the image
    
# *******************************************************************************************************************             
def TensorToNumpyImages(tens):
    
    assert torch.is_tensor(tens)
    assert len(tens.size()) == 3 or len(tens.size()) == 4

    np_im = tens.cpu().detach().numpy()                                     # Now we get the individual saliency maps to save

    if len(tens.size()) == 4:
        assert(tens.size()[0] == 1)
        np_im = np_im.transpose(0, 2, 3, 1)                                 # Transpose to [batch x height x width x channels]
        np_im = np_im.reshape(np_im.shape[1],np_im.shape[2],np_im.shape[3]) # Chop off the extra dimension since our batch size is 1
    else:
        np_im    = np_im.transpose(1, 2, 0)
    
    return np_im

# *******************************************************************************************************************             
def NumpyToTensorImages(np_im, device='cpu'):
    
    assert isinstance(np_im, np.ndarray)
    assert len(np_im.shape) == 3 or len(np_im.shape) == 4
    
    toTensor    = transforms.ToTensor()
    
    pt_im       = toTensor(np_im)
    pt_im       = pt_im.to(device)                                                      # Send to the device
    
    return pt_im   

# *******************************************************************************************************************             
def AlphaBlend(im1, im2, alpha=0.75):
    
    assert isinstance(im1,np.ndarray) or torch.is_tensor(im1)
    assert isinstance(im2,np.ndarray) or torch.is_tensor(im2)
    assert type(im1) == type(im2)
    assert isinstance(alpha,float)
    
    t_alpha     = alpha
    r_alpha     = 1.0
    norm        = t_alpha + r_alpha*(1.0 - t_alpha)
    
    return (im1*t_alpha + im2*r_alpha*(1.0 - t_alpha))/norm

# *******************************************************************************************************************             
def AlphaMask(im1, mask, alpha=1.0):
    
    assert isinstance(im1,np.ndarray) or torch.is_tensor(im1)
    assert isinstance(mask,np.ndarray) or torch.is_tensor(mask)
    assert type(im1) == type(mask)
    assert isinstance(alpha,float)
    
    if isinstance(im1,np.ndarray):
        im2         = np.zeros_like(im1)
    else:
        im2         = torch.zeros_like(im1)
    
    t_alpha     = mask
    r_alpha     = alpha
    norm        = t_alpha + r_alpha*(1.0 - t_alpha)
    
    return (im1*t_alpha + im2*r_alpha*(1.0 - t_alpha))/norm


# *******************************************************************************************************************             
def AttenuateBorders(im, ammount=[0.333,0.666]):
    
    assert isinstance(im,np.ndarray) or torch.is_tensor(im)
    assert isinstance(ammount,list)
    
    im[:,0,:]       = im[:,0,:]         * ammount[0]
    im[:,:,0]       = im[:,:,0]         * ammount[0]
    im[:,-1,:]      = im[:,-1,:]        * ammount[0]
    im[:,:,-1]      = im[:,:,-1]        * ammount[0]
    
    im[:,1,1:-2]    = im[:,1,1:-2]      * ammount[1]
    im[:,1:-2,1]    = im[:,1:-2,1]      * ammount[1]
    im[:,-2,1:-2]   = im[:,-2,1:-2]     * ammount[1]
    im[:,1:-2,-2]   = im[:,1:-2,-2]     * ammount[1]
    
    return im

# *******************************************************************************************************************             
def RangeNormalize(im):
    
    assert torch.is_tensor(im)
    
    imax = torch.max(im)
    imin = torch.min(im)
    rng  = imax - imin
    
    if rng != 0:
        return (im - imin)/rng
    else:
        return im

# *******************************************************************************************************************             
def TileOutput(tensor, mask, mask_func, image_list = []):

    assert torch.is_tensor(tensor)
    assert torch.is_tensor(mask)
    assert isinstance(image_list,list)
    assert callable(mask_func)

    heatmap, result         = visualize_cam(mask, tensor)
    
    hard_masked,_           = mask_func(tensor, mask)
    hard_masked             = hard_masked.squeeze(0)
    masked                  = AlphaMask(tensor, mask).squeeze(0)
    masked                  = RangeNormalize(masked)
        
    image_list.append(torch.stack([tensor.squeeze().cpu(), heatmap.cpu(), 
                                   result.cpu(), masked.cpu(), hard_masked.cpu()], 0))
    
    return image_list

# *******************************************************************************************************************
def show_hist(tens,file_name=None, max_range=256):
    

    '''
    Create histogram of pixel values and display.
    '''
    
    tens = RangeNormalize(tens) * float(max_range)
    
    img  = tens.cpu().detach().numpy()
    
    hist,bins = np.histogram(img.flatten(),max_range,[0,max_range])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),max_range,[0,max_range], color = 'r')
    plt.xlim([0,max_range])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    
    if file_name is not None:
        plt.savefig(file_name)
    
    plt.show()     

# *******************************************************************************************************************             
class DeNormalize:
    
    def __init__(self,mean,std):
        
        assert isinstance(mean,list)
        assert isinstance(std,list)
        
        assert len(mean)    == 3
        assert len(std)     == 3
        
        self.mean   = torch.tensor(mean).reshape(1,len(mean),1)
        self.std    = torch.tensor(std).reshape(1,len(std),1)
        
        
    def __call__(self, tens):

        assert torch.is_tensor(tens)
        assert len(tens.size()) == 4
              
        if tens.device != self.std.device:
            self.std = self.std.to(tens.device)
            
        if tens.device != self.mean.device:
            self.mean = self.mean.to(tens.device)
        
        sz      = tens.size()
        
        tens    = tens.reshape(sz[0],sz[1],sz[2]*sz[3])
        tens    = tens*self.std + self.mean
        tens    = tens.reshape(sz[0],sz[1],sz[2],sz[3])
        
        return tens
        
# *******************************************************************************************************************             
class SmoothGrad:
    
    def __init__(self, iters=15, magnitude=True, stdev_spread=.15, maps_magnitude=False):
        
        self.iters          = iters
        self.magnitude      = magnitude
        self.stdev_spread   = stdev_spread
    
        self.getSmap        = maps.SMOEScaleMap()                                   
        self.getNorm        = maps.GaussNorm2D()      
        self.maps_magnitude = maps_magnitude                              
 
    
    def __call__(self, in_tensor, model,  hooks, weights, debug=False):
        
        in_height       = in_tensor.size()[2]
        in_width        = in_tensor.size()[3]
        map_num         = len(hooks) 

        getCsmap        = maps.CombineSaliencyMaps(output_size=[in_height,in_width], 
                                                   map_num=map_num, weights=weights, resize_mode='bilinear',magnitude=self.maps_magnitude) 
        
        stdev           = self.stdev_spread * (torch.max(in_tensor) - torch.min(in_tensor))
        
        ret_image       = []
        
        if debug:
            out_csmap       = []
            out_smaps       = []
        else:
            out_csmap       = torch.zeros((1, in_height, in_width), dtype=in_tensor.dtype, device=in_tensor.device)
            out_smaps       = torch.zeros((1, map_num, in_height, in_width), dtype=in_tensor.dtype, device=in_tensor.device)
        
        for i in range(self.iters):
            
            noise           = torch.normal(mean=0, std=stdev, size=in_tensor.size())
            in_tensor_noise = in_tensor + noise
            
            model.eval()
            
            with torch.set_grad_enabled(False):
                _ = model(in_tensor_noise)
                
            smaps       = [ self.getNorm(self.getSmap(x.data)) for x in hooks ] 
            csmap,smaps = getCsmap(smaps)
            
            if debug:
                out_csmap.append(csmap)
                out_smaps.append(smaps)
                ret_image.append(in_tensor_noise)
            else:
                if self.magnitude:
                    out_csmap += (csmap * csmap)
                    out_smaps += (smaps * smaps)
                else:
                    out_csmap += csmap
                    out_smaps += smaps
        
        if not debug:
            out_csmap /= self.iters
            out_smaps /= self.iters
        
        return out_csmap, out_smaps, ret_image
    