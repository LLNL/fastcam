from __future__ import print_function, division, absolute_import

import cv2
import numpy as np
import torch.utils.data
import torch.cuda
from torchvision import models, transforms

r'''
    These are our python files.
'''
import misc
import maps
import draw

input_image_name    = "ILSVRC2012_val_00049934.224x224.png"     # Our input image to process
weights             = [0.18, 0.15, 0.37, 0.4, 0.72]             # Our saliency layer weights

save_prefix         = input_image_name[:-4]

# *******************************************************************************************************************
r'''
    Here we will load in a ResNet 50 network with ImageNet pretrained weights.
'''
print("Init model and parameters")
device              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model               = models.resnet50(pretrained=True)
model               = model.to(device)

# *******************************************************************************************************************
r'''
    We have different options for attaching saliency layers to our network. If we do not want to edit the network
    itself, we can attach call backs to each place we would put a layer. So, we are taking the output from five
    different ReLU layers which are all at the end of a scale.
    
    Example:
    
        If we print out the network:
         
            ...
         
            (layer1): Sequential(
                (0): Bottleneck(
                  (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU(inplace)
                  (downsample): Sequential(
                    (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  )
                )
                (1): Bottleneck(
                  (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU(inplace)
                )
                (2): Bottleneck(
                  (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (relu): ReLU(inplace)
                )
            )
  
  
        Lets grab that very last 'relu' layer at the very bottom. To get it, we would index as such:
      
            model.layer1[2].relu
          
        Then to attach a our callback via hook, we do:
        
            hook = misc.CaptureLayerOutput()
            model.layer1[2].relu.register_forward_hook(hook)
            
        When we run the network, the layer output will then be copied into hook.data
             
'''
print("Attach forward layer hooks")
hooks   = [ misc.CaptureLayerOutput() for i in range(5)]  # Create 5 callback hooks in a list
map_num = len(hooks) 

model.relu.register_forward_hook(hooks[0])
model.layer1[2].relu.register_forward_hook(hooks[1])
model.layer2[3].relu.register_forward_hook(hooks[2])
model.layer3[5].relu.register_forward_hook(hooks[3])
model.layer4[2].relu.register_forward_hook(hooks[4])

# *******************************************************************************************************************
print("Read in our image")
r'''
    We will read in an image and transform it in the usual manner for ImageNet. We then put it in a tensor and
    send it to our device.
'''
in_tensor   = misc.LoadImageToTensor(input_image_name, device)

in_height   = in_tensor.size()[2]
in_width    = in_tensor.size()[3]

# *******************************************************************************************************************
print("Evaluate our image using our network")        
r'''
    Run the network with the image. We do not care about the output. What we do care about is the layer data which
    will be placed in our callback objects when the network runs.
'''
model.eval()
with torch.set_grad_enabled(False):
    _  = model(in_tensor)

# *******************************************************************************************************************   
r'''
    Get all the call back data then run the saliency methods on them. We will produce a combined saliency map as
    well as return the upsampled individual saliency maps. 
'''
print("Get layer data and compute saliency")

r'''
    All three objects here are technically layers. So, they can be used inside your network as well. 
    
    example, in your __init__ something like:
    
        self.salmap_layer     = maps.SMOEScaleMap()
        
    then in forward(x) something like:
    
        x = self.relu(x)
        x = self.salmap_layer(x)
        
'''
getSmap     = maps.SMOEScaleMap()                                   # Create our saliency map object.
getNorm     = maps.Normalize2D()                                    # Create our cumulative distribution normalization object 
getCsmap    = maps.CombineSaliencyMaps(output_size=[in_height,in_width], map_num=map_num, weights=weights, resize_mode='bilinear') 

smaps       = [ getNorm(getSmap(x.data)) for x in hooks ]           # Grab each saliency map, normalize and put in a list
csmap,smaps = getCsmap(smaps)                                       # Combined the saliency maps, but also save individuals. 

misc.SaveTensorToImage(csmap, "{}.MAP_COMBINED.jpg".format(save_prefix))

# *******************************************************************************************************************
r'''
    Save each indivual saliency map to an image.
'''
print("Save individual saliency maps")

np_smaps = misc.TensorToNumpyImages(smaps)

for i in range(map_num):
    misc.SaveNumpyToImage(np_smaps[:,:,i], "{}.MAP_{}.jpg".format(save_prefix,i))

# *******************************************************************************************************************   
r'''
    We now create the combined visualization saliency maps.
'''
print("Compute combined display images")

SHM         = draw.HeatMap(shape=np_smaps.shape, weights=weights )              # Create our heat map drawer
LOVI        = draw.LOVI(shape=np_smaps.shape, weights=None)                     # Create out LOVI drawer
shm_im      = SHM.make(np_smaps)                                                # Combine the saliency maps into one heat map
lovi_im     = LOVI.make(np_smaps)                                               # Combine the saliency maps into one LOVI image

cv2.imwrite("{}.HEAT.jpg".format(save_prefix),(shm_im*255.0).astype(np.uint8))  # Save our heat map
cv2.imwrite("{}.LOVI.jpg".format(save_prefix),(lovi_im*255.0).astype(np.uint8)) # Save our LOVI image

# *******************************************************************************************************************    
print("DONE")

