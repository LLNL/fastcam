#!/usr/bin/env python
# coding: utf-8

# # DEMO: Running FastCAM for the Exceptionally Impatient

# ### Import Libs

# In[1]:


import os
from IPython.display import Image
import cv2
import numpy as np


# Lets load the **PyTorch** Stuff.

# In[2]:


import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import warnings
warnings.filterwarnings('ignore')


# Now we import the saliency libs for **this package**.

# In[3]:


import maps
import misc
import mask
import norm
import draw
import resnet


# ### Set Adjustable Parameters
# This is where we can set some parameters like the image name and the layer weights.

# In[4]:


input_image_names   = ["OMF2_715165_45972761_POST_160_267_EXIT_RB_191025_072939.jpg",
                       "OMF2_715165_45723294_POST_144_306_EXIT_RB_191024_174606.jpg",
                       "OMF2_715165_47391944_POST_85_273_EXIT_RB_191024_203453.jpg",
                       "OMF2_715165_48284471_POST_362_243_EXIT_RB_191024_204748.jpg",
                       "OMF2_715165_48286090_POST_105_340_EXIT_RB_191024_220021.jpg",
                       "OMF2_715165_49772491_POST_142_332_EXIT_RB_191025_075343.jpg",
                       "OMF2_715165_48738510_POST_82_283_EXIT_RB_191024_220953.jpg",
                       "OMF2_715165_48286273_POST_67_288_INPUT_RB_191024_194155.jpg"
                      ]


# In[5]:


input_image_name    = input_image_names[0]                      # Pick which image you want from the list
output_dir          = 'outputs'                                 # Where to save our output images
input_dir           = 'images'                                  # Where to load our inputs from
# Assumes input image size 1392x1040
in_height           = 524                                       # Size to scale input image to
in_width            = 696                                       # Size to scale input image to
in_patch            = 480
view_height         = 1040
view_width          = 1392
view_patch          = 952


# Now we set up what layers we want to sample from and what weights to give each layer. We specify the layer block name within ResNet were we will pull the forward SMOE Scale results from. The results will be from the end of each layer block.   

# In[6]:


weights             = [0.18, 0.15, 0.37, 0.4, 0.72]             # Our saliency layer weights 
                                                                # From paper:
                                                                # https://arxiv.org/abs/1911.11293
layers              = ['relu','layer1','layer2','layer3','layer4']


# **OPTIONAL:** We can auto compute which layers to run over by setting them to *None*. **This has not yet been quantitatively tested on ROAR/KARR.** 

# In[7]:


#weights             = None
#layers              = None


# ### Set Up Loading and Saving File Names
# Lets touch up where to save output and what name to use for output files. 

# In[8]:


save_prefix         = os.path.split(os.path.splitext(input_image_name)[0])[-1]  # Chop the file extension and path
load_image_name     = os.path.join(input_dir, input_image_name)

os.makedirs(output_dir, exist_ok=True)


# Take a look at the input image ...
# Good Doggy!

# In[9]:


Image(filename=load_image_name) 


# ### Set Up Usual PyTorch Network Stuff
# Go ahead and create a standard PyTorch device. It can use the CPU if no GPU is present. This demo works pretty well on just CPU. 

# In[10]:


device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Now we will create a model. Here we have a slightly customized ResNet that will only propagate backwards the last few layers. The customization is just a wrapper around the stock ResNet that comes with PyTorch. SMOE Scale computation does not need any gradients and GradCAM variants only need the last few layers. This will really speed things up, but don't try to train this network. You can train the regular ResNet if you need to do that. Since this network is just a wrapper, it will load any standard PyTorch ResNet training weights.  

# In[11]:


model       = resnet.resnet50(pretrained=True)
checkpoint  = torch.load("models/NIF_resnet50.random_rotate4.model.pth.tar")
model.load_state_dict(checkpoint,strict=True) 
model       = model.to(device)


# ### Load Images
# Lets load in our image into standard torch tensors. We will do a simple resize on it.

# In[12]:


h_offset    = int((in_height-in_patch)/2)
w_offset    = int((in_width -in_patch)/2)


# In[13]:


in_tensor   = misc.LoadImageToTensor(load_image_name, device)
in_tensor   = F.interpolate(in_tensor, size=(in_height, in_width), mode='bilinear', align_corners=False)
in_tensor   = in_tensor[:,:,h_offset:in_patch+h_offset, w_offset:in_patch+w_offset] # Crop


# For illustration purposes, Lets also load a non-normalized version of the input.

# In[14]:


h_offset    = int((view_height-view_patch)/2)
w_offset    = int((view_width -view_patch)/2)


# In[15]:


raw_tensor  = misc.LoadImageToTensor(load_image_name, device, norm=False)
raw_tensor  = raw_tensor[:,:,h_offset:view_patch+h_offset,w_offset:view_patch+w_offset] # Crop


# ### Set Up Saliency Objects

# We create an object to create the saliency map given the model and layers we have selected. 

# In[16]:


getSalmap   = maps.SaliencyMap(model, layers, output_size=[in_patch,in_patch],weights=weights,
                              norm_method=norm.GammaNorm2D, cam_each_map=True)


# Now we set up our masking object to create a nice mask image of the %10 most salient locations. You will see the results below when it is run.

# In[17]:


getMask     = mask.SaliencyMaskDropout(keep_percent = 0.1, scale_map=False)


# ### Run Saliency
# Now lets run our input tensor image through the net and get the 2D saliency map back. 

# In[18]:


cam_map,sal_maps,logit = getSalmap(in_tensor)
cam_map                = F.interpolate(cam_map.unsqueeze(0), size=(view_patch, view_patch), 
                                   mode='bilinear', align_corners=False)
cam_map                = cam_map.squeeze(0)


# ### Display Network Classification

# In[19]:


print("Class Likelihood Bad: {} Good: {}".format(logit[0,0],logit[0,1]))


# ### Visualize It
# Take the images and create a nice tiled image to look at. This will created a tiled image of:
# 
#     (1) The input image.
#     (2) The saliency map.
#     (3) The saliency map overlaid on the input image.
#     (4) The raw image enhanced with the most salient locations.
#     (5) The top 10% most salient locations. 

# In[20]:


images      = misc.TileOutput(raw_tensor, cam_map, getMask)


# We now put all the images into a nice grid for display.

# In[21]:


images      = make_grid(torch.cat(images,0), nrow=5)


# ... save and look at it. 

# In[22]:


output_name = "{}.FASTCAM.jpg".format(save_prefix)
output_path = os.path.join(output_dir, output_name)

save_image(images, output_path)
Image(filename=output_path) 


# This image should look **exactly** like the one on the README.md on Github minus the text. 

# ### Alternative Visualizations

# In[23]:


sal_maps    = sal_maps.squeeze(0)

SHM         = draw.HeatMap(shape=sal_maps.size, weights=weights)   # Create our heat map drawer
LOVI        = draw.LOVI(shape=sal_maps.size, weights=None)         # Create out LOVI drawer

shm_im      = SHM.make(sal_maps, raw_tensor)                       # Combine the saliency maps 
                                                                   # into one heat map
lovi_im     = LOVI.make(sal_maps, raw_tensor)                      # Combine the saliency maps 
                                                                   # into one LOVI image


# ### Overlay with Difference Pallet

# In[24]:


output_name = "{}.HEAT.jpg".format(save_prefix)
output_path = os.path.join(output_dir, output_name)
cv2.imwrite(output_path, (shm_im*255.0).astype(np.uint8))

Image(filename=output_path) 


# ### LOVI Layer Map

# In[25]:


output_name   = "{}.LOVI.jpg".format(save_prefix)
output_path   = os.path.join(output_dir, output_name)
cv2.imwrite(output_path, (lovi_im*255.0).astype(np.uint8))

Image(filename=output_path) 


# ### Each Layer Saliency Map by Itself

# In[26]:


# We will range normalize each saliency map from 0 to 1
getNorm     = norm.RangeNorm2D() 

# Put each saliency map into the figure
il = [getNorm(sal_maps[i,:,:].unsqueeze(0)).squeeze(0) for i in range(sal_maps.size()[0])] 
    
images        = [torch.stack(il, 0)]          
images        = make_grid(torch.cat(images, 0), nrow=5)
output_name   = "{}.SAL_MAPS.jpg".format(save_prefix)
sal_img_file  = os.path.join(output_dir, output_name)

save_image(images.unsqueeze(1), sal_img_file)

Image(filename=sal_img_file) 


# In[ ]:





# In[ ]:




