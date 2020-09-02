#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torchvision import models

# Lets load things we need for **Grad-CAM**
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

#from gradcam.utils import visualize_cam
from gradcam import GradCAM

# The GradCAM kit throws a warning we don't need to see for this demo. 
import warnings
warnings.filterwarnings('ignore')

# Now we import the code for **this package**.
import maps
import misc
import mask
import norm
import resnet

# This is where we can set some parameters like the image name and the layer weights.
files = [   "ILSVRC2012_val_00049169.JPEG",
            "ILSVRC2012_val_00049273.JPEG",
            "ILSVRC2012_val_00049702.JPEG",
            "ILSVRC2012_val_00049929.JPEG",
            "ILSVRC2012_val_00049931.JPEG",
            "ILSVRC2012_val_00049937.JPEG",
            "ILSVRC2012_val_00049965.JPEG",
            "ILSVRC2012_val_00049934.224x224.png",
            "IMG_1382.jpg",
            "IMG_2470.jpg",
            "IMG_2730.jpg",
            "Nate_Face.png",
            "brant.png",
            "cat_dog.png",
            "collies.JPG",
            "dd_tree.jpg",
            "elephant.png",
            "multiple_dogs.jpg",
            "sanity.jpg",
            "snake.jpg",
            "spider.png",
            "swan_image.png",
            "water-bird.JPEG"]


# Lets set up where to save output and what name to use. 
output_dir          = 'outputs'                                 # Where to save our output images
input_dir           = 'images'                                  # Where to load our inputs from

weights             = [1.0, 1.0, 1.0, 1.0, 1.0]                 # Equal Weights work best 
                                                                # when using with GradCAM
    
#weights             = [0.18, 0.15, 0.37, 0.4, 0.72]            # Our saliency layer weights 
                                                                # From paper:
                                                                # https://arxiv.org/abs/1911.11293
        
in_height           = 224                                       # Size to scale input image to
in_width            = 224                                       # Size to scale input image to

# Choose how we want to normalize each map. 
#norm_method         = norm.GaussNorm2D
norm_method         = norm.GammaNorm2D # A little more accurate, but much slower

# You will need to pick out which layers to process. Here we grab the end of each group of layers by scale. 
layers              = ['relu','layer1','layer2','layer3','layer4']

# Now we create a model in PyTorch and send it to our device.
device              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model               = models.resnet50(pretrained=True)
model               = model.to(device)

many_image = []


for f in files:    
    model.eval()
    
    print("Image {}".format(f))
    
    input_image_name    = f    # Our input image to process 
    
    save_prefix         = os.path.split(os.path.splitext(input_image_name)[0])[-1]  # Chop the file extension and path
    load_image_name     = os.path.join(input_dir, input_image_name)
    
    os.makedirs(output_dir, exist_ok=True)

    # Lets load in our image. We will do a simple resize on it.
    in_tensor           = misc.LoadImageToTensor(load_image_name, device)
    in_tensor           = F.interpolate(in_tensor, size=(in_height, in_width), mode='bilinear', align_corners=False)
    
    # Now, lets get the Grad-CAM++ saliency map only.    
    resnet_gradcam      = GradCAM.from_config(model_type='resnet', arch=model, layer_name='layer4')
    cam_map, logit      = resnet_gradcam(in_tensor)
    
    # Create our saliency map object. We hand it our Torch model and names for the layers we want to tap. 
    get_salmap          = maps.SaliencyModel(model, layers, output_size=[in_height,in_width], weights=weights, 
                                             norm_method=norm_method)

    
    # Get Forward sal map
    csmap,smaps,_       = get_salmap(in_tensor)


    # Let's get our original input image back. We will just use this one for visualization. 
    raw_tensor          = misc.LoadImageToTensor(load_image_name, device, norm=False)
    raw_tensor          = F.interpolate(raw_tensor, size=(in_height, in_width), mode='bilinear', align_corners=False)
    
    
    # We create an object to get back the mask of the saliency map
    getMask             = mask.SaliencyMaskDropout(keep_percent = 0.1, scale_map=False)
    
    
    # Now we will create illustrations of the combined saliency map. 
    images              = []
    images              = misc.TileOutput(raw_tensor,csmap,getMask,images)
    
    # Let's double check and make sure it's picking the correct class
    too_logit           = logit.max(1)
    print("Network Class Output: {} : Value {} ".format(too_logit[1][0],too_logit[0][0]))
    
    
    # Now visualize the results
    images              = misc.TileOutput(raw_tensor, cam_map.squeeze(0), getMask, images)
    
    
    # ### Combined CAM and SMOE Scale Maps
    # Now we combine the Grad-CAM map and the SMOE Scale saliency maps in the same way we would combine Grad-CAM with Guided Backprop.
    fastcam_map         = csmap*cam_map
    
    
    # Now let's visualize the combined saliency map from SMOE Scale and GradCAM++.
    images              = misc.TileOutput(raw_tensor, fastcam_map.squeeze(0), getMask, images)
    
    
    # ### Get Non-class map
    # Now we combine the Grad-CAM map and the SMOE Scale saliency maps but create a map of the **non-class** objects. These are salient locations that the network found interesting, but are not part of the object class. 
    nonclass_map        = csmap*(1.0 - cam_map)
    
    
    # Now let's visualize the combined non-class saliency map from SMOE Scale and GradCAM++.
    images              = misc.TileOutput(raw_tensor, nonclass_map.squeeze(0), getMask, images)
    
    many_image          = misc.TileMaps(raw_tensor, csmap, cam_map.squeeze(0), fastcam_map.squeeze(0), many_image)
    
    
    # ### Visualize this....
    # We now put all the images into a nice grid for display.
    images              = make_grid(torch.cat(images,0), nrow=5)
    
    # ... save and look at it. 
    output_name         = "{}.CAM.jpg".format(save_prefix)
    output_path         = os.path.join(output_dir, output_name)
    
    save_image(images, output_path)
    
    del in_tensor
    del raw_tensor
    
many_image          = make_grid(torch.cat(many_image,0), nrow=4)
output_name         = "many.CAM.jpg".format(save_prefix)
output_path         = os.path.join(output_dir, output_name)
    
save_image(many_image, output_path)
    