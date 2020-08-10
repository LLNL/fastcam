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

from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
import torch
from . import norm

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
class ResNet_FastCAM(models.ResNet):
    r'''
        Some of the code here is borrowed from the PyTorch source code.
        
        This is just a wrapper around PyTorch's own ResNet. We use it so we can only compute
        gradents over the last few layers and speed things up. Otherwise, ResNet will
        compute the usual gradients all the way through the network. 
        
        It is declared the usual way, but returns a saliency map in addition to the logits. 
        
        See: torchvision/models/resnet.py in the torchvision package. 
        
        Parameters:
        
            block:         This is the ResNet block pattern in a list. Something like [3, 4, 23, 3]
            layers:        What kind of blocks are we using. For instance models.resnet.Bottleneck . This should be callable. 
            num_classes:   How many classes will this network use? Should be an integer. Default: 1000  
            
        Will Return:
        
            logit:         The standard ResNet logits.
            saliency_map:  This is the combined, normalized saliency map which will resized to be the same
                           as the input [batch size x height x width]. 
        
    '''
    def __init__(self, block, layers, num_classes=1000, **kwargs):
        
        assert callable(block)
        assert isinstance(layers,list)
        assert isinstance(num_classes,int)
        
        super(ResNet_FastCAM, self).__init__(block, layers, num_classes=num_classes, **kwargs)
        
        self.get_norm       = norm.RangeNorm2D()    # Hard range normalization between 0 to 1
        self.num_classes    = num_classes           # We need this to define the max logits for the CAM map 
        self.layer          = None                  # This is a dummy layer to stash activations and gradients prior to average pooling
        
    def _forward_impl(self, x):

        r'''
            Turn off gradients so we don't prop all the way back. We only want to
            go back a few layers. Saves time and memory. 
        ''' 
        with torch.set_grad_enabled(False):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
        r'''
            To do:
                combine layer4 output with the SaliencyMap final layer
        '''
        
        r'''
            Turn on gradients and then get them into a container for usage by CAM
        '''
        with torch.set_grad_enabled(True):
            r'''
                Here we save out the layer so we can process it later
            '''
            x.requires_grad = True
            self.layer      = x
            self.layer.retain_grad()
            
            r'''
                Now run the rest of the network with gradients on
            '''
            x               = self.avgpool(self.layer)
            x               = torch.flatten(x, 1)
            x               = self.fc(x)

        return x

    def forward(self, x, class_idx=None, method='gradcam', retain_graph=False):
        r'''
            Call forward on the input x and return saliency map and logits. 
            
            Args:
            
                x:             A standard Torch Tensor of size [Batch x 3 x Height x Width]
                class_idx:     For CAM, what class should we propagate from. If None, use the max logit. 
                method:        A string, either 'gradcam' or 'gradcampp'. both yeild the same ROAR/KARR score.  
                retain_graph:  If you don't know what this means, leave it alone.   
                
            Return:
            
                logit:         The standard ResNet logits.
                saliency_map:  This is the combined, normalized saliency map which will resized to be the same
                               as the input [batch size x height x width]. 
        
            NOTE:
        
                Some of this code is borrowed from pytorch gradcam:
                https://github.com/vickyliin/gradcam_plus_plus-pytorch
        '''
        
        assert torch.is_tensor(x), "Input x must be a Torch Tensor" 
        assert len(x.size()) == 4, "Input x must have for dims [Batch x 3 x Height x Width]"
        assert class_idx is None or (isinstance(class_idx,int) and class_idx >=0 and class_idx < self.num_classes), "class_idx should not be silly"
        assert isinstance(retain_graph,bool), "retain_graph must be a bool."
        assert isinstance(method,str), "method must be a string"
        
        b, c, h, w  = x.size()
        
        r'''
            Run network forward on input x. Grads will only be kept on the last few layers. 
        '''
        logit       = self._forward_impl(x)
        
        r'''
            Torch will need to keep grads for these things.
        '''
        with torch.set_grad_enabled(True):
        
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
                
                NOTE:     This will only run backwards to the average pool layer then stop.
                          This is because we have set torch.set_grad_enabled(False) for all other layers. 
            '''
            self.zero_grad()
            score_end.backward(retain_graph=retain_graph)
            
            r'''
                Make naming clearer for next parts
            '''
            gradients   = self.layer.grad 
            activations = self.layer
        
        r'''
            Make sure torch doesn't keep grads for all this stuff since it will not be
            needed.
        '''
        with torch.set_grad_enabled(False):
            
            b, k, u, v          = gradients.size()

            if method=='gradcampp':
                r'''
                    GradCAM++ Computation
                '''
                alpha_num           = gradients.pow(2)
                alpha_denom         = gradients.pow(2).mul(2) + \
                                        activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
                alpha_denom         = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
                alpha               = alpha_num.div(alpha_denom+1e-7)
                positive_gradients  = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
                weights             = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

            elif method=='gradcam':
                r'''
                    Standard GradCAM Computation
                '''
                alpha               = gradients.view(b, k, -1).mean(2)
                weights             = alpha.view(b, k, 1, 1)
            elif method=='xgradcam':
                r'''
                    XGradCAM Model
                '''
                alpha               = (gradients*activations).view(b, k, -1).sum(2)
                alpha               = alpha / (activations.view(b, k, -1).sum(2) + 1e-6)
                weights             = alpha.view(b, k, 1, 1)
            else:
                r'''
                    Just GradCAM++ and original GradCAM
                '''
                raise ValueError("Unknown CAM type: \"{}\"".format(method))
    
            saliency_map        = (weights*activations).sum(1, keepdim=True) 
            
            r'''
                Lets just deal with positive gradients
            '''
            saliency_map        = F.relu(saliency_map)
            
            r'''
                Get back to input image size
            '''
            saliency_map        = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            
            r'''
                Hard range normalization
            '''
            saliency_map        = self.get_norm(saliency_map.squeeze(1))

        return  logit, saliency_map
    
# *******************************************************************************************************************
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    
    assert isinstance(pretrained, bool)
    assert isinstance(progress, bool)
    
    model = ResNet_FastCAM(block, layers, **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(models.resnet.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
        
    return model

# *******************************************************************************************************************
r'''
    Everything from here on down is just copied verbatum from torchvision. 
    
    Let me know if there is a better way to do this. 
'''

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', models.resnet.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', models.resnet.BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', models.resnet.Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', models.resnet.Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', models.resnet.Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', models.resnet.Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', models.resnet.Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the models.resnet.Bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', models.resnet.Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the models.resnet.Bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', models.resnet.Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
       
