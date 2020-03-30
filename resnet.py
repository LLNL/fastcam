from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
import torch
import norm

# *******************************************************************************************************************       
class ScoreMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scores):
        
        ctx.save_for_backward(scores)
        return torch.tensor(1)
    
    @staticmethod
    def backward(ctx, grad):
                
        saved        = ctx.saved_tensors
        g_scores     = torch.ones_like(saved[0])
        
        return g_scores

class ResNet_FastCAM(models.ResNet):
    r'''
        Some of the code here is borrowed from the PyTorch source code.
        
        This is just a wrapper around PyTorches ResNet. We use it so we can only compute
        gradents over the last few layers and speed things up. Otherwise, ResNet will
        compute the usual gradients all the way through the network. 
        
        It is declared the usual way, but returns a saliency map in addition to the logits. 
    '''
    def __init__(self, block, layers, num_classes=1000, **kwargs):
        
        super(ResNet_FastCAM, self).__init__(block, layers, num_classes=num_classes, **kwargs)
        
        assert isinstance(num_classes,int)
        
        self.get_norm       = norm.RangeNorm2D()
        self.num_classes    = num_classes
        self.layer          = None
        
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
        
        r'''
            To do:
                Only compute gradients on the very last layer of layer4
                combine layer4 output with the SaliencyMap final layer
        '''
        
        r'''
            Turn on gradients and then get them into a container for usage by CAM
        '''
        with torch.set_grad_enabled(True):
            # Here we save out the layer so we can process it later
            x           = self.layer4(x)
            self.layer  = x
            self.layer.retain_grad()
            x           = self.layer
            x           = self.avgpool(x)
            x           = torch.flatten(x, 1)
            x           = self.fc(x)

        return x

    def forward(self, x, class_idx=None, retain_graph=False, method='gradcampp'):
        
        r'''
            Some of this code is borrowed from pytorch gradcam:
            https://github.com/vickyliin/gradcam_plus_plus-pytorch
        '''
        
        b, c, h, w  = x.size()
        
        logit       = self._forward_impl(x)
        
        with torch.set_grad_enabled(True):
        
            if class_idx is None:
    
                sz      = logit.size()
                
                lm      = logit.max(1)[1]  
                r'''
                    This gets the logits into a form usable when we run a batch. This seems suboptimal.
                    Open to ideas about how to make this better/faster
                '''
                lm      = torch.stack([i*self.num_classes + v for i,v in enumerate(lm)])
                    
                logit   = logit.reshape(sz[0]*sz[1])
                
                score   = logit[lm]
                    
                logit   = logit.reshape(sz[0],sz[1])    
                score   = score.reshape(sz[0],1,1,1)
            else:
                score   = logit[:, class_idx].squeeze()
    
    
            score_end   = ScoreMap.apply(score)
    
            self.zero_grad()
    
            score_end.backward(retain_graph=retain_graph)
            
            gradients   = self.layer.grad 
            activations = self.layer
        
        with torch.set_grad_enabled(False):
            
            b, k, u, v          = gradients.size()
            
            if method=='gradcampp':
                alpha_num           = gradients.pow(2)
                alpha_denom         = gradients.pow(2).mul(2) + \
                                        activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
                alpha_denom         = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
                alpha               = alpha_num.div(alpha_denom+1e-7)
                positive_gradients  = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
                weights             = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
            elif method=='gradcam':
                alpha               = gradients.view(b, k, -1).mean(2)
                weights             = alpha.view(b, k, 1, 1)
    
            saliency_map        = (weights*activations).sum(1, keepdim=True) 
            saliency_map        = F.relu(saliency_map)
            
            saliency_map        = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            
            saliency_map        = self.get_norm(saliency_map.squeeze(1))

        return  logit, saliency_map

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_FastCAM(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(models.resnet.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


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
       