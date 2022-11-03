import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url


class FrozenDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, training=False)
    
    def __repr__(self):
        return "FrozenDropout(p={})".format(self.p)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long)) # NOTE: added due to unexpected parameters when loading pre-trained weights 

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        version = None # NOTE: added

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        # NOTE: if a checkpoint is trained with BatchNorm and loaded (together with
        # version number) to FrozenBatchNorm, running_var will be wrong. One solution
        # is to remove the version number from the checkpoint.
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def ResNet101(output_stride, BatchNorm):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm)
    return model


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride, BatchNorm):
        self.inplanes = 64
        super().__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,  64, blocks=layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm) 
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm) 
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm) 
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        #self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm)
            )

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                             nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                             BatchNorm(planes * block.expansion),
                         )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, dilation=blocks[0] * dilation, downsample=downsample, BatchNorm=BatchNorm)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(self.inplanes, planes, stride=stride, dilation=blocks[i] * dilation, BatchNorm=BatchNorm)
            )

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # It is equivalent to nn.init.kaiming_normal_(m.weight, "relu", "fan_out")
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n)) 
            elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def freeze_bn(self): 
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False
        self.freeze_bn()


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, BatchNorm):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
        

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, BatchNorm, global_avg_pool_bn=False):
        if global_avg_pool_bn:  # If Batchsize is 1, error occur.
            super(ASPPPooling, self).__init__(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU())
        else:
            super(ASPPPooling, self).__init__(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True) 

    
class ASPP(nn.Module):
    def __init__(self, output_stride, BatchNorm, global_avg_pool_bn=False, in_channels=2048, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU())
        )
        
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise NotImplementedError
            
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, BatchNorm))

        modules.append(ASPPPooling(in_channels, out_channels, BatchNorm, global_avg_pool_bn))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.5) 
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        return self.dropout(res)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self): 
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        self.dropout = FrozenDropout(0.5) 

    def freeze_all(self): 
        for p in self.parameters():
            p.requires_grad = False
        self.freeze_bn()
        self.dropout = FrozenDropout(0.5) 


class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super().__init__()
        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
        )
        self.classifier = nn.ModuleList([nn.Conv2d(256, num_cls, kernel_size=1, stride=1, bias=True) for num_cls in num_classes]) 
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.last_conv(x)
        out = torch.cat([mod(x) for mod in self.classifier], dim=1)
        return out, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _init_cls(self):
        stdv = 1. / np.sqrt(256)
        for mod in self.classifier:
            nn.init.uniform_(mod.weight, -stdv, stdv)

    def _init_like_MiB(self):
        bg_weight = self.classifier[0].weight[0] 
        bg_bias   = self.classifier[0].bias[0] 
        constant  = torch.log(torch.tensor([self.num_classes[-1]+1], dtype=torch.float)).to(torch.device("cuda"))
        new_bg_bias = bg_bias - constant 
        self.classifier[-1].weight.data.copy_(bg_weight)
        self.classifier[-1].bias.data.copy_(new_bg_bias)
        self.classifier[0].bias[0].data.copy_(new_bg_bias[0])

    def freeze_bn(self): 
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def freeze_1st(self): 
        fix_list = [f"classifier.{ii}" for ii in range(len(self.num_classes)-1)] 
        for name, p in self.named_parameters():
            if any([ff in name for ff in fix_list]):
                p.requires_grad = False

    def freeze_2nd(self): 
        for name, p in self.named_parameters():
            if "last_conv" in name:
                p.requires_grad = False
        self.freeze_bn()

    def freeze_all(self): 
        for p in self.parameters():
            p.requires_grad = False
        self.freeze_bn()

    def get_features(self, x):
        x = self.last_conv(x)
        return x

    def get_prediction(self, x):
        return torch.cat([mod(x) for mod in self.classifier], dim=1)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes, sync_bn=False, output_stride=16, global_avg_pool_bn=True, freeze_type=None):
        super().__init__()
        if sync_bn:
            self.BatchNorm = nn.SyncBatchNorm 
        else:
            self.BatchNorm = nn.BatchNorm2d

        self.backbone   = ResNet101(output_stride, self.BatchNorm)
        self.aspp       = ASPP(output_stride, self.BatchNorm, global_avg_pool_bn, in_channels=2048, out_channels=256)
        self.decoder    = Decoder(num_classes, self.BatchNorm)

        if freeze_type == "step1":
            self.backbone.freeze_bn() 
            self.aspp.freeze_bn() 
        if freeze_type == "step3":
            self.backbone.freeze_all() 
            self.aspp.freeze_all() 
            self.decoder.freeze_2nd() 
        if freeze_type == "all":
            self.backbone.freeze_all() 
            self.aspp.freeze_all() 
            self.decoder.freeze_all() 

    def forward(self, x, memory=None, return_fea=False):
        input_size = x.shape[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x, f = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        if memory is not None:
            x_m = self.decoder.get_prediction(memory)
            if return_fea:  
                return x, x_m, f
            else:
                return x, x_m
        if return_fea:  
            return x, f
        else:
            return x

    def _init_like_MiB(self):
        self.decoder._init_like_MiB()

    def _init_cls(self):
        self.decoder._init_cls()

    def get_features(self, x):
        return self.decoder.get_features(self.aspp(self.backbone(x)))

    def get_prediction(self, x): 
        return self.decoder.get_prediction(x)

