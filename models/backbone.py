# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import math

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        print(return_layers,backbone)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
    
    
from typing import Dict, Iterable, Callable

    
class Backbone_dino(nn.Module):

    def __init__(self, enc_output_layer):
        super().__init__()   
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.qkv_feats = {'qkv_feats':torch.empty(0)}
        
        self.backbone._modules["blocks"][enc_output_layer]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_qkv)  #self.hook_fn_forward_qkv())
        
    def hook_fn_forward_qkv(self, module, input, output) -> Callable:
#         def fn(_, __, output):
        self.qkv_feats['qkv_feats'] = output
            

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
#         self.qkv_feats = []    
#         qkv_feats = []
            
#         self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(lambda self, input, output: qkv_feats.append(output))
        
        xs = self.backbone.get_intermediate_layers(xs)[0]

        feats = self.qkv_feats['qkv_feats']
        # Dimensions
        nh = 12 #Number of heads
        
        feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q, k, v = feats[0], feats[1], feats[2]
        q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
        
        xs = q[:,1:,:]

        xs = {'layer_top':xs}
#         xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0],int(math.sqrt(x.shape[1])),int(math.sqrt(x.shape[1])),self.num_channels)).permute(0,3,1,2)
            
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
import torch.nn.functional as F

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, qkv



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, qkv = self.attn(x)
        x = self.dropout(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = h + x
        return x, qkv

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.qkv_feats = {'qkv_feats': torch.empty(0)}

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        qkv_feats = []
        for encoder in self.encoder:
            x, qkv = encoder(x)
            qkv_feats.append(qkv)
        
        self.qkv_feats['qkv_feats'] = qkv_feats[-1]
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


class BackboneScratch(nn.Module):
    def __init__(self, enc_output_layer):
        super().__init__()
        self.backbone = VisionTransformer()
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.qkv_feats = {'qkv_feats':torch.empty(0)}
        
        self.backbone.encoder[enc_output_layer].attn.qkv.register_forward_hook(self.hook_fn_forward_qkv)

    def hook_fn_forward_qkv(self, module, input, output):
        self.qkv_feats['qkv_feats'] = output

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        xs = self.backbone(xs)
        
        feats = self.qkv_feats['qkv_feats']
        nh = 12
        
        feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q, k, v = feats[0], feats[1], feats[2]
        q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
        
        xs = q[:,1:,:]

        xs = {'layer_top':xs}

        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), self.num_channels)).permute(0,3,1,2)
            
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
#####
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import models


class BackboneBaseVGG(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        self.train_backbone = train_backbone
        self.num_channels = num_channels

        for param in backbone.parameters():
            param.requires_grad = train_backbone

        if return_interm_layers:
            return_layers = {
                '0': 'layer1',   # After first MaxPool2d
                '5': 'layer2',   # After second MaxPool2d
                '10': 'layer3',  # After third MaxPool2d
                '19': 'layer4',  # After fourth MaxPool2d
                '28': 'layer5'   # After fifth MaxPool2d
            }
        else:
            return_layers = {'28': 'layer5'}

        self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class BackboneVGG(BackboneBaseVGG):
    def __init__(self, train_backbone: bool, return_interm_layers: bool):
        # Load pretrained VGG19 model
        backbone = models.vgg16(pretrained=True)
        super().__init__(backbone, train_backbone, 512, return_interm_layers)



def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = True #args.masks
    
    if args.backbone == 'resnet50':
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    elif args.backbone == 'VGG19':
        backbone = BackboneVGG(train_backbone=train_backbone,return_interm_layers=return_interm_layers)
    elif args.backbone == "trans":
        backbone = BackboneScratch(enc_output_layer=8)
        
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model