# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatioTemporalFocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0., num_frames=128):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.num_frames = num_frames

        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.f_temporal = nn.Linear(dim, dim + (self.focal_level+1), bias=bias)
        self.h_temporal = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.focal_layers_temporal = nn.ModuleList()
        
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                )
            )
            self.focal_layers_temporal.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1,
                    padding=kernel_size//2, bias=False),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        B, H, W, C = x.shape

        # Temporal modulation
        x_temporal = torch.clone(x)
        x_temporal = rearrange(x_temporal, '(b t) h w c -> (b h w) t c', t=self.num_frames)
        x_temporal = self.f_temporal(x_temporal).permute(0, 2, 1).contiguous()
        ctx_temporal, gates_temporal = torch.split(x_temporal, (C, self.focal_level+1), 1)

        ctx_all_temporal = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers_temporal[l](ctx_temporal)
            ctx_all_temporal = ctx_all_temporal + ctx*gates_temporal[:, l:l+1]
        ctx_global_temporal = self.act(ctx_temporal.mean(2, keepdim=True))
        ctx_all_temporal = ctx_all_temporal + ctx_global_temporal*gates_temporal[:,self.focal_level:]

        # Spatial modulation
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        # Combine temporal and spatial modulation
        modulator_temporal = self.h_temporal(ctx_all_temporal)
        modulator_temporal = rearrange(modulator_temporal, '(b h w) c t -> (b t) c h w', t=self.num_frames, h=H, w=W)
        modulator = self.h(ctx_all)

        x_out = q * modulator * modulator_temporal
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        return x_out

class FocalPhys(nn.Module):
    def __init__(self, frames=128):
        super().__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # Focal blocks
        self.focal1 = SpatioTemporalFocalModulation(dim=64, num_frames=frames)
        self.focal2 = SpatioTemporalFocalModulation(dim=64, num_frames=frames//2)
        self.focal3 = SpatioTemporalFocalModulation(dim=64, num_frames=frames//4)
        
        # Spatial pooling
        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.pool_spatiotemporal = nn.MaxPool3d((2, 2, 2), stride=2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d((frames, 1, 1))
        
        # Final prediction
        self.conv_final = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initial convolutions
        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x_visual64 = x.clone()
        
        # First focal block
        x = rearrange(x, 'b c t h w -> (b t) h w c')
        x = self.focal1(x)
        x = rearrange(x, '(b t) h w c -> b c t h w', b=batch_size)
        x = self.pool_spatiotemporal(x)
        x_visual32 = x.clone()
        
        # Second focal block
        x = rearrange(x, 'b c t h w -> (b t) h w c')
        x = self.focal2(x)
        x = rearrange(x, '(b t) h w c -> b c t h w', b=batch_size)
        x = self.pool_spatiotemporal(x)
        x_visual16 = x.clone()
        
        # Third focal block
        x = rearrange(x, 'b c t h w -> (b t) h w c')
        x = self.focal3(x)
        x = rearrange(x, '(b t) h w c -> b c t h w', b=batch_size)
        
        # Global pooling and final prediction
        x = self.global_pool(x)
        x = self.conv_final(x)
        
        rppg = x.view(batch_size, -1)
        
        return rppg, x_visual64, x_visual32, x_visual16

def build_transforms(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)    
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )        
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_transforms4display(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)    
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )  
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

model_urls = {
    "videofocalnet_tiny": "",
    "videofocalnet_small": "",
    "videofocalnet_base": "",
}

@register_model
def videofocalnet_tiny(pretrained=False, **kwargs):
    model = FocalPhys(frames=128)
    if pretrained:
        url = model_urls['videofocalnet_tiny']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def videofocalnet_small(pretrained=False, **kwargs):
    model = FocalPhys(frames=256)
    if pretrained:
        url = model_urls['videofocalnet_small']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def videofocalnet_base(pretrained=False, **kwargs):
    model = FocalPhys(frames=512)
    if pretrained:
        url = model_urls['videofocalnet_base']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == '__main__':
    print('test')