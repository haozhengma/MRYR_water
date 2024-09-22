import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
# from block import fusions
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from resnet import ResNet

import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from backbones import MiT, ResNet, PVTv2, ResNetD
from backbones.layers import trunc_normal_
from heads import SegFormerHead, FaPNHead, SFHead, UPerHead, FPNHead
# from heads import SFHead
# from Deformable_ConvNet import DeformConv2D
# from baseline_models import FCN, deeplabv3
# from Unet import UNet


class ConvAtt(nn.Module):
    def __init__(self, out_C):
        super().__init__()
        self.kernel_size = 1
        self.d_model = out_C

        # self.head_instance = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.PPM = PPM(self.n_class*2, self.n_class)
        self.depth_wise_conv = nn.Conv2d(self.d_model, self.d_model,
                                         kernel_size=self.kernel_size,
                                         groups=self.d_model,
                                         padding=(self.kernel_size - 1) // 2)
        # Activation after depth-wise convolution
        self.act1 = nn.GELU()
        # Normalization after depth-wise convolution
        self.norm1 = nn.BatchNorm2d(self.d_model)

        # Point-wise convolution is a $1 \times 1$ convolution.
        # i.e. a linear transformation of patch embeddings
        self.point_wise_conv = nn.Conv2d(self.d_model, 3, kernel_size=1)
        # Activation after point-wise convolution
        # self.act2 = nn.GELU()
        # # Normalization after point-wise convolution
        # self.norm2 = nn.BatchNorm2d(3)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        # print(x.shape)
        # Depth-wise convolution, activation and normalization
        x = self.depth_wise_conv(x)
        x = self.act1(x)
        x = self.norm1(x)

        # Add residual connection
        x += residual

        # Point-wise convolution, activation and normalization
        x = self.point_wise_conv(x)
        # x = self.act2(x)
        out = self.norm2(x)

        return out


class ConvAtt_Block(nn.Module):
    def __init__(self, out_C):
        super().__init__()
        # self.kernel_size = 1
        self.d_model = out_C

        self.ConvAtt1 = ConvAtt(self.d_model)
        self.ConvAtt2 = ConvAtt(self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        # print(x.shape)
        # Depth-wise convolution, activation and normalization
        x = self.ConvAtt1(x)

        # Add residual connection
        # x += residual

        # Point-wise convolution, activation and normalization
        out = self.ConvAtt2(x) + residual

        return out


class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


segformer_settings = {
    'B0': 256,        # head_dim
    'B1': 256,
    'B2': 768,
    'B3': 768,
    'B4': 768,
    'B5': 768
}


class SegFormer(nn.Module):
    def __init__(self, variant: str = 'B0', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = MiT(variant)
        self.decode_head = SegFormerHead(self.backbone.embed_dims, segformer_settings[variant], num_classes)
        # self.decode_head = FaPNHead([64, 128, 320, 512], 128, 19)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        feature = self.backbone(x)
        # for i in y:
        #     print(i.shape)
        y = self.decode_head(feature)   # 4x reduction in image size
        # y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return feature, y


class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class Attention(nn.Module):
    """Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input.
    Modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, 'dim should be divisible by head_dim'
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1) # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels: int = 256, key_channels: int = 128, scale: int = 1):
        super(ObjectAttentionBlock, self).__init__()
        self.key_channels = key_channels
        self.scale = scale
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

        self.to_q = nn.Sequential(
            ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0),
            ConvReLU(key_channels, out_channels=key_channels, kernel_size=1, padding=0)
        )
        self.to_k = nn.Sequential(
            ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0),
            ConvReLU(key_channels, out_channels=key_channels, kernel_size=1, padding=0)
        )
        self.to_v = ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0)
        self.f_up = ConvReLU(key_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, features, context):
        n, c, h, w = features.shape
        if self.scale == 1:
            features = self.pool(features)

        query = self.to_q(features).view(n, -1, c)
        key = self.to_k(context).view(n, c, -1)
        value = self.to_v(context).view(n, -1, c)

        sim_map = torch.matmul(query, key)
        sim_map *= self.key_channels ** -0.5
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map.squeeze(), value.squeeze()).view(n, c, h, w)

        context = self.f_up(context)
        context = self.up(context)
        return context


class SpatialOCR(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.5):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale)
        self.conv_bn_dropout = nn.Sequential(
            ConvReLU(2 * in_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, context):
        context = self.object_context_block(feats, context)
        output = self.conv_bn_dropout(torch.cat([context, feats], dim=1))
        return output


class SpatialGather(nn.Module):
    def __init__(self, n_classes: int = 100, scale: int = 1):
        super(SpatialGather, self).__init__()
        self.cls_num = n_classes
        self.scale = scale

    def forward(self, features, probs):
        n, k, _, _ = probs.shape
        _, c, _, _ = features.shape
        probs = probs.view(n, k, -1)
        features = features.view(n, -1, c)
        probs = torch.softmax(self.scale * probs, dim=-1)
        ocr_context = torch.matmul(probs, features)
        return ocr_context.view(n, k, c, 1)


class csa_layer(nn.Module):
    def __init__(self, channel, num_layers, k_size=3):
        super(csa_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)  # padding=(k_size - 1) // 2
        self.sigmoid = nn.Sigmoid()
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channel,
            nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # self.mlp = torch.nn.Linear(256, 256)
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 256)
        # )

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # y_max = self.max_pool(x)
        # print(y.shape)

        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.transformer_encoder(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y_max = self.transformer_encoder(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y_local = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print(y.shape)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y_local = self.sigmoid(y_local)
        # print(y.shape)

        return x * y.expand_as(x)   # * y_local.expand_as(x)


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class ConvReLU(nn.Module):
    """docstring for ConvReLU"""

    def __init__(self, channels: int = 2048, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1,
                 out_channels=None):
        super(ConvReLU, self).__init__()
        if out_channels is None:
            self.out_channels = channels
        else:
            self.out_channels = out_channels

        self.conv = nn.Conv2d(channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              bias=False)
        self.bn = nn.GroupNorm(self.out_channels, self.out_channels)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SegFormer_Att_New(nn.Module):
    def __init__(self, n_class):
        super(SegFormer_Att_New,self).__init__()
        self.n_class=n_class
        # self.out_channels = 64
        self.head = 1
        self.dim = 256
        self.channels = n_class
        self.semantic_img_model = SegFormer('B2', 150)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_img_model.decode_head = FaPNHead([64, 128, 320, 512], self.dim, self.channels)  # SegFormerHead([64, 128, 320, 512], 768, self.n_class)

    def forward(self, h_rs):

        features, out = self.semantic_img_model(h_rs)


        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Adaformer_teacher(nn.Module):
    def __init__(self, n_class):
        super(Adaformer_teacher,self).__init__()
        self.n_class=n_class
        # self.out_channels = 64
        self.head = 1
        self.dim = 256
        self.channels = n_class
        self.semantic_img_model = SegFormer('B2', 150)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_img_model.decode_head = FaPNHead([64, 128, 320, 512], self.dim, self.channels)  # SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.soft_object_regions = nn.Sequential(
        #     nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(self.channels, self.channels),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
        # )
        # self.pixel_representations = ConvReLU(self.channels, out_channels=self.channels, kernel_size=3)
        # self.object_region_representations = SpatialGather(self.n_class)
        # self.object_contextual_representations = SpatialOCR(
        #     in_channels=self.channels,
        #     key_channels=self.channels,
        #     out_channels=self.channels,
        #     scale=1,
        #     dropout=0,
        # )
        self.csa_layer = csa_layer(256, 1)
        # self.ChannelAttentionModule = ChannelAttentionModule()
        # self.augmented_representation = nn.Conv2d(self.channels, self.n_class, kernel_size=1, padding=0)
        self.augmented_representation = nn.Conv2d(256, self.channels, kernel_size=1, padding=0)

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        # feature_list = []
        features, out = self.semantic_img_model(h_rs)
        out = self.csa_layer(out) + out
        out = self.augmented_representation(out)

        # out = self.ChannelAttentionModule(out)
        # context = self.object_region_representations(out, out)
        # # print(context.shape)
        # features = self.object_contextual_representations(out, context)
        # # print(features.shape)
        # out = self.augmented_representation(features) + out
        # print(out.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class ConvBlock(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(True)
        )


class Adaformer_student(nn.Module):
    def __init__(self, n_class):
        super(Adaformer_student,self).__init__()
        self.n_class=n_class
        # self.out_channels = 64
        self.head = 1
        self.dim = 256
        self.channels = n_class
        self.semantic_img_model = SegFormer('B2', 150)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_img_model.decode_head = FaPNHead([64, 128, 320, 512], self.dim, self.channels)  # SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.backbone.patch_embed1.proj = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.csa_layer = csa_layer(256, 1)
        # self.ChannelAttentionModule = ChannelAttentionModule()
        # self.augmented_representation = nn.Conv2d(self.channels, self.n_class, kernel_size=1, padding=0)
        self.augmented_representation = nn.Conv2d(256, 16, kernel_size=1, padding=0)
        self.conv_fc = nn.Conv2d(20, self.channels, kernel_size=1, padding=0)
        self.knowguide_refinement = nn.Sequential(
                    ConvModule(1, 128, 3, 2, 1),  # 13
                    ConvModule(128, 4, 3, 1, 1),
                    ConvModule(4, 4, 3, 2, 1))
                    # MobileViTBlock(32, 2, 64, 3, (2, 2), int(32*2)),
                    # ConvModule(128, 256, 3, 1, 1),
                    # ConvModule(128, 256, 3, 1, 1))
                    # ConvModule(128, 128, 3, 1, 1),
                    # MobileViTBlock(64, 2, 128, 3, (2, 2), int(64*2))
                    # ConvModule(128, 256, 3, 1, 1))

    def forward(self, h_rs, sam):
        # print(h_rs.shape, x_floor.shape)
        # h_rs = torch.cat([h_rs, sam], 1)
        # feature_list = []
        features, out = self.semantic_img_model(h_rs)
        sam_out = self.knowguide_refinement(sam)

        # print(out.shape, sam_out.shape)
        # out = torch.cat([out, sam_out], 1)
        out = self.csa_layer(out) + out
        out = self.augmented_representation(out)
        out = torch.cat([out, sam_out], 1)
        out = self.conv_fc(out)

        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class AdaFormer_Fuse(nn.Module):
    def __init__(self, n_class):
        super(AdaFormer_Fuse,self).__init__()
        self.n_class=n_class
        # self.out_channels = 64
        self.head = 1
        self.dim = 256
        self.channels = n_class
        self.img_model = SegFormer('B2', 150)
        self.img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        # self.img_model = list(img_model.backbone)
        self.img_model.decode_head = FaPNHead([64, 128, 320, 512], self.dim, self.channels)

        # self.fuse_decode_head = FaPNHead([96, 128, 320, 512], self.dim, self.channels)

        # self.conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU(inplace=True)
        self.csa_layer = csa_layer(512, 1)
        # self.relu = nn.LeakyReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(8, stride=1)

        self.dove_model = SegFormer('B2', 150)
        self.dove_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.dove_model.decode_head = FaPNHead([64, 128, 320, 512], self.dim // 2, self.channels)

        # print(self.dove_model)
        self.upsample0 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.upsample1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(320, 64, 3, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(512, 64, 3, stride=2, padding=1)  # nn.ConvTranspose2d(512, 32, 3, stride=2, padding=1)

        self.fuse_decode_head = FaPNHead([96+32, 160+32, 352+32, 544+32], self.dim // 2, self.channels)

        self.dove_model.backbone.patch_embed1.proj = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.augmented_representation = nn.Conv2d(512, 4, kernel_size=1, padding=0)
    def forward(self, h_rs, dove):

        # x_dove = self.ChannelAttentionModule(x_dove)
        # x_dove = self.avgpool(x_dove)
        # print(h_rs.shape, x_floor.shape)
        # print(x_dove.shape)
        # dove = F.interpolate(dove, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # mmdata = torch.cat([h_rs, dove[:, 6, :, :].unsqueeze(1), dove[:, 7, :, :].unsqueeze(1)], 1)
        # feature_list = []
        # print(mmdata.shape)
        features, out = self.img_model(h_rs)
        features_dove, out_dove = self.dove_model(dove)

        # for i in features:
        #     print(i.shape)
        # features_dove, out_dove = self.dove_model(dove)


        features_dove0 = self.upsample0(features_dove[0], output_size=features[0].size())
        features_dove1 = self.upsample1(features_dove[1], output_size=features[1].size())
        features_dove2 = self.upsample2(features_dove[2], output_size=features[2].size())
        features_dove3 = self.upsample3(features_dove[3], output_size=features[3].size())


        # features_dove0 = F.interpolate(features_dove[0], size=features[0].shape[-2:], mode='bilinear', align_corners=False)
        # features_dove1 = F.interpolate(features_dove[1], size=features[1].shape[-2:], mode='bilinear', align_corners=False)
        # features_dove2 = F.interpolate(features_dove[2], size=features[2].shape[-2:], mode='bilinear', align_corners=False)
        # features_dove3 = F.interpolate(features_dove[3], size=features[3].shape[-2:], mode='bilinear', align_corners=False)

        feature_fuse = [torch.cat([features[0], features_dove0], 1), torch.cat([features[1], features_dove1], 1), torch.cat([features[2], features_dove2], 1), torch.cat([features[3], features_dove3], 1)]
        # for i in feature_fuse:
        #     print(i.shape)
        # for i in features_dove:
        #     print(i.shape)
        out_fuse = self.fuse_decode_head(feature_fuse)

        out_dove = F.interpolate(out_dove, size=out.shape[-2:], mode='bilinear', align_corners=False)

        out_cat = torch.cat([out, out_dove, out_fuse], 1)
        out_cat = self.csa_layer(out_cat) + out_cat
        # print(out_cat.shape)
        out_cat = self.augmented_representation(out_cat)

        out = F.interpolate(out_cat, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # out = F.interpolate(out_fuse, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class AdaFormer_Dove(nn.Module):
    def __init__(self, n_class):
        super(AdaFormer_Dove,self).__init__()
        self.n_class=n_class
        # self.out_channels = 64
        self.head = 1
        self.dim = 256
        self.channels = n_class

        self.dove_model = SegFormer('B2', 150)
        self.dove_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.dove_model.decode_head = FaPNHead([64, 128, 320, 512], self.dim, self.channels)
        self.dove_model.backbone.patch_embed1.proj = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.augmented_representation = nn.Conv2d(self.dim, self.channels, kernel_size=1, padding=0)
        self.csa_layer = csa_layer(256, 1)

    def forward(self, h_rs, dove):

        features_dove, out_dove = self.dove_model(dove)
        out_dove = out_dove + self.csa_layer(out_dove)
        out_dove = self.augmented_representation(out_dove)

        out = F.interpolate(out_dove, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # out = F.interpolate(out_fuse, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class AdaFormer_FPN(nn.Module):
    def __init__(self, n_class):
        super(AdaFormer_FPN,self).__init__()
        self.n_class=n_class
        # self.out_channels = 64
        self.head = 1
        self.dim = 512
        self.semantic_img_model = SegFormer('B2', 150)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_img_model.decode_head = FPNHead([64, 128, 320, 512], self.dim, self.n_class*self.head)  # SegFormerHead([64, 128, 320, 512], 768, self.n_class)

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        # feature_list = []
        features, out = self.semantic_img_model(h_rs)
        # print(out.shape)
        # for feature in features:
        #     print(feature.shape)

        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Segformer_baseline(nn.Module):
    def __init__(self, n_class):
        super(Segformer_baseline, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B1', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('backbones/segformer.b1.ade.pth', map_location='cpu'))

        # 修改模型的第一层
        self.semantic_img_model.backbone.patch_embed1.proj = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            # nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features, out = self.semantic_img_model(h_rs)

        out = self.conv_block_fc(out)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Segformer_glh(nn.Module):
    def __init__(self, n_class):
        super(Segformer_glh, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B1', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('backbones/segformer.b1.ade.pth', map_location='cpu'))

        # 修改模型的第一层
        # self.semantic_img_model.backbone.patch_embed1.proj = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            # nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features, out = self.semantic_img_model(h_rs)

        out = self.conv_block_fc(out)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Segformer_from_gsw(nn.Module):
    def __init__(self, n_class):
        super(Segformer_from_gsw, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        # self.semantic_img_model = SegFormer('B2', self.out_channels)
        # # 修改模型的第一层
        # self.semantic_img_model.backbone.patch_embed1.proj = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(4, 4),
        #                                                                padding=(3, 3))
        # self.semantic_img_model.load_state_dict(torch.load(r'I:\demo\pytorch\WaterSegment\model\240521\CE-Segformer-gsw-2000-2020-4.pth', map_location='cpu'))
        premodel = torch.load(r'I:\demo\pytorch\WaterSegment\model\240521\CE-Segformer-gsw-2000-2020-4.pth', map_location='cpu')
        self.semantic_img_model = premodel.semantic_img_model
        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            # nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features, out = self.semantic_img_model(h_rs)

        out = self.conv_block_fc(out)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class SegFormer_Dove(nn.Module):
    def __init__(self, n_class):
        super(SegFormer_Dove, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_img_model.backbone.patch_embed1.proj = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        x = np.zeros((8, 8, 512, 512))
        features, out = self.semantic_img_model(h_rs)

        out = self.conv_block_fc(out)
        # print(features.shape)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out


class SegFormer_Fuse(nn.Module):

    def __init__(self, n_class):
        super(SegFormer_Fuse, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))

        self.semantic_dove_model = SegFormer('B2', self.out_channels)
        self.semantic_dove_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_dove_model.backbone.patch_embed1.proj = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            # nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(150*2, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, h_rs, x):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features, out = self.semantic_img_model(h_rs)
        features_dove, out_dove = self.semantic_dove_model(x)

        out_dove = F.interpolate(out_dove, size=out.shape[-2:], mode='bilinear', align_corners=False)

        fuse = torch.cat([out, out_dove], 1)

        # out = self.fc(fuse)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)

        out = self.conv_block_fc(fuse)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class AdaFormer_FPN_Fuse(nn.Module):
    def __init__(self, n_class):
        super(AdaFormer_FPN_Fuse, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_img_model.decode_head = FPNHead([64, 128, 320, 512], 256, 64)  # SegFormerHead([64, 128, 320, 512], 768, self.n_class)


        self.semantic_dove_model = SegFormer('B2', self.out_channels)
        self.semantic_dove_model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_dove_model.backbone.patch_embed1.proj = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.semantic_dove_model.decode_head = FPNHead([64, 128, 320, 512], 256, 64)  # SegFormerHead([64, 128, 320, 512], 768, self.n_class)


        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            # nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64*2, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, h_rs, x):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features, out = self.semantic_img_model(h_rs)
        features_dove, out_dove = self.semantic_dove_model(x)

        out_dove = F.interpolate(out_dove, size=out.shape[-2:], mode='bilinear', align_corners=False)

        fuse = torch.cat([out, out_dove], 1)

        # out = self.fc(fuse)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)

        out = self.conv_block_fc(fuse)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Segformer_b3(nn.Module):
    def __init__(self, n_class):
        super(Segformer_b3, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B3', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('segformer.b3.ade.pth', map_location='cpu'))
        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features, out = self.semantic_img_model(h_rs)

        out = self.conv_block_fc(out)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

# class FPNHead(nn.Module):
#     """Panoptic Feature Pyramid Networks
#     https://arxiv.org/abs/1901.02446
#     """
#     def __init__(self, in_channels, channel=128, num_classes=19):
#         super().__init__()
#         self.lateral_convs = nn.ModuleList([])
#         self.output_convs = nn.ModuleList([])
#
#         for ch in in_channels[::-1]:
#             self.lateral_convs.append(ConvModule(ch, channel, 1))
#             self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))
#
#         self.conv_seg = nn.Conv2d(channel, num_classes, 1)
#         self.dropout = nn.Dropout2d(0.1)
#
#     def forward(self, features) -> Tensor:
#         features = features[::-1]
#         out = self.lateral_convs[0](features[0])
#
#         for i in range(1, len(features)):
#             out = F.interpolate(out, scale_factor=2.0, mode='nearest')
#             out = out + self.lateral_convs[i](features[i])
#             out = self.output_convs[i](out)
#         out = self.conv_seg(self.dropout(out))
#         return out


if __name__ == '__main__':
    # FCN = SegFormer_Fuse(4)
    # FCN = FCN.to('cuda')
    # # summary(model, (3, 512, 512), device="cpu")
    # images = torch.randn(2, 3, 1024, 1024)
    # images = images.to('cuda')
    # # mmdata = mmdata.clone().detach().float()
    # images = images.clone().detach().float()
    # dove = torch.randn(2, 8, 256, 256)
    # dove = dove.to('cuda')
    # # mmdata = mmdata.clone().detach().float()
    # dove = dove.clone().detach().float()
    # out = FCN(images, dove)
    # print(out.shape)
    model1 = torch.load(r'I:\demo\pytorch\WaterSegment\model\240521\CE-Segformer-gsw-2000-2020-4.pth', map_location='cpu')
    print(model1)
    model2 = Segformer_from_gsw(5)
    # delattr(model, 'conv_block_fc')
    print(model2)
    # model = SegFormer('B2', 150)
    # model.load_state_dict(torch.load('segformer.b2.ade.pth', map_location='cpu'))
    # model.backbone.patch_embed1.proj = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
    # print(model)
