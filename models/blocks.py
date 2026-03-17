import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import random
from timm.models.layers import trunc_normal_, DropPath

'''
#Residual block
  
        x --------------------------------------------------------------------| + |

    ---------[conv1 ----- BN1 ---- Relu] ---------[conv2 ----- BN2]----(Per channel addition)------Relu-------> Out

'''

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, kernel_size=3):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.SiLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=kernel_size//2),
        )
        self.conv_skip = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2))
        self.SiLU = nn.SiLU()

    def forward(self, x):
        out = self.conv_block(x)
        skip = self.conv_skip(x)
        out = self.SiLU(out+skip)
        return out    


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)    
    


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextV2Block(nn.Module):

    def __init__(self, input_dim, output_dim, drop_path=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dwconv = nn.Conv2d(input_dim, input_dim, kernel_size=7, padding=3, groups=input_dim) #depthWise conv
        self.norm = LayerNorm(input_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(input_dim, 4*input_dim)

        self.act = nn.GELU()
        self.grn = GRN(4*input_dim)
        self.pwconv2 = nn.Linear(4*input_dim, output_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.conv_skip = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1))

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #N H W C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) #N, C, H, W
        input = input if self.input_dim == self.output_dim else self.conv_skip(input)
        x = input + self.drop_path(x)
        return x

class AffineRegressor(nn.Module):
    def __init__(self, patch_size, in_channel = 3):
        super(AffineRegressor, self).__init__()
        
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel*2, kernel_size=7, stride = 2, padding=3), #H/4, W/4
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.Conv2d(in_channel*2, in_channel*4, kernel_size=7, stride = 2, padding=3),  #H/8, W/8
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.Conv2d(in_channel*4, in_channel*8, kernel_size=5, stride = 2, padding=2), #H/16, W/16
                    nn.LeakyReLU(negative_slope = 0.2)
                    )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size//16 * patch_size//16 * in_channel*8, 128),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(32, 6),
        )

        self._init_identity_affine()
        
    def _init_identity_affine(self):
        nn.init.kaiming_normal_(self.fc[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc[1].bias)
        nn.init.kaiming_normal_(self.out[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.out[0].bias)
        self.out[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float))
        nn.init.zeros_(self.out[-1].weight)


    def forward(self, fixed, moving):
        B, C, H, W = moving.shape
        ref_encoder = self.fc(self.conv(fixed))
        mov_encoder = self.fc(self.conv(moving))

        concat = torch.cat((ref_encoder, mov_encoder), dim=1)
        theta = self.out(concat).view(-1, 2, 3)

        grid = F.affine_grid(theta, size = (B, C, H, W), align_corners=True)
        moving_affined = F.grid_sample(moving, grid, mode='bilinear',  padding_mode = 'zeros', align_corners=True)
        return  moving_affined  
    



#Deformable Alignment Network (From TDAN Super resolution Paper)    
from torchvision.ops import DeformConv2d


class AlignmentNet(nn.Module):
    def __init__(self, in_channel, out_channel, num_residual=3):
        super(AlignmentNet, self).__init__()

        self.conv_first = nn.Conv2d(in_channel, 64, 3, padding=1, bias=True)

        self.residual = make_layer(Res_Block, num_residual)

        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

        #Deformable
        def_kernel = 3
        group = 4

        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off1 = nn.Conv2d(64, 2*def_kernel*def_kernel*group, kernel_size=3, padding=1, bias=True)
        self.deform1 = DeformConvBlock(64, 64, def_kernel, stride=1, padding=2, dilation=2, groups=group)

        self.off2 = nn.Conv2d(64, 2*def_kernel*def_kernel*group, kernel_size=3, padding=1, bias=True)
        self.deform2 = DeformConvBlock(64, 64, def_kernel, stride=1, padding=2, dilation=2, groups=group)

        self.off3 = nn.Conv2d(64, 2*def_kernel*def_kernel*group, kernel_size=3, padding=1, bias=True)
        self.deform3 = DeformConvBlock(64, 64, def_kernel, stride=1, padding=2, dilation=2, groups=group)

        self.off = nn.Conv2d(64, 2*def_kernel*def_kernel*group, kernel_size=3, padding=1, bias=True)
        self.deform = DeformConvBlock(64, 64, def_kernel, stride=1, padding=2, dilation=2, groups=group)

        #self.recon = nn.Conv2d(64, out_channel, kernel_size=3, padding=1, bias=True)

    def align(self, Fref, Fmov):

        fea = torch.cat([Fref, Fmov], 1)
        fea = self.cr(fea)

        offset1 = self.off1(fea)
        fea = self.deform1(fea, offset1)

        offset2 = self.off2(fea)
        fea = self.deform2(fea, offset2)

        offset3 = self.off3(fea)
        fea = self.deform3(Fmov, offset3)

        offset4 = self.off(fea)
        aligned_feat = self.deform(fea, offset4)

        #aligned_feat = self.lrelu(self.recon(fea))

        return aligned_feat


    def forward(self, Fref, Fmov1, Fmov2):

        # Fref = self.residual(self.lrelu(self.conv_first(ref_feat)))
        # Fmov1 = self.residual(self.lrelu(self.conv_first(moving_feat1)))
        # Fmov2 = self.residual(self.lrelu(self.conv_first(moving_feat2)))

        moving_feat1_aligned = self.align(Fref, Fmov1)
        moving_feat2_aligned = self.align(Fref, Fmov2)

        return moving_feat1_aligned, moving_feat2_aligned

        

class DeformConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, dilation, groups, bias=True, use_mask=True):
        super(DeformConvBlock, self).__init__()

        self.use_mask=use_mask
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x, offset, mask=None):
        return self.dcn(x, offset, mask)    


def make_layer(block, num_layer):
    layers=[]
    for _ in range(num_layer):
        layers.append(block())

    return nn.Sequential(*layers)    


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding =1, bias=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.lrelu(feat)
        feat = self.conv2(feat)
        return feat+x    
    
##########
class DFF_block(nn.Module):
    def __init__(self, in_channel=64, feat_dim=128):
        super(DFF_block, self).__init__()  
        self.conv0 = nn.Conv2d(in_channel, feat_dim, kernel_size=3, padding=1)
        self.dconv1 = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, dilation=1, groups=feat_dim)
        self.dconv2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=2, dilation=2, groups=feat_dim)
        self.dconv3 = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=3, dilation=3, groups=feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(feat_dim, feat_dim//4, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(feat_dim//4, feat_dim, kernel_size=1),
                                nn.Sigmoid()
                                )
        self.conv_out = nn.Conv2d(feat_dim, in_channel, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        self.norm = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        feat = self.conv0(x)
        y1 = self.dconv1(feat)
        y2 = self.dconv2(feat)
        y3 = self.dconv3(feat)

        y = y1+y2+y3

        chn_wt = self.pool(y)
        chn_wt = self.fc(chn_wt)
        y = y*chn_wt

        out = self.conv_out(y)
        out = self.norm(self.act(out))+x
        return out
    
class Channel_attn(nn.Module):
    def __init__(self, num_channel):
        super(Channel_attn, self).__init__()
        self.channel = num_channel
        self.pool1 = nn.AdaptiveAvgPool2d(1) 

        self.fc = nn.Sequential(nn.Conv2d(num_channel, num_channel//4, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(num_channel//4, num_channel, kernel_size=1),
                                nn.Sigmoid()
                                )
    def forward(self, x):
        wt1 = self.pool1(x)
        wt1 = self.fc(wt1)
        return wt1    
           


#Fusion Block from AHDR Paper
# 
class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(DRDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dilation_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out
  

# Fusion Net
class Fusion_net(nn.Module):
    def __init__(self):
        super(Fusion_net, self).__init__()
        nFeat = 16

        self.conv2_0 = nn.Conv2d(nFeat*9, nFeat*3, kernel_size=3, padding=1)

        self.conv2_1 = DFF_block(in_channel=nFeat*3, feat_dim=nFeat*3)
        self.chann_attn1 = Channel_attn(nFeat*3)
        self.conv2_2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        
        # DRDBs 3
        self.RDB1 = DFF_block(in_channel=nFeat, feat_dim=nFeat*2)
        self.RDB2 = DFF_block(in_channel=nFeat, feat_dim=nFeat*2)
        self.RDB3 = DFF_block(in_channel=nFeat, feat_dim=nFeat*2)
        # feature fusion (GFF)

        self.chann_attn2 = Channel_attn(nFeat*3)
        self.GFF_3x3 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        # # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv 
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv3 = nn.Conv2d(nFeat//4, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        # self.shuf = nn.PixelShuffle(2)


    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):

        F_ = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9), 1)
        F_ = self.conv2_0(F_)

        F_0 = self.conv2_1(F_)
        chann_attn1 = self.chann_attn1(F_0)
        F_0 = F_0*chann_attn1
        F_0 = self.conv2_2(F_0)

        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        chann_attn2 = self.chann_attn2(FF)
        FF = FF*chann_attn2

        FdLF = self.GFF_3x3(FF)         
        # FGF = self.GFF_3x3(FdLF)
        FDF = FdLF+x2
        us = self.conv_up(FDF)
        #us = self.shuf(us)
        us = self.pixel_shuffle(us)
        output = self.conv3(us)
        output = nn.functional.sigmoid(output)

        return output
    




#Attention Block from: https://github.com/steven-tel/CEN-HDR/blob/main/models/modules/SCRAM.py

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import interpolate


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            #self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1] // 2))

    def forward(self, x):
        _, c, h, w = x.shape
        res = self.avgpool(x)
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand(-1, c // 2, h, w)
        return res


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, padding=1, dilation=dia_val))
            #self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        _, c, h, w = x.shape
        res = self.sa(x)
        res = interpolate(res, size=(h, w))
        res = res.expand((-1, c // 2, -1, -1))
        return res


class Model(nn.Module):

    def __init__(self, channel=64, reduction=3, dia_val=2):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.reduce = nn.Conv2d(in_channels=channel, out_channels=channel//2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        return weight    

    
