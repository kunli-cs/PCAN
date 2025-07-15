# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.model.weight_init import kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint

from mmaction.registry import MODELS
from .resnet3d import ResNet3d
from torch.nn import functional as F


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos



class Spatial_Attention(nn.Module):
    """ Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    空间窗口自我关注。
        它支持矩形窗口（包含正方形窗口）。
    参数：
        dim （int）： 输入通道数。
        idx （int）： 窗口的指数。(0/1)
        split_size（元组（int））： 空间窗口的高度和宽度。
        dim_out （int | 无）： 注意力输出的维度。默认值： 无
        num_heads (int)： 注意力头数。默认值：6
        attn_drop (float)： 注意力权重的下降率。默认值：0.0
        proj_drop（浮点数）： 输出的 dropout 比率。默认值：0.0
        qk_scale (float | None)： qk 比例： 覆盖 head_dim 的默认 qk 比例 ** -0.5（如果已设置
        position_bias (bool)： 动态相对位置偏置。默认值： 真

    """
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class Adaptive_Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """ Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
        
        dim（int）： 输入通道数。
        num_heads (int)： 注意头的数量。默认值： 6
        split_size（tuple(int)）： 空间窗口的高度和宽度： 空间窗口的高度和宽度。
        shift_size （元组（int））： 空间窗口的移动大小： 空间窗口的移动大小。
        qkv_bias（bool）： 如果为 True，则为查询、键、值添加可学习的偏置。默认值： 为 True
        qk_scale (float | None)： 覆盖默认的 head_dim ** -0.5 的 qk 比例（如果已设置）。
        drop (float)： 辍学率。默认值：0.0
        attn_drop (float)： 注意力下降率。默认值：0.0
        rg_idx（int）： 残差组 (RG) 的指数
        b_idx (int)： 每个 RG 中区块的序号      
    """
    def __init__(self, dim, num_heads=6, 
                 reso=64, split_size=[8,8], shift_size=[1,2], qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., rg_idx=0, b_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx  = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                Spatial_Attention(
                    dim//2, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])

        if (self.rg_idx % 2 == 0 and self.b_idx  > 0 and (self.b_idx  - 2) % 4 == 0) or (self.rg_idx % 2 != 0 and self.b_idx  % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # V without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1]) #default:8 ?
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size #default:0 ?
        pad_b = (max_split_size - H % max_split_size) % max_split_size #default:0 ?

        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        # 在分割通道 [C/2，C/2] 上的窗口-0 和窗口-1；对于方形窗口（如 8x8），窗口-0 和窗口-1 可以合并
        # 移位块：（0，4，8，...），（2，6，10，...），（0，4，8，...），（2，6，10，...），...
        if (self.rg_idx % 2 == 0 and self.b_idx  > 0 and (self.b_idx  - 2) % 4 == 0) or (self.rg_idx % 2 != 0 and self.b_idx  % 4 == 0):
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            # attention output
            attened_x = torch.cat([x1,x2], dim=2)

        else:
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # attention output
            attened_x = torch.cat([x1,x2], dim=2)
        
        # convolution output
        conv_x = self.dwconv(v)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x






class DeConvModule(BaseModule):
    """A deconv module that bundles deconv/norm/activation layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        stride (int | tuple[int]): Stride of the convolution.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input.
        bias (bool): Whether to add a learnable bias to the output.
            Defaults to False.
        with_bn (bool): Whether to add a BN layer. Defaults to True.
        with_relu (bool): Whether to add a ReLU layer. Defaults to True.
        in_channels （int）： 输入特征图中的通道数。
        out_channels (int)： 卷积产生的通道数。
        kernel_size (int | tuple[int])： 卷积核的大小。
        stride（int | tuple[int]）：卷积的步长： 卷积的步长。
        padding (int | tuple[int]): 填充： 输入两边的零填充。
            输入。
        bias（bool）：偏置： 是否为输出添加可学习的偏置。
            默认为假。
        with_bn（bool）：是否添加 BN 层： 是否添加 BN 层。默认为 True。
        with_relu (bool): 是否添加 ReLU 层： 是否添加 ReLU 层。默认为 True。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: Union[int, Tuple[int]] = (1, 1, 1),
                 padding: Union[int, Tuple[int]] = 0,
                 bias: bool = False,
                 with_bn: bool = True,
                 with_relu: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.with_relu = with_relu

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        # x should be a 5-d tensor
        assert len(x.shape) == 5
        N, C, T, H, W = x.shape
        out_shape = (N, self.out_channels, self.stride[0] * T,
                     self.stride[1] * H, self.stride[2] * W)
        x = self.conv(x, output_size=out_shape)
        if self.with_bn:
            x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Defaults to False.
        lateral_inv (bool): Whether to use deconv to upscale the time
            dimension of features from another pathway. Defaults to False.
        lateral_norm (bool): Determines whether to enable the lateral norm
            in lateral layers. Defaults to False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Defaults to 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Defaults to 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Defaults to 5.
        lateral_infl (int): The ratio of the inflated channels.
            Defaults to 2.
        lateral_activate (list[int]): Flags for activating the lateral
            connection. Defaults to ``[1, 1, 1, 1]``.
    """

    def __init__(self,
                 lateral: bool = False,
                 lateral_inv: bool = False,
                 lateral_norm: bool = False,
                 speed_ratio: int = 8,
                 channel_ratio: int = 8,
                 fusion_kernel: int = 5,
                 lateral_infl: int = 2,
                 lateral_activate: List[int] = [1, 1, 1, 1],
                 **kwargs) -> None:
        self.lateral = lateral
        self.lateral_inv = lateral_inv
        self.lateral_norm = lateral_norm
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        self.lateral_infl = lateral_infl
        self.lateral_activate = lateral_activate
        self._calculate_lateral_inplanes(kwargs)


        super().__init__(**kwargs)
        self.inplanes = self.base_channels
        if self.lateral and self.lateral_activate[0] == 1:
            if self.lateral_inv:
                self.conv1_lateral = DeConvModule(
                    self.inplanes * self.channel_ratio,
                    self.inplanes * self.channel_ratio // lateral_infl,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    with_bn=True,
                    with_relu=True)
            else:
                self.conv1_lateral = ConvModule(
                    self.inplanes // self.channel_ratio,
                    self.inplanes * lateral_infl // self.channel_ratio,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg if self.lateral_norm else None,
                    act_cfg=self.act_cfg if self.lateral_norm else None)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1 \
                    and self.lateral_activate[i + 1]:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                if self.lateral_inv:
                    conv_module = DeConvModule(
                        self.inplanes * self.channel_ratio,
                        self.inplanes * self.channel_ratio // lateral_infl,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        with_bn=True,
                        with_relu=True)
                else:
                    conv_module = ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * lateral_infl // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg if self.lateral_norm else None,
                        act_cfg=self.act_cfg if self.lateral_norm else None)
                setattr(self, lateral_name, conv_module)
                self.lateral_connections.append(lateral_name)

    def _calculate_lateral_inplanes(self, kwargs):
        """Calculate inplanes for lateral connection."""
        depth = kwargs.get('depth', 50)
        expansion = 1 if depth < 50 else 4
        base_channels = kwargs.get('base_channels', 64)
        lateral_inplanes = []
        for i in range(kwargs.get('num_stages', 4)):
            if expansion % 2 == 0:
                planes = base_channels * (2 ** i) * \
                         ((expansion // 2) ** (i > 0))
            else:
                planes = base_channels * (2**i) // (2**(i > 0))
            if self.lateral and self.lateral_activate[i]:
                if self.lateral_inv:
                    lateral_inplane = planes * \
                                      self.channel_ratio // self.lateral_infl
                else:
                    lateral_inplane = planes * \
                                      self.lateral_infl // self.channel_ratio
            else:
                lateral_inplane = 0
            lateral_inplanes.append(lateral_inplane)
        self.lateral_inplanes = lateral_inplanes

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the ``lateral_connection`` part should
        not be inflated from 2d weights.

        Args:
            logger (MMLogger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    self._inflate_conv_params(module.conv, state_dict_r2d,
                                              original_conv_name,
                                              inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def _inflate_conv_params(self, conv3d: nn.Module,
                             state_dict_2d: OrderedDict, module_name_2d: str,
                             inflated_param_names: List[str]) -> None:
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]

        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            # Expand parameters by concat extra channels
            conv2d_weight = torch.cat(
                (conv2d_weight,
                 torch.zeros(pad_shape).type_as(conv2d_weight).to(
                     conv2d_weight.device)),
                dim=1)

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        # Override the init_weights of i3d
        super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)


pathway_cfg = {
    'resnet3d': ResNet3dPathway,
    # TODO: BNInceptionPathway
}


def build_pathway(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build pathway.

    Args:
        cfg (dict): cfg should contain:
            - type (str): identify backbone type.

    Returns:
        nn.Module: Created pathway.
    """
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    if pathway_type not in pathway_cfg:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')

    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@MODELS.register_module()
class ResNet3dSlowFast(BaseModule):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride ``resample_rate``
            on input frames. The actual resample rate is calculated by
            multipling the ``interval`` in ``SampleFrames`` in the
            pipeline with ``resample_rate``, equivalent to the :math:`\\tau`
            in the paper, i.e. it processes only one out of
            ``resample_rate * interval`` frames. Defaults to 8.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Defaults to 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Defaults to 8.
        slow_pathway (dict): Configuration of slow branch. Defaults to
            ``dict(type='resnet3d', depth=50, pretrained=None, lateral=True,
            conv1_kernel=(1, 7, 7), conv1_stride_t=1, pool1_stride_t=1,
            inflate=(0, 0, 1, 1))``.
        fast_pathway (dict): Configuration of fast branch. Defaults to
            ``dict(type='resnet3d', depth=50, pretrained=None, lateral=False,
            base_channels=8, conv1_kernel=(5, 7, 7), conv1_stride_t=1,
            pool1_stride_t=1)``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 resample_rate: int = 8,
                 speed_ratio: int = 8,
                 channel_ratio: int = 8,
                 slow_pathway: Dict = dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway: Dict = dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio
            slow_pathway['channel_ratio'] = channel_ratio

        self.slow_path = build_pathway(slow_pathway)
        self.fast_path = build_pathway(fast_pathway)

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.fast_path.init_weights()
            self.slow_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor) -> tuple:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples
                extracted by the backbone.
        """
        x_slow = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0,
                          1.0))
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            if (i != len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral):
                # No fusion needed in the final stage
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        out = (x_slow, x_fast)

        return out
