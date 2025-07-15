# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.logging import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
import torch.nn.functional as F
from mmaction.registry import MODELS
from .resnet3d_slowfast import ResNet3dPathway

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(MultiHeadCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        assert self.head_dim * \
            num_heads == out_channels, "out_channels must be divisible by num_heads"
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.fc = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.tensor(
            self.head_dim, dtype=torch.float32))

    def forward(self, x, y):
        # x: (N, T, C)
        # y: (N, T, C)
        N, T, C = x.shape
        # q: (N, T, C)
        q = self.q_proj(x)
        # k: (N, T, C)
        k = self.k_proj(y)
        # v: (N, T, C)
        v = self.v_proj(y)
        # q: (N, H, T, D)
        q = q.reshape(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k: (N, H, T, D)
        k = k.reshape(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v: (N, H, T, D)
        v = v.reshape(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # attn: (N, H, T, T)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # x: (N, H, T, D)
        x = attn @ v
        # x: (N, T, C)
        x = x.transpose(1, 2).reshape(N, T, self.out_channels)
        # x: (N, T, C)
        x = self.fc(x)
        return x


class FeatureInteraction(nn.Module):
    def __init__(self, rgb_channels=512, skt_channels=128, hidden_channels=128, num_heads=8):
        super(FeatureInteraction, self).__init__()
        self.rgb_channels = rgb_channels
        self.skt_channels = skt_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.avgpooling = nn.AdaptiveMaxPool3d((8, 1, 1))

        self.conv1 = nn.Conv3d(rgb_channels, hidden_channels,
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(skt_channels, hidden_channels, kernel_size=(
            4, 1, 1), stride=(4, 1, 1), padding=0)
        self.conv4 = nn.Conv3d(hidden_channels, rgb_channels,
                               kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.ConvTranspose3d(
            hidden_channels, skt_channels, kernel_size=(4, 1, 1), stride=(4, 1, 1))
        self.rgb_cross_attn = MultiHeadCrossAttention(
            hidden_channels, hidden_channels, num_heads)
        self.skt_cross_attn = MultiHeadCrossAttention(
            hidden_channels, hidden_channels, num_heads)

    def forward(self, rgb, skt):
        # rgb: (N, C1, T, H, W)
        # skt: (N, C2, T, H, W)
        origin_rgb = rgb.clone()
        origin_skt = skt.clone()
        rgb = self.conv1(rgb)  # N*128*8*1*1
        rgb = self.avgpooling(rgb)  # N*128*8*1*1
        skt = self.conv2(skt)  # N*128*8*1*1
        skt = self.avgpooling(skt)

        N, C1, T, H, W = rgb.shape
        N, C2, T, H, W = skt.shape
        # rgb: (N, T, C1)
        rgb = rgb.reshape(N, C1, -1).permute(0, 2, 1)
        # skt: (N, T, C2)
        skt = skt.reshape(N, C2, -1).permute(0, 2, 1)
        # rgb: (N, T, C3)
        rgb = self.rgb_cross_attn(rgb, skt)
        # skt: (N, T, C3)
        skt = self.skt_cross_attn(skt, rgb)

        # attention map 6,128,8,1,1
        rgb = rgb.permute(0, 2, 1).contiguous().unsqueeze(-1).unsqueeze(-1)
        # attention map 6,128,8,1,1
        skt = skt.permute(0, 2, 1).contiguous().unsqueeze(-1).unsqueeze(-1)
        rgb = self.conv4(rgb)  # N*512*8*28*28
        rgb = origin_rgb + rgb * origin_rgb
        skt = self.conv5(skt)  # N*128*32*28*28
        skt = origin_skt+skt * origin_skt
        # fusion: (N, T, 2 * C3)
        # fusion = torch.cat([rgb, skt], dim=-1)
        return rgb, skt


@MODELS.register_module()
class RGBPoseConv3D(BaseModule):
    """RGBPoseConv3D backbone.

    Args:
        pretrained (str): The file path to a pretrained model.
            Defaults to None.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Defaults to 4.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Defaults to 4.
        rgb_detach (bool): Whether to detach the gradients from the pose path.
            Defaults to False.
        pose_detach (bool): Whether to detach the gradients from the rgb path.
            Defaults to False.
        rgb_drop_path (float): The drop rate for dropping the features from
            the pose path. Defaults to 0.
        pose_drop_path (float): The drop rate for dropping the features from
            the rgb path. Defaults to 0.
        rgb_pathway (dict): Configuration of rgb branch. Defaults to
            ``dict(num_stages=4, lateral=True, lateral_infl=1,
            lateral_activate=(0, 0, 1, 1), fusion_kernel=7, base_channels=64,
            conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1), with_pool2=False)``.
        pose_pathway (dict): Configuration of pose branch. Defaults to
            ``dict(num_stages=3, stage_blocks=(4, 6, 3), lateral=True,
            lateral_inv=True, lateral_infl=16, lateral_activate=(0, 1, 1),
            fusion_kernel=7, in_channels=17, base_channels=32,
            out_indices=(2, ), conv1_kernel=(1, 7, 7), conv1_stride_s=1,
            conv1_stride_t=1, pool1_stride_s=1, pool1_stride_t=1,
            inflate=(0, 1, 1), spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 1), with_pool2=False)``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 speed_ratio: int = 4,
                 channel_ratio: int = 4,
                 rgb_detach: bool = False,
                 pose_detach: bool = False,
                 rgb_drop_path: float = 0,
                 pose_drop_path: float = 0,
                 rgb_pathway: Dict = dict(
                     num_stages=4,
                     lateral=True,
                     lateral_infl=1,
                     lateral_activate=(0, 0, 1, 1),
                     fusion_kernel=7,
                     base_channels=64,
                     conv1_kernel=(1, 7, 7),
                     inflate=(0, 0, 1, 1),
                     with_pool2=False),
                 pose_pathway: Dict = dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     fusion_kernel=7,
                     in_channels=17,
                     base_channels=32,
                     out_indices=(2, ),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_s=1,
                     conv1_stride_t=1,
                     pool1_stride_s=1,
                     pool1_stride_t=1,
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1),
                     dilations=(1, 1, 1),
                     with_pool2=False),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if rgb_pathway['lateral']:
            rgb_pathway['speed_ratio'] = speed_ratio
            rgb_pathway['channel_ratio'] = channel_ratio

        if pose_pathway['lateral']:
            pose_pathway['speed_ratio'] = speed_ratio
            pose_pathway['channel_ratio'] = channel_ratio

        self.rgb_path = ResNet3dPathway(**rgb_pathway)
        self.pose_path = ResNet3dPathway(**pose_pathway)
        self.rgb_detach = rgb_detach
        self.pose_detach = pose_detach
        assert 0 <= rgb_drop_path <= 1
        assert 0 <= pose_drop_path <= 1
        self.rgb_drop_path = rgb_drop_path
        self.pose_drop_path = pose_drop_path

        self.featuredifference2 = FeatureInteraction(
            512, 128, hidden_channels=128)

        self.featuredifference3 = FeatureInteraction(
            1024, 256, hidden_channels=256)

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.rgb_path.init_weights()
            self.pose_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, imgs: torch.Tensor, heatmap_imgs: torch.Tensor, gt: torch.Tensor, gt_coarse: torch.Tensor) -> tuple:
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input data.
            heatmap_imgs (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        if self.training:
            rgb_drop_path = torch.rand(1) < self.rgb_drop_path
            pose_drop_path = torch.rand(1) < self.pose_drop_path
        else:
            rgb_drop_path, pose_drop_path = False, False
        # We assume base_channel for RGB and Pose are 64 and 32.
        x_rgb = self.rgb_path.conv1(imgs)
        x_rgb = self.rgb_path.maxpool(x_rgb)
        
        # N x 64 x 8 x 56 x 56
        x_pose = self.pose_path.conv1(heatmap_imgs)
        x_pose = self.pose_path.maxpool(x_pose)

        x_rgb = self.rgb_path.layer1(x_rgb)

        x_rgb = self.rgb_path.layer2(x_rgb)

        x_pose = self.pose_path.layer1(x_pose)

        x_rgb_attn, x_pose_attn = self.featuredifference2(x_rgb, x_pose)

        if hasattr(self.rgb_path, 'layer2_lateral'):
            feat = x_pose.detach() if self.rgb_detach else x_pose

            x_pose_lateral = self.rgb_path.layer2_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)

        if hasattr(self.pose_path, 'layer1_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb

            x_rgb_lateral = self.pose_path.layer1_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)

        if hasattr(self.rgb_path, 'layer2_lateral'):

            x_rgb = torch.cat((x_rgb_attn, x_pose_lateral), dim=1)

        if hasattr(self.pose_path, 'layer1_lateral'):

            x_pose = torch.cat((x_pose_attn, x_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer3(x_rgb)

        x_pose = self.pose_path.layer2(x_pose)

        x_rgb_attn, x_pose_attn = self.featuredifference3(x_rgb, x_pose)
        # x_pose2=x_pose.clone()

        if hasattr(self.rgb_path, 'layer3_lateral'):
            feat = x_pose.detach() if self.rgb_detach else x_pose

            x_pose_lateral = self.rgb_path.layer3_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)

        if hasattr(self.pose_path, 'layer2_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb

            x_rgb_lateral = self.pose_path.layer2_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)

        if hasattr(self.rgb_path, 'layer3_lateral'):

            x_rgb = torch.cat((x_rgb_attn, x_pose_lateral), dim=1)

        if hasattr(self.pose_path, 'layer2_lateral'):

            x_pose = torch.cat((x_pose_attn, x_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer4(x_rgb)
        x_pose = self.pose_path.layer3(x_pose)

        x_rgb1 = x_rgb.clone()
        x_pose1 = x_pose.clone()

        return x_rgb, x_pose, x_rgb1, x_pose1, gt, gt_coarse
