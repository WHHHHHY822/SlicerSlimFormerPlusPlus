# SPDX-License-Identifier: Apache-2.0
"""Third-party building blocks used by the SlimFormer++ network.

Vendored so that the plugin does not have to install MedNeXt (only available
from git, and its package metadata pulls SimpleITK, which must not replace
3D Slicer's own copy) or MONAI. Parameter names and computations are identical
to the originals, so released checkpoints load with ``strict=True``.

* ``MedNeXtBlock``, ``MedNeXtDownBlock``, ``MedNeXtUpBlock``: copied from
  MedNeXt ``nnunet_mednext/network_architecture/mednextv1/blocks.py`` at revision
  0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba (https://github.com/MIC-DKFZ/MedNeXt),
  Copyright German Cancer Research Center (DKFZ), Apache License 2.0. Modified:
  the unused 2-D, layer-norm and GRN code paths were removed.
* ``UnetrBasicBlock`` (``res_block=True`` only) and ``UnetOutBlock``: written to
  match MONAI 1.4.0 ``monai.networks.blocks`` (https://github.com/Project-MONAI/MONAI),
  Copyright MONAI Consortium, Apache License 2.0: same module tree
  (``layer.conv1.conv``, ...), instance norm without affine parameters, and
  LeakyReLU(0.01).
"""
from collections import OrderedDict

import torch
import torch.nn as nn


class MedNeXtBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, exp_r: int = 4, kernel_size: int = 7,
                 do_res: int = True, norm_type: str = 'group', n_groups=None):
        super().__init__()
        if norm_type != 'group':
            raise NotImplementedError('only group norm is used by SlimFormer++')
        self.do_res = do_res
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2,
                               groups=in_channels if n_groups is None else n_groups)
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        self.conv2 = nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, dummy_tensor=None):
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False, norm_type='group'):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False, norm_type=norm_type)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=2,
                               padding=kernel_size // 2, groups=in_channels)

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        if self.resample_do_res:
            x1 = x1 + self.res_conv(x)
        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False, norm_type='group'):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False, norm_type=norm_type)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2)
        self.conv1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=kernel_size, stride=2,
                                        padding=kernel_size // 2, groups=in_channels)

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))
        if self.resample_do_res:
            res = torch.nn.functional.pad(self.res_conv(x), (1, 0, 1, 0, 1, 0))
            x1 = x1 + res
        return x1


def _conv(in_channels, out_channels, kernel_size, stride, bias):
    """MONAI ``get_conv_layer(..., act=None, norm=None, conv_only=False)``: a Sequential
    holding ``conv`` (plus an empty ADN block, which has no effect and no parameters)."""
    padding = (kernel_size - stride + 1) // 2
    return nn.Sequential(OrderedDict(
        conv=nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)))


class _UnetResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            raise NotImplementedError('SlimFormer++ only uses the non-downsampling residual block')
        self.conv1 = _conv(in_channels, out_channels, kernel_size, stride, bias=False)
        self.conv2 = _conv(out_channels, out_channels, kernel_size, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, inp):
        residual = inp
        out = self.lrelu(self.norm1(self.conv1(inp)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.lrelu(out)


class UnetrBasicBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, res_block=False):
        super().__init__()
        if spatial_dims != 3 or norm_name != 'instance' or not res_block:
            raise NotImplementedError('SlimFormer++ only uses the 3-D residual instance-norm block')
        self.layer = _UnetResBlock(in_channels, out_channels, kernel_size, stride)

    def forward(self, inp):
        return self.layer(inp)


class UnetOutBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels):
        super().__init__()
        if spatial_dims != 3:
            raise NotImplementedError
        self.conv = _conv(in_channels, out_channels, 1, 1, bias=True)

    def forward(self, inp):
        return self.conv(inp)
