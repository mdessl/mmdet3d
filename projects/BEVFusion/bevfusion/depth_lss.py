# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from torch import nn

from mmdet3d.registry import MODELS
from .ops import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseViewTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )
        for b in range(batch_size):
            if points[b].sum() == 0:
                continue

            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth = depth.to(masked_dist.dtype)
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class DepthLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from torchvision.models.resnet import resnet18
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from torchvision.utils import save_image
from mmdet3d.models.fusion_layers import apply_3d_transformation
import torch.nn.functional as F


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.shape[2:], mode="bilinear", align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class CamEncode(nn.Module):
    def __init__(self, D, C, inputC):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.depthnet = nn.Conv2d(inputC, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, : self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x, depth


@MODELS.register_module()
class LiftSplatShoot(nn.Module):
    def __init__(
        self,
        lss=False,
        final_dim=(900, 1600),
        camera_depth_range=[4.0, 45.0, 1.0],
        pc_range=[-50, -50, -5, 50, 50, 3],
        downsample=4,
        grid=3,
        inputC=256,
        camC=64,
    ):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            final_dim: actual RGB image size for actual BEV coordinates, default (900, 1600)
            downsample (int): the downsampling rate of the input camera feature spatial dimension (default (224, 400)) to final_dim (900, 1600), default 4.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            pc_range: point cloud range.
            inputC: input camera feature channel dimension (default 256).
            grid: stride for splat, see https://github.com/nv-tlabs/lift-splat-shoot.

        """
        super(LiftSplatShoot, self).__init__()
        self.pc_range = pc_range
        self.grid_conf = {
            "xbound": [pc_range[0], pc_range[3], grid],
            "ybound": [pc_range[1], pc_range[4], grid],
            "zbound": [pc_range[2], pc_range[5], grid],
            "dbound": camera_depth_range,
        }
        self.final_dim = final_dim
        self.grid = grid

        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = downsample
        self.fH, self.fW = (
            self.final_dim[0] // self.downsample,
            self.final_dim[1] // self.downsample,
        )
        self.camC = camC
        self.inputC = inputC
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.inputC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        z = self.grid_conf["zbound"]
        cz = int(self.camC * ((z[1] - z[0]) // z[2]))
        self.lss = lss
        self.bevencode = nn.Sequential(
            nn.Conv2d(cz, cz, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cz),
            nn.ReLU(inplace=True),
            nn.Conv2d(cz, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, inputC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inputC),
            nn.ReLU(inplace=True),
        )
        if self.lss:
            self.bevencode = nn.Sequential(
                nn.Conv2d(cz, camC, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(camC),
                BevEncode(inC=camC, outC=inputC),
            )

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = self.fH, self.fW
        ds = (
            torch.arange(*self.grid_conf["dbound"], dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        rots,
        trans,
        post_rots=None,
        post_trans=None,
        extra_rots=None,
        extra_trans=None,
    ):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        # ADD
        # undo post-transformation
        # B x N x D x H x W x 3
        if post_rots is not None or post_trans is not None:
            if post_trans is not None:
                points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots is not None:
                points = (
                    torch.inverse(post_rots)
                    .view(B, N, 1, 1, 1, 3, 3)
                    .matmul(points.unsqueeze(-1))
                )
        else:
            points = self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(
                -1
            )  # B x N x D x H x W x 3 x 1

        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        if extra_rots is not None or extra_trans is not None:
            if extra_rots is not None:
                points = (
                    extra_rots.view(B, N, 1, 1, 1, 3, 3)
                    .matmul(points.unsqueeze(-1))
                    .squeeze(-1)
                )
            if extra_trans is not None:
                points += extra_trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C"""
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        x, depth = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, H, W)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, H, W)
        return x, depth

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        batch_size = x.shape[0]

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        batch_ix = batch_ix.to(geom_feats.device)
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[
            geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]
        ] = x

        return final

    def get_voxels(
        self,
        x,
        rots=None,
        trans=None,
        post_rots=None,
        post_trans=None,
        extra_rots=None,
        extra_trans=None,
    ):
        geom = self.get_geometry(
            rots, trans, post_rots, post_trans, extra_rots, extra_trans
        )
        x, depth = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)
        return x, depth

    def s2c(self, x):
        B, C, H, W, L = x.shape
        bev = torch.reshape(x, (B, C * H, W, L))
        bev = bev.permute((0, 1, 3, 2))
        return bev

    def forward(
        self,
        x,
        rots,
        trans,
        lidar2img_rt=None,
        img_metas=None,
        post_rots=None,
        post_trans=None,
        extra_rots=None,
        extra_trans=None,
    ):
        x, depth = self.get_voxels(
            x, rots, trans, post_rots, post_trans, extra_rots, extra_trans
        )  # [B, C, H, W, L]
        bev = self.s2c(x)
        x = self.bevencode(bev)
        return x, depth


@MODELS.register_module()
class LSSNoPoints(nn.Module):
    def __init__(
        self,
        grid=1,
        num_views=6,
        final_dim=(900, 1600),
        pc_range=[-50, -50, -5, 50, 50, 3],
        downsample=4,
        imc=256,
        **kwargs,
    ):
        super().__init__()
        self.lift_splat_shot_vis = LiftSplatShoot(
            grid=grid,
            inputC=imc,
            camC=64,
            pc_range=pc_range,
            final_dim=final_dim,
            downsample=downsample,
        )

    def forward(
        self,
        img_feats,
        img_metas,
        img_aug_matrix=None,
        lidar_aug_matrix=None,
        gt_bboxes_3d=None,
    ):
        rots = []
        trans = []
        for sample_idx in range(batch_size):
            rot_list = []
            trans_list = []
            for mat in img_metas[sample_idx]["lidar2img"]:
                mat = torch.Tensor(mat).to(img_feats.device)
                rot_list.append(mat.inverse()[:3, :3])
                trans_list.append(mat.inverse()[:3, 3].view(-1))

            rot_list = torch.stack(rot_list, dim=0)
            trans_list = torch.stack(trans_list, dim=0)
            rots.append(rot_list)
            trans.append(trans_list)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        lidar2img_rt = img_metas[sample_idx]["lidar2img"]

        post_rots = None
        post_trans = None
        if img_aug_matrix is not None:
            img_aug_matrix = torch.stack(img_aug_matrix).permute(1, 0, 2, 3)
            post_rots = img_aug_matrix[..., :3, :3]
            post_trans = img_aug_matrix[..., :3, 3]

        extra_rots = None
        extra_trans = None
        if lidar_aug_matrix is not None:
            lidar_aug_matrix = lidar_aug_matrix.unsqueeze(1).repeat(
                1, self.num_views, 1, 1
            )
            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]
        img_bev_feat, depth_dist = self.lift_splat_shot_vis(
            img_feats,
            rots,
            trans,
            lidar2img_rt=lidar2img_rt,
            img_metas=img_metas,
            post_rots=post_rots,
            post_trans=post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        if pts_feats is None:
            pts_feats = [img_bev_feat]  ####cam stream only
        else:
            if self.lc_fusion:
                pts_feats = [
                    self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))
                ]
                if self.se:
                    pts_feats = [self.seblock(pts_feats[0])]

    return dict(img_feats=img_feats, pts_feats=pts_feats, depth_dist=depth_dist)
