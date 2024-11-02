from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
import torch.cuda as cuda
import gc
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import time


@MODELS.register_module()
class SBNet(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()
        #self.print_model_params()
        # freezing both encoders except for view transform (part of img encoder) bc n channels is different from pretrained bevfusion model (channels from both encoders must be the same)
        self.freeze_modules(module_keywords=["data_preprocessor","img_backbone","img_neck", 'pts_voxel_encoder', 'pts_middle_encoder'], exclude_keywords=["view_transform", 'pts_backbone', "pts_neck", "bbox_head"])


    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = None
        


        # Process camera if images exist and are non-zero
        feats_cam = None
        if batch_inputs_dict.get('imgs') is not None and batch_inputs_dict['imgs'].abs().sum() > 0:
            cam_input_metas = deepcopy(batch_input_metas)
            for meta in cam_input_metas:
                meta['sbnet_modality'] = 'img'
            feats_cam = self.extract_feat(batch_inputs_dict, cam_input_metas)
        
        # Process lidar if points exist and are non-zero
        feats_lidar = None
        if batch_inputs_dict.get('points') is not None and any(p.abs().sum() > 0 for p in batch_inputs_dict['points']):
            lidar_input_metas = deepcopy(batch_input_metas)
            for meta in lidar_input_metas:
                meta['sbnet_modality'] = 'lidar'
            feats_lidar = self.extract_feat(batch_inputs_dict, lidar_input_metas)

        if feats_cam is not None and feats_lidar is not None:
            feats = (feats_cam + feats_lidar) / 2
        elif feats_cam is not None:
            feats = feats_cam
        elif feats_lidar is not None:
            feats = feats_lidar
        else:
            raise ValueError("No valid features found")

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        return res

    @contextmanager
    def gpu_memory_log(self, description: str):
        """Context manager to track GPU memory usage of a specific operation."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated() / 1024**2
            start_time = time.time()
            
            try:
                yield
            finally:
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated() / 1024**2
                end_time = time.time()
                
                print(f"\n=== {description} ===")
                print(f"Memory: {end_memory - start_memory:.2f} MB")
                print(f"Time: {(end_time - start_time) * 1000:.2f} ms")
                print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        # Initialize memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        
        # Determine the dtype based on inputs
        if imgs is not None:
            dtype = imgs.dtype
        elif points is not None and len(points) > 0:
            dtype = points[0].dtype
        
        # Create modality masks for the batch
        batch_size = len(batch_input_metas)
        
        if batch_input_metas[0].get('sbnet_modality', None) is None:
            raise ValueError("sbnet_modality not found in batch_input_metas")

        modalities = [meta.get('sbnet_modality', None) for meta in batch_input_metas]
        camera_mask = torch.tensor([m == 'img' for m in modalities], 
                                 device=imgs.device if imgs is not None else points[0].device)
        lidar_mask = torch.tensor([m == 'lidar' for m in modalities], 
                                device=imgs.device if imgs is not None else points[0].device)
        

        dtype = None
        cam_feat = None
        lidar_feat = None
        
        # Process camera samples
        if imgs is not None and camera_mask.any():
            with self.gpu_memory_log("Camera Processing"):
                imgs = imgs[camera_mask]
                cam_metas = [meta for meta, is_cam in zip(batch_input_metas, camera_mask) if is_cam]
                # Prepare camera inputs for the selected samples
                lidar2image = imgs.new_tensor(np.asarray([meta['lidar2img'] for meta in cam_metas]))
                camera_intrinsics = imgs.new_tensor(np.array([meta['cam2img'] for meta in cam_metas]))
                camera2lidar = imgs.new_tensor(np.asarray([meta['cam2lidar'] for meta in cam_metas]))
                img_aug_matrix = imgs.new_tensor(np.asarray([meta.get('img_aug_matrix', np.eye(4)) 
                                                            for meta in cam_metas]))
                lidar_aug_matrix = imgs.new_tensor(np.asarray([meta.get('lidar_aug_matrix', np.eye(4)) 
                                                              for meta in cam_metas]))
                
                # Get camera features
                cam_indices = camera_mask.nonzero().squeeze(1).tolist()
                cam_points = [points[i] for i in cam_indices] if points is not None else None
                
                cam_feat = self.extract_img_feat(
                    imgs, cam_points, lidar2image, camera_intrinsics,
                    camera2lidar, img_aug_matrix, lidar_aug_matrix,
                    cam_metas
                )
                cam_feat = self.pts_backbone(cam_feat)
                cam_feat = self.pts_neck(cam_feat)
                if isinstance(cam_feat, list):
                    cam_feat = cam_feat[0]  # Take the first feature map if it's a list
                dtype = cam_feat.dtype

        # Process lidar samples
        if points is not None and lidar_mask.any():
            with self.gpu_memory_log("Lidar Processing"):
                lidar_indices = lidar_mask.nonzero().squeeze(1).tolist()
                lidar_points = [points[i] for i in lidar_indices]
                lidar_dict = {'points': lidar_points}
                lidar_feat = self.extract_pts_feat(lidar_dict) # lidar_feat is a tensor (bs, ...)
                lidar_feat = self.pts_backbone(lidar_feat)
                lidar_feat = self.pts_neck(lidar_feat)
                assert len(lidar_feat) == 1 and isinstance(lidar_feat, list)
                lidar_feat = lidar_feat[0]  # Take the first feature map if it's a list
                if dtype is None:
                    dtype = lidar_feat.dtype

        if dtype is None:
            raise ValueError("Neither camera nor lidar features were processed successfully")

        output_shape = (180, 180)
        device = imgs.device if imgs is not None else points[0].device
        x = torch.zeros((batch_size, 512, *output_shape), device=device, dtype=dtype)
        
        if lidar_feat is not None:
            x[lidar_mask] = lidar_feat
        if cam_feat is not None:
            x[camera_mask] = cam_feat
        # Memory reporting and cleanup

        """
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"\nGPU Memory Usage in extract_feat:")
            print(f"Current Memory: {current_memory:.2f} MB")
            print(f"Peak Memory: {peak_memory:.2f} MB")
            print(f"Memory Change: {current_memory - initial_memory:.2f} MB")
            
            if peak_memory > 16000:
                print("High memory usage detected, cleaning up...")
                torch.cuda.empty_cache()
                gc.collect()
        """
        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        with self.gpu_memory_log("Total Loss Calculation"):
            with self.gpu_memory_log("Feature Extraction"):
                batch_input_metas = [item.metainfo for item in batch_data_samples]
                feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

            losses = dict()
            if self.with_bbox_head:
                with self.gpu_memory_log("BBox Head Loss"):
                    bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
                losses.update(bbox_loss)

        return losses

    def freeze_modules(self, module_keywords=None, exclude_keywords=None, verbose=True):
        """Freeze model weights based on module names.
        
        Args:
            module_keywords (list[str], optional): List of keywords to match module names for freezing.
                If None, no modules will be frozen based on keywords.
            exclude_keywords (list[str], optional): List of keywords to exclude modules from freezing.
                Takes precedence over module_keywords.
            verbose (bool): Whether to print freezing status. Defaults to True.
        """
        if module_keywords is None:
            module_keywords = []
        if exclude_keywords is None:
            exclude_keywords = []
        
        frozen_params = 0
        total_params = 0
        
        for name, module in self.named_modules():
            # Only process leaf modules (those without children)
            if len(list(module.children())) == 0:
                params = list(module.parameters())
                if not params:  # Skip modules without parameters
                    continue
                
                should_freeze = any(keyword in name for keyword in module_keywords) if module_keywords else False
                should_exclude = any(keyword in name for keyword in exclude_keywords)
                
                # Count parameters
                num_params = sum(p.numel() for p in params)
                total_params += num_params
                
                # Freeze if module matches criteria and isn't excluded
                if should_freeze and not should_exclude:
                    for param in params:
                        param.requires_grad = False
                    frozen_params += num_params
                    if verbose:
                        print(f"Froze {name}: {num_params:,} parameters")
                else:
                    if verbose:
                        print(f"Left {name} unfrozen: {num_params:,} parameters")
        
        if verbose:
            print(f"\nFroze {frozen_params:,} parameters out of {total_params:,} total")
            print(f"Trainable parameters: {total_params - frozen_params:,}")

    def print_model_params(self):
        """Print model parameters statistics and GPU memory usage."""
        print("\n=== Model Parameters and GPU Usage ===")
        total_params = 0
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Only print leaf modules
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name}: {num_params:,} parameters")
                    total_params += num_params
        
        print(f"\nTotal parameters: {total_params:,}")
        
        if torch.cuda.is_available():
            print("\nGPU Information:")
            print(f"GPU Device: {torch.cuda.get_device_name()}")
            print(f"Current Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Current Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print(f"Max Memory Usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print("=====================")


