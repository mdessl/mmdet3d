from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import time
import numpy as np
import torch
import torch.cuda as cuda
import gc

from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmdet3d.registry import MODELS
from .bevfusion import BEVFusion

@MODELS.register_module()
class SBNet(BEVFusion):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Add SBNet-specific initialization
        self.freeze_modules(
            module_keywords=["data_preprocessor", "img_backbone", "img_neck", 'pts_voxel_encoder', "view_transform", 'pts_middle_encoder'],
            exclude_keywords=['pts_backbone', "pts_neck"]
        )
        print("nothing frozen!!")
        print("nothing frozen!!")
        print("nothing frozen!!")
        print("nothing frozen!!")
        print("nothing frozen!!")

    def predict(self, batch_inputs_dict: Dict[str, Optional[torch.Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Override predict to handle modality-specific processing"""
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = None
        
        # Process camera if images exist and are non-zero
        feats_cam = None
        if batch_inputs_dict.get('imgs') is not None and batch_input_metas[0].get('img_not_zero', True):
            cam_input_metas = deepcopy(batch_input_metas)
            for meta in cam_input_metas:
                meta['sbnet_modality'] = 'img'
            feats_cam = self.extract_feat(batch_inputs_dict, cam_input_metas)
        
        # Process lidar if points exist and are non-zero
        feats_lidar = None
        if batch_inputs_dict.get('points') is not None and batch_input_metas[0].get('lidar_not_zero', True):
            lidar_input_metas = deepcopy(batch_input_metas)
            for meta in lidar_input_metas:
                meta['sbnet_modality'] = 'lidar'
            feats_lidar = self.extract_feat(batch_inputs_dict, lidar_input_metas)

        # Combine features
        if feats_cam is not None and feats_lidar is not None:
            if type(feats_cam) == list:
                if len(feats_cam) > 1:
                    raise ValueError("more than one camera feats")
                feats_cam = feats_cam[0]
            if type(feats_lidar) == list:
                if len(feats_lidar) > 1:
                    raise ValueError("more than one lidar feats")
                feats_lidar = feats_lidar[0]
            feats = (feats_cam + feats_lidar) / 2
        elif feats_cam is not None:
            feats = feats_cam
        elif feats_lidar is not None:
            feats = feats_lidar
        else:
            raise ValueError("No valid features found")

        if self.with_seg_head:
            outputs = self.seg_head.predict(feats, batch_input_metas)

            # Add ground truth BEV masks to outputs
            for i, data_sample in enumerate(batch_data_samples):
                outputs[i]['gt_masks_bev'] = batch_input_metas[i]["gt_masks_bev"]
            return self.add_pred_to_datasample(batch_data_samples, 
                                             data_instances_3d=outputs)
        elif self.with_bbox_head:
            losses = self.bbox_head.loss(feats, batch_data_samples)

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
    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        # Get inputs and determine device
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        device = imgs.device if imgs is not None else points[0].device

        # Ensure modality information is present
        if batch_input_metas[0].get('sbnet_modality') is None:
            raise ValueError("sbnet_modality not found in batch_input_metas")
        modalities = ["img" for meta in batch_input_metas] #meta.get('sbnet_modality')
        #print(modalities)
        batch_size = len(batch_input_metas)

        # Create modality masks
        camera_mask = torch.tensor([m == 'img' for m in modalities], device=device)
        lidar_mask = torch.tensor([m == 'lidar' for m in modalities], device=device)

        dtype = None
        cam_feat = None
        lidar_feat = None

        # Process camera samples if available
        if imgs is not None and camera_mask.any():
            imgs = imgs[camera_mask]
            cam_metas = [meta for meta, is_cam in zip(batch_input_metas, camera_mask) if is_cam]
            # Prepare camera transformation matrices and augmentation info
            lidar2image = imgs.new_tensor(np.asarray([meta['lidar2img'] for meta in cam_metas]))
            camera_intrinsics = imgs.new_tensor(np.asarray([meta['cam2img'] for meta in cam_metas]))
            camera2lidar = imgs.new_tensor(np.asarray([meta['cam2lidar'] for meta in cam_metas]))
            img_aug_matrix = imgs.new_tensor(np.asarray([meta.get('img_aug_matrix', np.eye(4)) for meta in cam_metas]))
            lidar_aug_matrix = imgs.new_tensor(np.asarray([meta.get('lidar_aug_matrix', np.eye(4)) for meta in cam_metas]))

            # Get camera points corresponding to the camera indices
            cam_indices = camera_mask.nonzero().squeeze(1).tolist()
            cam_points = [points[i] for i in cam_indices] if points is not None else None

            cam_feat = self.extract_img_feat(
                imgs, cam_points, lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix,
                cam_metas
            )
            # If a list is returned, take the first element
            if isinstance(cam_feat, list):
                cam_feat = cam_feat[0]
            dtype = cam_feat.dtype

        # Process lidar samples if available
        if points is not None and lidar_mask.any():
            lidar_indices = lidar_mask.nonzero().squeeze(1).tolist()
            lidar_points = [points[i] for i in lidar_indices]
            lidar_dict = {'points': lidar_points}
            lidar_feat = self.extract_pts_feat(lidar_dict)
            if isinstance(lidar_feat, list):
                lidar_feat = lidar_feat[0]
            if dtype is None:
                dtype = lidar_feat.dtype

        # Raise error if no features were extracted
        if dtype is None:
            raise ValueError("Neither camera nor lidar features were processed successfully")

        # Initialize combined feature tensor and assign modality features
        output_shape = (128, 128)
        combined_feat = torch.zeros((batch_size, 256, *output_shape), device=device, dtype=dtype)
        if lidar_feat is not None:
            combined_feat[lidar_mask] = lidar_feat
        if cam_feat is not None:
            combined_feat[camera_mask] = cam_feat

        # Apply backbone and neck
        combined_feat = self.pts_backbone(combined_feat)
        combined_feat = self.pts_neck(combined_feat)

        return combined_feat


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



