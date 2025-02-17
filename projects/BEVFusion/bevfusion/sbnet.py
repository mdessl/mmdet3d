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

import torch.nn as nn
class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.adapter(x)


import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        if isinstance(output, tuple):
            self.activations = output[0]  # Take first element if tuple
        else:
            self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0]  # Take first element if tuple
        else:
            self.gradients = grad_output[0]
    
    def generate_cam(self, feature_tensor):
        # For segmentation, we'll use the mean across all classes
        if isinstance(feature_tensor, tuple):
            feature_tensor = feature_tensor[0]  # Take first element if tuple
            
        # Create a copy of the tensor that requires gradients
        feature_tensor = feature_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass to get activations
        if self.activations is None:
            raise ValueError("No activations captured")
            
        # Backward pass
        self.model.zero_grad()
        
        # Sum across all output channels (classes)
        if isinstance(self.activations, tuple):
            target = self.activations[0].sum()
        else:
            target = self.activations.sum()
            
        target.backward(retain_graph=True)
        
        # Check if we have gradients
        if self.gradients is None:
            raise ValueError("No gradients captured")
        
        # Calculate weights - average across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # Generate CAM
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(128, 128),  # Match your BEV size
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        return cam.cpu().numpy()


@MODELS.register_module()
class SBNet(BEVFusion):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Add SBNet-specific initialization
        self.lidar_adapter = FeatureAdapter(in_channels=256, out_channels=256)
        self.freeze_modules(
            module_keywords=["data_preprocessor", "img_backbone", "img_neck"],
            exclude_keywords=["lidar_adapter",'pts_backbone', "pts_neck", 'pts_voxel_encoder','pts_middle_encoder', "view_transform"]
        )
        self.init_gradcam()
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
            print("both")
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
        modalities = [meta.get('sbnet_modality') for meta in batch_input_metas] #meta.get('sbnet_modality')
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
            #lidar_feat = self.lidar_adapter(lidar_feat)
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
        # File I/O adds ~1-10ms overhead per write, but logging only first 20 iters
        # and every 50th iter after means minimal impact on overall training time.
        # Computing stats still only costs ~0.01ms
        # Overall impact on training speed should be negligible
        if hasattr(self, '_iter_count'):
            self._iter_count += 1
        else:
            self._iter_count = 0
            
        if self._iter_count < 20 or (self._iter_count % 50 == 0):
            if cam_feat is not None:
                with open('feature_stats.txt', 'a') as f:
                    f.write(f"\nIteration {self._iter_count}:\n")
                    f.write("Camera feature stats:\n")
                    f.write("Mean: {:.4f}\n".format(cam_feat.mean().item()))
                    f.write("Std:  {:.4f}\n".format(cam_feat.std().item()))
                    f.write("Min:  {:.4f}\n".format(cam_feat.min().item()))
                    f.write("Max:  {:.4f}\n".format(cam_feat.max().item()))

            if lidar_feat is not None:
                with open('feature_stats.txt', 'a') as f:
                    f.write("LiDAR feature stats:\n") 
                    f.write("Mean: {:.4f}\n".format(lidar_feat.mean().item()))
                    f.write("Std:  {:.4f}\n".format(lidar_feat.std().item()))
                    f.write("Min:  {:.4f}\n".format(lidar_feat.min().item()))
                    f.write("Max:  {:.4f}\n".format(lidar_feat.max().item()))
                    f.write("\n")
        # Apply backbone and neck
        combined_feat = self.pts_backbone(combined_feat)
        combined_feat = self.pts_neck(combined_feat)

        if hasattr(self, '_iter_count') and (self._iter_count < 20 or self._iter_count % 50 == 0):
            self.visualize_gradcam(
                cam_feat, 
                lidar_feat,
                f'gradcam_iter_{self._iter_count}'
            )

        return combined_feat

    def init_gradcam(self):
        # Initialize GradCAM for camera and lidar features
        self.cam_gradcam = GradCAM(self, self.pts_backbone)
        self.lidar_gradcam = GradCAM(self, self.pts_backbone)

    def visualize_gradcam(self, cam_feat, lidar_feat, save_path_prefix):
        # Generate GradCAM for camera features
        if cam_feat is not None:
            with torch.enable_grad():
                cam_heatmap = self.cam_gradcam.generate_cam(cam_feat)
                cam_heatmap = np.uint8(255 * cam_heatmap)
                cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(f'{save_path_prefix}_cam_gradcam.png', cam_heatmap)
        
        # Generate GradCAM for lidar features
        if lidar_feat is not None:
            with torch.enable_grad():
                lidar_heatmap = self.lidar_gradcam.generate_cam(lidar_feat)
                lidar_heatmap = np.uint8(255 * lidar_heatmap)
                lidar_heatmap = cv2.applyColorMap(lidar_heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(f'{save_path_prefix}_lidar_gradcam.png', lidar_heatmap)




