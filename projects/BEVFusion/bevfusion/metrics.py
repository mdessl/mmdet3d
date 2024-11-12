from typing import Dict, List, Optional
import torch
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmdet3d.registry import METRICS

@METRICS.register_module()
class NuScenesBEVFusionMetric(NuScenesMetric):
    """NuScenes metric with additional BEV segmentation evaluation."""

    def __init__(self, 
                 *args, 
                 seg_classes: Optional[List[str]] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seg_classes = seg_classes
        self.seg_thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute metrics for both detection and segmentation tasks.

        Args:
            results (List[dict]): Testing results of the dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # First compute detection metrics using parent class
        #metrics = super().compute_metrics(results)
        metrics = {}
        # Add segmentation metrics if segmentation results exist
        if any('pred_instances_3d' in result and 
               hasattr(result['pred_instances_3d'], 'seg_logits') 
               for result in results):
            seg_metrics = self._compute_segmentation_metrics(results)
            metrics.update(seg_metrics)

        return metrics

    def _compute_segmentation_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute BEV segmentation metrics.

        Args:
            results (List[dict]): Testing results containing segmentation predictions.

        Returns:
            Dict[str, float]: Segmentation metrics including per-class IoU and mean IoU.
        """
        num_classes = len(self.seg_classes)
        num_thresholds = len(self.seg_thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred_logits = result['pred_instances_3d'].seg_logits
            gt_mask = result['gt_seg_mask']

            # Reshape predictions and ground truth
            pred = pred_logits.detach().reshape(num_classes, -1)
            label = gt_mask.detach().bool().reshape(num_classes, -1)

            # Calculate metrics at different thresholds
            pred = pred[:, :, None] >= self.seg_thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        # Calculate IoU for each class and threshold
        ious = tp / (tp + fp + fn + 1e-7)

        # Compile metrics dictionary
        metrics = {}
        
        # Per-class metrics
        for class_idx, class_name in enumerate(self.seg_classes):
            metrics[f'seg/{class_name}/iou@max'] = ious[class_idx].max().item()
            for threshold, iou in zip(self.seg_thresholds, ious[class_idx]):
                metrics[f'seg/{class_name}/iou@{threshold.item():.2f}'] = iou.item()

        # Mean metrics
        metrics['seg/mean/iou@max'] = ious.max(dim=1).values.mean().item()

        return metrics 