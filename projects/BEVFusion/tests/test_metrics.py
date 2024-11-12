import unittest
import torch
import numpy as np
from mmdet3d.evaluation.metrics import NuScenesMetric
from projects.BEVFusion.bevfusion.metrics import NuScenesBEVFusionMetric

class TestNuScenesBEVFusionMetric(unittest.TestCase):
    def setUp(self):
        seg_classes = ['drivable_area', 'ped_crossing', 'walkway']
        
        self.metric = NuScenesBEVFusionMetric(
            data_root='tests/data/nuscenes',
            ann_file='tests/data/nuscenes/nuscenes_infos_val.pkl',
            metric='bbox',
            seg_classes=seg_classes
        )
        
        self.metric.dataset_meta = {
            'classes': ['car', 'truck', 'bus'],
            'box_type_3d': 'LiDAR'
        }

    def test_segmentation_metrics(self):
        batch_size = 2
        num_classes = 3
        h, w = 100, 100
        
        results = []
        for i in range(batch_size):
            pred_logits = torch.rand(num_classes, h, w)
            gt_mask = torch.zeros(num_classes, h, w, dtype=torch.bool)
            gt_mask[:, :h//2, :w//2] = True
            
            result = {
                'pred_instances_3d': type('', (), {})(),
                'sample_idx': i,
                'eval_ann_info': {},
                'gt_seg_mask': gt_mask
            }
            result['pred_instances_3d'].seg_logits = pred_logits
            
            results.append(result)

        metrics = self.metric._compute_segmentation_metrics(results)

        self.assertIn('seg/drivable_area/iou@max', metrics)
        self.assertIn('seg/mean/iou@max', metrics)
        self.assertTrue(0 <= metrics['seg/mean/iou@max'] <= 1)

if __name__ == '__main__':
    unittest.main() 