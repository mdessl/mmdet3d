import argparse
import copy
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--removal-rates', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8],
                      help='Percentage of samples to remove for each modality')
    parser.add_argument('--modalities', nargs='+', default=['img', 'points'],
                      help='Modalities to test removal')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

class ModalityDropCallback:
    """Callback to drop modalities during testing."""
    def __init__(self, modality, drop_rate):
        self.modality = modality
        self.drop_rate = drop_rate
        
    def before_test_iter(self, runner):
        """Zero out random samples of specified modality."""
        batch_inputs = runner.iter_inputs['inputs']
        batch_size = len(runner.iter_inputs['data_samples'])
        
        # Randomly select indices to zero out
        num_to_drop = int(batch_size * self.drop_rate)
        drop_indices = np.random.choice(batch_size, num_to_drop, replace=False)
        
        if self.modality == 'img':
            if 'imgs' in batch_inputs:
                batch_inputs['imgs'][drop_indices] = torch.zeros_like(
                    batch_inputs['imgs'][drop_indices])
        elif self.modality == 'points':
            if 'points' in batch_inputs:
                for idx in drop_indices:
                    batch_inputs['points'][idx] = torch.zeros_like(
                        batch_inputs['points'][idx][:1])  # Keep only one zero point
                    
        runner.iter_inputs['inputs'] = batch_inputs

def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Run tests for each modality and removal rate
    for modality in args.modalities:
        for rate in args.removal_rates:
            # Create a new config for this test
            test_cfg = copy.deepcopy(cfg)
            
            # Set work directory for this specific test
            if args.work_dir:
                test_cfg.work_dir = f"{args.work_dir}/{modality}_removal_{int(rate*100)}percent"
            
            # Add modality drop callback
            test_cfg.custom_hooks = [
                dict(
                    type='ModalityDropCallback',
                    modality=modality,
                    drop_rate=rate,
                    priority='HIGHEST'
                )
            ]
            
            # Build runner
            runner = Runner.from_cfg(test_cfg)
            
            # Run testing
            runner.test(checkpoint=args.checkpoint)

if __name__ == '__main__':
    main() 


"""
python tools/test.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    work_dirs/bevfusion/latest.pth \
    --removal-rates 0.2 0.4 0.6 0.8 \
    --work-dir work_dirs/modality_robustness
"""