import argparse
import copy
import os
from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('gpus', type=int, help='number of gpus')
    parser.add_argument('--removal-rates', nargs='+', type=float, 
                       default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument('--modalities', nargs='+', default=['img', 'points'])
    parser.add_argument('--work-dir', default='work_dirs/modality_robustness')
    parser.add_argument('--cfg-options', nargs='+', action='append', default=[])
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # Add custom imports to config
    cfg.custom_imports = dict(
        imports=['projects.BEVFusion.hooks'],
        allow_failed_imports=False)
    
    # Update config with any provided options
    if args.cfg_options:
        cfg.merge_from_dict(dict(args.cfg_options))
        
    # Debug print
    print("Custom hooks configuration:")
    print(cfg.custom_hooks)

    # Run tests for each modality and rate
    for modality in args.modalities:
        for rate in args.removal_rates:
            print(f"\nTesting {modality} removal rate: {rate*100}%")
            
            # Create a new config for this test
            test_cfg = copy.deepcopy(cfg)
            test_cfg.work_dir = os.path.join(
                args.work_dir, 
                f"{modality}_removal_{int(rate*100)}percent"
            )
            test_cfg.custom_hooks = [
                dict(
                    type='ModalityDropHook',
                    modality=modality,
                    drop_rate=rate,
                    priority='HIGHEST'
                )
            ]
            
            # Build runner
            runner = Runner.from_cfg(test_cfg)
            
            # Run testing
            runner.test()

if __name__ == '__main__':
    main()


"""
bash tools/dist_test.sh \
    projects/BEVFusion/tools/test_modality_robustness.py \
    projects/BEVFusion/configs
"""