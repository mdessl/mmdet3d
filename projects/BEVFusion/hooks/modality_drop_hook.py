from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class ModalityDropHook(Hook):
    """Hook to drop modalities during testing."""
    def __init__(self, modality, drop_rate):
        self.modality = modality
        self.drop_rate = float(drop_rate)
        print(f"Initialized ModalityDropHook: {modality} at {self.drop_rate*100}%")
        
    def before_test_iter(self, runner, batch_idx: int, data_batch: dict = None) -> None:
        """Zero out specified modality before each test iteration."""
        if self.drop_rate <= 0:
            return
            
        if 'inputs' not in runner.iter_inputs:
            return
            
        batch_inputs = runner.iter_inputs['inputs']
        
        if self.modality == 'img' and 'imgs' in batch_inputs:
            batch_inputs['imgs'].zero_()
        elif self.modality == 'points' and 'points' in batch_inputs:
            for points in batch_inputs['points']:
                points.zero_()
                
        runner.iter_inputs['inputs'] = batch_inputs 