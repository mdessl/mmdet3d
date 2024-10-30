from typing import List, Optional
import numpy as np
from torch.utils.data import Sampler
from mmengine.registry import SAMPLERS

@SAMPLERS.register_module()
class GroupSampler(Sampler):
    """Sampler that ensures each batch contains samples of the same modality."""
    
    def __init__(self, dataset, groups: List[str], group_key: str, shuffle: bool = True):
        self.dataset = dataset
        self.groups = groups
        self.group_key = group_key
        self.shuffle = shuffle
        self.batch_size = dataset.batch_size  # Get batch size from dataset config
        
        # Create indices for each group
        self.group_indices = {group: [] for group in groups}
        for idx in range(len(dataset)):
            group = dataset.get_data_info(idx)[group_key]
            self.group_indices[group].append(idx)
            
        # Convert to numpy arrays for faster operations
        self.group_indices = {k: np.array(v) for k, v in self.group_indices.items()}
        
    def __iter__(self):
        if self.shuffle:
            # Shuffle indices within each group
            for group in self.groups:
                np.random.shuffle(self.group_indices[group])
        
        # Create batches ensuring same modality
        all_indices = []
        for group in self.groups:
            indices = self.group_indices[group]
            # Pad if necessary to make complete batches
            if len(indices) % self.batch_size != 0:
                pad_size = self.batch_size - (len(indices) % self.batch_size)
                indices = np.pad(indices, (0, pad_size), mode='wrap')
            # Add batch-sized chunks to final list
            for i in range(0, len(indices), self.batch_size):
                all_indices.extend(indices[i:i + self.batch_size])
                
        if self.shuffle:
            # Shuffle the batches while maintaining batch integrity
            batch_indices = np.array(all_indices).reshape(-1, self.batch_size)
            np.random.shuffle(batch_indices)
            all_indices = batch_indices.flatten().tolist()
            
        return iter(all_indices)
    
    def __len__(self):
        total_samples = sum(len(indices) for indices in self.group_indices.values())
        # Round up to nearest multiple of batch_size
        return ((total_samples + self.batch_size - 1) // self.batch_size) * self.batch_size 