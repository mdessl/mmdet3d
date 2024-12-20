from typing import List, Optional
import numpy as np
from torch.utils.data import Sampler
from mmengine.registry import DATA_SAMPLERS
from mmengine.dataset import DefaultSampler
import copy
import torch
from mmengine.dist import get_dist_info, sync_random_seed

@DATA_SAMPLERS.register_module()
class AlternatingSampler(Sampler):
    def __init__(self, dataset, shuffle=True, seed=None):
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        
        self.dataset = dataset
        self.total_length = len(self.dataset.dataset)
        assert self.total_length % 2 == 0, "Dataset length must be even"
        self.half_length = self.total_length // 2

        # Print first few samples to verify dataset structure
        print("\nDataset structure check:")
        for i in range(min(5, self.half_length)):
            print(f"Index {i}: {self.dataset.dataset[i].get('sbnet_modality', 'unknown')}")
            print(f"Index {i + self.half_length}: {self.dataset.dataset[i + self.half_length].get('sbnet_modality', 'unknown')}")

    def __iter__(self):
        # Create strictly alternating indices
        indices = []
        for i in range(self.half_length):
            indices.extend([i, i + self.half_length])
        
        # Handle distributed training if needed
        if self.world_size > 1:
            indices = indices[self.rank:self.total_length:self.world_size]
            
        print("\nFirst 10 pairs of indices:")
        for i in range(0, min(20, len(indices)), 2):
            if i+1 < len(indices):
                idx1, idx2 = indices[i], indices[i+1]
                mod1 = self.dataset.dataset[idx1].get('sbnet_modality', 'unknown')
                mod2 = self.dataset.dataset[idx2].get('sbnet_modality', 'unknown')
                print(f"Pair {i//2}: [{idx1}({mod1}), {idx2}({mod2})]")
            
        return iter(indices)

    def __len__(self):
        return self.total_length
        
    def set_epoch(self, epoch):
        # Do nothing to prevent any shuffling
        pass