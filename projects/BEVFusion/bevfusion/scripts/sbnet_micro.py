import pickle
import copy

# Load the original data
with open("/mmdet3d/data/nuscenes/nuscenes_infos_val.pkl", 'rb') as f:
    nuscenes_infos = pickle.load(f)

# Get original data list and limit to first 10 samples
import random
original_data = random.sample(nuscenes_infos["data_list"], 100)  # Take 50 random samples
sample_indices = [eg["sample_idx"] for eg in original_data]
total_samples = max(sample_indices) + 1

# Create a new list with duplicated entries
new_data_list = [None] * (total_samples * 2)

for entry in original_data:
    # Add original entry (image modality)
    img_entry = copy.deepcopy(entry)
    img_entry['sbnet_modality'] = 'img'
    new_data_list[entry['sample_idx']] = img_entry
    
    # Add duplicated entry (lidar modality)
    lidar_entry = copy.deepcopy(entry)
    lidar_entry['sbnet_modality'] = 'lidar'
    lidar_entry['sample_idx'] += total_samples
    new_data_list[lidar_entry['sample_idx']] = lidar_entry

# Remove any None entries
new_data_list = [entry for entry in new_data_list if entry is not None]

# Replace the original data_list with the new one
nuscenes_infos["data_list"] = new_data_list

# Save modified file
with open("/mmdet3d/data/nuscenes/micro.pkl", 'wb') as f:
    pickle.dump(nuscenes_infos, f)