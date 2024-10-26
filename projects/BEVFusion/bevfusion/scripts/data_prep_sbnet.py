import pickle

# Load original file
with open("/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl", 'rb') as f:
    nuscenes_infos = pickle.load(f)

# Get original data list
original_data = nuscenes_infos["data_list"]
total_samples = len(original_data)

# Extend the list with itself (doubles the size)
nuscenes_infos["data_list"].extend(original_data)

# Add modality field and update sample_idx for each entry
for i in range(total_samples * 2):
    if i < total_samples:
        nuscenes_infos["data_list"][i]['sbnet_modality'] = 'img'
    else:
        nuscenes_infos["data_list"][i]['sbnet_modality'] = 'lidar'
        # Update sample_idx for duplicated entries - simply add total_samples
        nuscenes_infos["data_list"][i]['sample_idx'] += total_samples

# Save modified file
with open("/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl", 'wb') as f:
    pickle.dump(nuscenes_infos, f)