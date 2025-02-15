import pickle
import copy
# /mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl
with open("/home/markus-essl/mmdet3d/data/mini/nuscenes_infos_train.pkl", 'rb') as f:
    nuscenes_infos = pickle.load(f)

import pickle
import copy
# /mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl

# Get original data list
original_data = nuscenes_infos["data_list"]
# Create a new list for both modalities
new_infos = []

for entry in original_data:
    # Add original entry (image modality)
    img_entry = copy.deepcopy(entry)
    img_entry['sbnet_modality'] = 'img'
    new_infos.append(img_entry)
    
    # Add duplicated entry (lidar modality)
    lidar_entry = copy.deepcopy(entry)
    lidar_entry['sbnet_modality'] = 'lidar'
    new_infos.append(lidar_entry)

# Replace the original infos with the new one
nuscenes_infos["data_list"] = new_infos

# Verify the changes
for i, entry in enumerate(nuscenes_infos["data_list"]):
    print(f"Entry {i}, Modality: {entry['sbnet_modality']}")

# Save modified file
with open("/home/markus-essl/mmdet3d/data/mini/nuscenes_infos_train.pkl", 'wb') as f:
    pickle.dump(nuscenes_infos, f)

