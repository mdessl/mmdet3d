import pickle

# Load the dataset
with open("/mmdet3d/data/nuscenes/nuscenes_infos_train.pkl", 'rb') as f:
    nuscenes_infos = pickle.load(f)

# Modify each entry in the original data list
for entry in nuscenes_infos["data_list"]:
    entry['sbnet_modality'] = 'lidar'

# Verify the changes
for i, entry in enumerate(nuscenes_infos["data_list"]):
    print(f"Index: {i}, Sample Index: {entry['sample_idx']}, Modality: {entry['sbnet_modality']}")

# Save modified file
with open("/mmdet3d/data/nuscenes/nuscenes_infos_train.pkl", 'wb') as f:
    pickle.dump(nuscenes_infos, f) 