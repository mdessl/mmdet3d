import torch

# Load the checkpoint
checkpoint = torch.load('/mmdet3d/bevfusion-seg.pth', map_location='cpu')

# Define the key mappings
prefix_mapping = {
    'encoders.camera.backbone': 'img_backbone',
    'encoders.camera.neck': 'img_neck',
    'encoders.camera.vtransform': 'view_transform',
    'encoders.lidar.backbone': 'pts_middle_encoder',
    'decoder.backbone': 'pts_backbone',
    'decoder.neck': 'pts_neck',
    'heads.map': 'seg_head'
}

# Create new state dict with renamed keys
new_state_dict = {}
for k, v in checkpoint['state_dict'].items():
    new_key = k
    for old_prefix, new_prefix in prefix_mapping.items():
        if k.startswith(old_prefix):
            new_key = k.replace(old_prefix, new_prefix)
            break
    new_state_dict[new_key] = v

# Replace the old state dict with the new one
checkpoint['state_dict'] = new_state_dict

# Save the modified checkpoint
torch.save(checkpoint, '/mmdet3d/bevfusion-seg2.pth')