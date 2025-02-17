import torch
cp = torch.load("/home/markus-essl/mmdet3d/bevfusion-seg.pth")["state_dict"]

# Create mapping dictionary for key renaming
key_map = {
    "encoders.camera.backbone": "module.img_backbone",
    "encoders.camera.neck": "module.img_neck",
    "encoders.camera.vtransform": "module.view_transform",
    "encoders.lidar.backbone": "module.pts_middle_encoder",
    "fuser": "module.fusion_layer",
    "decoder.backbone": "module.pts_backbone",
    "decoder.neck": "module.pts_neck",
    "heads.map": "module.seg_head"
}

# Remap the keys
new_state_dict = {}
for k, v in cp.items():
    for old_key, new_key in key_map.items():
        if k.startswith(old_key):
            new_k = k.replace(old_key, new_key)
            new_state_dict[new_k] = v
            break

# Save the new state dictionary
torch.save(new_state_dict, "bevfusion-seg-remapped.pth")