_base_ = ["./bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

custom_imports = dict(
    imports=[
        'projects.BEVFusion.bevfusion',
        'projects.BEVFusion.bevfusion.utils'
    ],
    allow_failed_imports=False
)

model = dict(
    type="BEVFusion",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
    ),
    img_backbone=dict(
        type="mmdet.SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2], #2, 2, 6, 2
        num_heads=[3, 6, 12, 24], #3, 6, 12, 24
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",  # noqa: E251  # noqa: E501
        ),
    ),
    img_neck=dict(
        type="FPNC",
        final_dim=(800, 1400),  # (900, 1600),
        downsample=8,
        in_channels=[192, 384, 768],  # was: [96, 192, 384, 768
        out_channels=256,
        outC=256,
        use_adp=True,
        num_outs=3,
    ),
    view_transform=dict(
        type="LSSNoPoints",
        image_size=(800, 1400),  # (900, 1600),
        feature_size=(50, 88),  # Halve resolution
        xbound=[-45.0, 45.0, 0.9],  # Coarser 0.9m grid
        ybound=[-45.0, 45.0, 0.9],
        zbound=[-5.0, 3.0, 8.0],  # [min, max, bin_size]
        dbound=[4.0, 45.0, 1.0],  # [min, max, bin_size]
        downsample=8,
    ),
    fusion_layer=dict(type="ConvFuser", in_channels=[256, 256], out_channels=256),
)

train_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        to_float32=True,
        color_type="color",
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(
        type="ImageAug3D",
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True,
    ),
    dict(
        type="BEVFusionGlobalRotScaleTrans",
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5,
    ),
    dict(type="BEVFusionRandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="ObjectNameFilter",
        classes=[
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
    ),
    # Actually, 'GridMask' is not used here
    dict(
        type="GridMask",
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True,
    ),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=[
            "points",
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_bboxes",
            "gt_labels",
        ],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "transformation_3d_flow",
            "pcd_rotation",
            "pcd_scale_factor",
            "pcd_trans",
            "img_aug_matrix",
            "lidar_aug_matrix",
            "num_pts_feats",
        ],
    ),
]

test_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        to_float32=True,
        color_type="color",
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
    ),
    dict(
        type="ImageAug3D",
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False,
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    ),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "num_pts_feats",
        ],
    ),
]

train_dataloader = dict(
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

param_scheduler = [
    dict(type="LinearLR", start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        eta_min=0.85 / 0.95,
        begin=0,
        end=2.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        eta_min=1,
        begin=2.4,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.000005, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=16)


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=32)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1000,  # Save every 500 iterations
        by_epoch=False,  # Change to iteration-based saving
        max_keep_ckpts=10,
    ),  # Keep only the last 3 checkpoints to save disk space
)
del _base_.custom_hooks
# find_unused_parameters = False
