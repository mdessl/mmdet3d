_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], 
    allow_failed_imports=False
)

##############################################################################
# Common definitions (classes, ranges, etc.)
##############################################################################
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
image_size = [256, 704]
#voxel_size = [0.075, 0.075, 0.2]
#point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP'
)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

map_classes = [
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
    'carpark_area', 'divider'
]

##############################################################################
# Model definition: merges both lidar (base) + camera segmentation
##############################################################################
model = dict(
    type='BEVFusion',
    # Merge the voxelize part (for LiDAR) and the image normalization part
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        # from the lidar base:
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[90000, 120000], #changed from 120000, 160000
            voxelize_reduce=True
        ),
        # from the camera config:
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False
    ),

    # ------------------- LIDAR modules (from the base config) -----------
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1024, 1024, 41], # [1440, 1440, 41],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),

    # ------------------- Camera-related modules (from the second config) ----
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
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
            type='Pretrained',
            checkpoint=(
                'https://github.com/SwinTransformer/storage/releases/'
                'download/v1.0.0/swin_tiny_patch4_window7_224.pth'
            )
        )
    ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),

    view_transform=dict(
        type='LSSTransform',
        in_channels=256,
        out_channels=256,
        image_size=[256, 704],
        feature_size=[32, 88],  # Matches [image_size[0] // 8, image_size[1] // 8]
        xbound=[-51.2, 51.2, 0.4],  # Changed from [-54.0, 54.0, 0.3]
        ybound=[-51.2, 51.2, 0.4],  # Changed from [-54.0, 54.0, 0.3]
        zbound=[-10.0, 10.0, 20.0],  # Changed from previous values
        dbound=[1.0, 60.0, 0.5],
        downsample=2
    ),
    fusion_layer=dict(type='ConvFuser', in_channels=[256, 256], out_channels=256),
    seg_head=dict(
        type='BEVSegmentationHead',
        in_channels=512,
        grid_transform=dict(
            input_scope=[[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            output_scope=[[-50, 50, 0.5], [-50, 50, 0.5]],
        ),
        classes=map_classes,
        loss="focal"
    )
)

##############################################################################
# Pipeline (the second config overrides the base pipelines entirely)
##############################################################################
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True
    ),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5
    ),
    dict(
        type='LoadBEVSegmentation',
        classes=map_classes,
        dataset_root=data_root,
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
    ),
    dict(type='BEVFusionRandomFlip3D'),
    dict(
        type='PointsRangeFilter', 
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='ObjectRangeFilter', 
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='ObjectNameFilter',
        classes=class_names  # 10 detection classes
    ),
    # Actually 'ObjectSample' from the original base is overridden away here.
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True
    ),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
              'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
            'cam2lidar', 'ori_lidar2img', 'img_aug_matrix', 'box_type_3d',
            'sample_idx', 'lidar_path', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats', 'gt_masks_bev', 'sbnet_modality'
        ]
    )
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args
    ),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False
    ),
    dict(
        type='LoadBEVSegmentation',
        classes=map_classes,
        dataset_root=data_root,
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
            'cam2lidar', 'ori_lidar2img', 'img_aug_matrix', 'box_type_3d',
            'sample_idx', 'lidar_path', 'img_path', 'num_pts_feats',
            'lidar_aug_matrix', 'gt_masks_bev'
        ]
    )
]

##############################################################################
# Dataloaders (the second config references the base but overrides pipeline)
##############################################################################
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',  # from the base config
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            box_type_3d='LiDAR'
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

##############################################################################
# Evaluators
##############################################################################
val_evaluator = dict(
    type='NuScenesBEVFusionMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',  # for backward compatibility with NuScenesMetric base
    seg_classes=map_classes,  # for segmentation evaluation
    backend_args=backend_args
)
test_evaluator = val_evaluator

##############################################################################
# Hooks, runtime, schedules
##############################################################################
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Overridden schedule from the second config (6 epochs total)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True
    ),
    # momentum scheduler
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=2.4,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=2.4,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

auto_scale_lr = dict(enable=True, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# If you want to enable find_unused_parameters or add custom hooks:
find_unused_parameters = True

# Example custom hooks usage if needed (commented out if not working):
# custom_hooks = [
#     dict(
#         type='EarlyStoppingHook',
#         monitor='iter',
#         rule='greater',
#         stopping_threshold=2,  # Stop after 2 iterations
#         patience=0 
#     )
# ]
