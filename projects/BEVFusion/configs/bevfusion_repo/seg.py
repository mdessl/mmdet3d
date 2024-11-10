# FULL CONFIG of: configs/nuscenes/seg/fusion-bev256d2-lss.yaml



point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

voxel_size = [0.1, 0.1, 0.2]

image_size = [256, 704]

seed = 0

deterministic = false

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
)

log_config = dict(
    interval=50,
    hooks=[
    dict(type='TextLoggerHook',
    ),
    dict(type='TensorboardLoggerHook',
    ),
],
)

load_from = None

resume_from = None

cudnn_benchmark = false

fp16 = dict(
    loss_scale=dict(
        growth_interval=2000,
    ),
)

max_epochs = 20

runner = dict(
    type='CustomEpochBasedRunner',
    max_epochs=${max_epochs},
)

dataset_type = 'NuScenesDataset'

dataset_root = 'data/nuscenes/'

gt_paste_stop_epoch = -1

reduce_beams = 32

load_dim = 5

use_dim = 5

load_augmented = None

augment2d = dict(
    resize=[
    [0.38, 0.55],
    [0.48, 0.48]
],
    rotate=[-5.4, 5.4],
    gridmask=dict(
        prob=0.0,
        fixed_prob=true,
    ),
)

augment3d = dict(
    scale=[0.9, 1.1],
    rotate=[-0.78539816, 0.78539816],
    translate=0.5,
)

object_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

map_classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']

input_modality = dict(
    use_lidar=true,
    use_camera=true,
    use_radar=false,
    use_map=false,
    use_external=false,
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=true,
    ),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_augmented=${load_augmented},
         load_dim=${load_dim},
         reduce_beams=${reduce_beams},
         use_dim=${use_dim},
    ),
    dict(type='LoadPointsFromMultiSweeps',
         load_augmented=${load_augmented},
         load_dim=${load_dim},
         pad_empty_sweeps=true,
         reduce_beams=${reduce_beams},
         remove_close=true,
         sweeps_num=9,
         use_dim=${use_dim},
    ),
    dict(type='LoadAnnotations3D',
         with_attr_label=false,
         with_bbox_3d=true,
         with_label_3d=true,
    ),
    dict(type='ObjectPaste',
         db_sampler=
         {
             dataset_root: ${dataset_root}
             info_path: ${dataset_root + "nuscenes_dbinfos_train.pkl"}
             rate: 1.0
             prepare:
             {
                 filter_by_difficulty: [-1]
                 filter_by_min_points:
                 {
                     car: 5
                     truck: 5
                     bus: 5
                     trailer: 5
                     construction_vehicle: 5
                     traffic_cone: 5
                     barrier: 5
                     motorcycle: 5
                     bicycle: 5
                     pedestrian: 5
                 }
             }
             classes: ${object_classes}
             sample_groups:
             {
                 car: 2
                 truck: 3
                 construction_vehicle: 7
                 bus: 4
                 trailer: 6
                 barrier: 2
                 motorcycle: 6
                 bicycle: 6
                 pedestrian: 2
                 traffic_cone: 2
             }
             points_loader:
             {
                 type: 'LoadPointsFromFile'
                 coord_type: 'LIDAR'
                 load_dim: ${load_dim}
                 use_dim: ${use_dim}
                 reduce_beams: ${reduce_beams}
             }
         },
         stop_epoch=${gt_paste_stop_epoch},
    ),
    dict(type='ImageAug3D',
         bot_pct_lim=[0.0, 0.0],
         final_dim=${image_size},
         is_train=true,
         rand_flip=true,
         resize_lim=${augment2d.resize[0]},
         rot_lim=${augment2d.rotate},
    ),
    dict(type='GlobalRotScaleTrans',
         is_train=true,
         resize_lim=${augment3d.scale},
         rot_lim=${augment3d.rotate},
         trans_lim=${augment3d.translate},
    ),
    dict(type='LoadBEVSegmentation',
         classes=${map_classes},
         dataset_root=${dataset_root},
         xbound=[-50.0, 50.0, 0.5],
         ybound=[-50.0, 50.0, 0.5],
    ),
    dict(type='RandomFlip3D',
    ),
    dict(type='PointsRangeFilter',
         point_cloud_range=${point_cloud_range},
    ),
    dict(type='ObjectRangeFilter',
         point_cloud_range=${point_cloud_range},
    ),
    dict(type='ObjectNameFilter',
         classes=${object_classes},
    ),
    dict(type='ImageNormalize',
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
    ),
    dict(type='GridMask',
         fixed_prob=${augment2d.gridmask.fixed_prob},
         max_epoch=${max_epochs},
         mode=1,
         offset=false,
         prob=${augment2d.gridmask.prob},
         ratio=0.5,
         rotate=1,
         use_h=true,
         use_w=true,
    ),
    dict(type='PointShuffle',
    ),
    dict(type='DefaultFormatBundle3D',
         classes=${object_classes},
    ),
    dict(type='Collect3D',
         keys=
         ['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'],
         meta_keys=
         ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix', 'scene_token'],
    ),
    dict(type='GTDepth',
         keyframe_only=true,
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=true,
    ),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_augmented=${load_augmented},
         load_dim=${load_dim},
         reduce_beams=${reduce_beams},
         use_dim=${use_dim},
    ),
    dict(type='LoadPointsFromMultiSweeps',
         load_augmented=${load_augmented},
         load_dim=${load_dim},
         pad_empty_sweeps=true,
         reduce_beams=${reduce_beams},
         remove_close=true,
         sweeps_num=9,
         use_dim=${use_dim},
    ),
    dict(type='LoadAnnotations3D',
         with_attr_label=false,
         with_bbox_3d=true,
         with_label_3d=true,
    ),
    dict(type='ImageAug3D',
         bot_pct_lim=[0.0, 0.0],
         final_dim=${image_size},
         is_train=false,
         rand_flip=false,
         resize_lim=${augment2d.resize[1]},
         rot_lim=[0.0, 0.0],
    ),
    dict(type='GlobalRotScaleTrans',
         is_train=false,
         resize_lim=[1.0, 1.0],
         rot_lim=[0.0, 0.0],
         trans_lim=0.0,
    ),
    dict(type='LoadBEVSegmentation',
         classes=${map_classes},
         dataset_root=${dataset_root},
         xbound=[-50.0, 50.0, 0.5],
         ybound=[-50.0, 50.0, 0.5],
    ),
    dict(type='PointsRangeFilter',
         point_cloud_range=${point_cloud_range},
    ),
    dict(type='ImageNormalize',
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
    ),
    dict(type='DefaultFormatBundle3D',
         classes=${object_classes},
    ),
    dict(type='Collect3D',
         keys=
         ['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'],
         meta_keys=
         ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix', 'scene_token'],
    ),
    dict(type='GTDepth',
         keyframe_only=true,
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=${dataset_type},
            dataset_root=${dataset_root},
            ann_file=${dataset_root + "nuscenes_infos_train.pkl"},
            pipeline=${train_pipeline},
            object_classes=${object_classes},
            map_classes=${map_classes},
            modality=${input_modality},
            test_mode=false,
            use_valid_flag=true,
            box_type_3d='LiDAR',
        ),
    ),
    val=dict(
        type=${dataset_type},
        dataset_root=${dataset_root},
        ann_file=${dataset_root + "nuscenes_infos_val.pkl"},
        pipeline=${test_pipeline},
        object_classes=${object_classes},
        map_classes=${map_classes},
        modality=${input_modality},
        test_mode=false,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type=${dataset_type},
        dataset_root=${dataset_root},
        ann_file=${dataset_root + "nuscenes_infos_val.pkl"},
        pipeline=${test_pipeline},
        object_classes=${object_classes},
        map_classes=${map_classes},
        modality=${input_modality},
        test_mode=true,
        box_type_3d='LiDAR',
    ),
)

evaluation = dict(
    interval=1,
    pipeline=${test_pipeline},
)

model = dict(
    type='BEVFusion',
    heads=dict(
        object=None,
        map=dict(
            type='BEVSegmentationHead',
            in_channels=512,
            grid_transform=dict(
                input_scope=[
    [-51.2, 51.2, 0.8],
    [-51.2, 51.2, 0.8]
],
                output_scope=[
    [-50, 50, 0.5],
    [-50, 50, 0.5]
],
            ),
            classes=${map_classes},
            loss='focal',
        ),
    ),
    encoders=dict(
        camera=dict(
            backbone=dict(
                type='SwinTransformer',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=true,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.3,
                patch_norm=true,
                out_indices=[1, 2, 3],
                with_cp=false,
                convert_weights=true,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
                ),
            ),
            neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=256,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(
                    type='BN2d',
                    requires_grad=true,
                ),
                act_cfg=dict(
                    type='ReLU',
                    inplace=true,
                ),
                upsample_cfg=dict(
                    mode='bilinear',
                    align_corners=false,
                ),
            ),
            vtransform=dict(
                type='LSSTransform',
                in_channels=256,
                out_channels=80,
                image_size=${image_size},
                feature_size=${[image_size[0] // 8, image_size[1] // 8]},
                xbound=[-51.2, 51.2, 0.4],
                ybound=[-51.2, 51.2, 0.4],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2,
            ),
        ),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=${point_cloud_range},
                voxel_size=${voxel_size},
                max_voxels=[90000, 120000],
            ),
            backbone=dict(
                type='SparseEncoder',
                in_channels=5,
                sparse_shape=[1024, 1024, 41],
                output_channels=128,
                order=['conv', 'norm', 'act'],
                encoder_channels=[
    [16, 16, 32],
    [32, 32, 64],
    [64, 64, 128],
    [128, 128]
],
                encoder_paddings=[
    [0, 0, 1],
    [0, 0, 1],
    [
    0,
    0,
    [1, 1, 0]
],
    [0, 0]
],
                block_type='basicblock',
            ),
        ),
    ),
    fuser=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256,
    ),
    decoder=dict(
        backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(
                type='BN',
                eps=0.001,
                momentum=0.01,
            ),
            conv_cfg=dict(
                type='Conv2d',
                bias=false,
            ),
        ),
        neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(
                type='BN',
                eps=0.001,
                momentum=0.01,
            ),
            upsample_cfg=dict(
                type='deconv',
                bias=false,
            ),
            use_conv_for_no_stride=true,
        ),
    ),
)

model_lidar = dict(
    type='BEVFusion',
    heads=dict(
        object=None,
        map=dict(
            type='BEVSegmentationHead',
            in_channels=256,
            grid_transform=dict(
                input_scope=[
    [-51.2, 51.2, 0.8],
    [-51.2, 51.2, 0.8]
],
                output_scope=[
    [-50, 50, 0.5],
    [-50, 50, 0.5]
],
            ),
            classes=${map_classes},
            loss='focal',
        ),
    ),
)

optimizer = dict(
    type='AdamW',
    lr=5e-05,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(
                decay_mult=0,
            ),
            relative_position_bias_table=dict(
                decay_mult=0,
            ),
        ),
    ),
)

optimizer_config = dict(
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
    cumulative_iters=4,
)

lr_config = dict(
    policy='cyclic',
)

momentum_config = dict(
    policy='cyclic',
)