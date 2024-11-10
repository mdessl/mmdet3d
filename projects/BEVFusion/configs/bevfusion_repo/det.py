# FULL CONFIG of: configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

voxel_size = [0.075, 0.075, 0.2]

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

max_epochs = 6

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
         sweeps_num=0,
         use_dim=${use_dim},
    ),
    dict(type='LoadRadarPointsMultiSweeps',
         compensate_velocity=${radar_compensate_velocity},
         filtering=${radar_filtering},
         load_dim=18,
         max_num=${radar_max_points},
         normalize=${radar_normalize},
         sweeps_num=${radar_sweeps},
         use_dim=${radar_use_dims},
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
             info_path: ${'data/nuscenes/' + "nuscenes_dbinfos_train.pkl"}
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
         ['img', 'points', 'radar', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=
         ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix'],
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
    dict(type='LoadRadarPointsMultiSweeps',
         compensate_velocity=${radar_compensate_velocity},
         filtering=${radar_filtering},
         load_dim=18,
         max_num=${radar_max_points},
         normalize=${radar_normalize},
         sweeps_num=${radar_sweeps},
         use_dim=${radar_use_dims},
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
         ['img', 'points', 'radar', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=
         ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix'],
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

radar_sweeps = 6

radar_max_points = 2500

radar_use_dims = [0, 1, 2, 5, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

radar_compensate_velocity = true

radar_filtering = 'none'

radar_voxel_size = [0.8, 0.8, 8]

radar_jitter = 0

radar_normalize = false

model = dict(
    type='BEVFusion',
    encoders=dict(
        camera=dict(
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
                type='DepthLSSTransform',
                in_channels=256,
                out_channels=80,
                image_size=${image_size},
                feature_size=${[image_size[0] // 8, image_size[1] // 8]},
                xbound=[-54.0, 54.0, 0.3],
                ybound=[-54.0, 54.0, 0.3],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2,
            ),
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
                drop_path_rate=0.2,
                patch_norm=true,
                out_indices=[1, 2, 3],
                with_cp=false,
                convert_weights=true,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
                ),
            ),
        ),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=${point_cloud_range},
                voxel_size=${voxel_size},
                max_voxels=[120000, 160000],
            ),
            backbone=dict(
                type='SparseEncoder',
                in_channels=5,
                sparse_shape=[1440, 1440, 41],
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
    heads=dict(
        map=None,
        object=dict(
            type='TransFusionHead',
            num_proposals=200,
            auxiliary=true,
            in_channels=512,
            hidden_channel=128,
            num_classes=10,
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation='relu',
            train_cfg=dict(
                dataset='nuScenes',
                point_cloud_range=${point_cloud_range},
                grid_size=[1440, 1440, 41],
                voxel_size=${voxel_size},
                out_size_factor=8,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                assigner=dict(
                    type='HungarianAssigner3D',
                    iou_calculator=dict(
                        type='BboxOverlaps3D',
                        coordinate='lidar',
                    ),
                    cls_cost=dict(
                        type='FocalLossCost',
                        gamma=2.0,
                        alpha=0.25,
                        weight=0.15,
                    ),
                    reg_cost=dict(
                        type='BBoxBEVL1Cost',
                        weight=0.25,
                    ),
                    iou_cost=dict(
                        type='IoU3DCost',
                        weight=0.25,
                    ),
                ),
            ),
            test_cfg=dict(
                dataset='nuScenes',
                grid_size=[1440, 1440, 41],
                out_size_factor=8,
                voxel_size=${voxel_size[:2]},
                pc_range=${point_cloud_range[:2]},
                nms_type=None,
            ),
            common_heads=dict(
                center=[2, 2],
                height=[1, 2],
                dim=[3, 2],
                rot=[2, 2],
                vel=[2, 2],
            ),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=${point_cloud_range[:2]},
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                out_size_factor=8,
                voxel_size=${voxel_size[:2]},
                code_size=10,
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=true,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=1.0,
            ),
            loss_heatmap=dict(
                type='GaussianFocalLoss',
                reduction='mean',
                loss_weight=1.0,
            ),
            loss_bbox=dict(
                type='L1Loss',
                reduction='mean',
                loss_weight=0.25,
            ),
        ),
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

optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=0.01,
)

optimizer_config = dict(
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
    cumulative_iters=2,
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=0.001,
)

momentum_config = dict(
    policy='cyclic',
)