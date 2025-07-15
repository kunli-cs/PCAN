_base_ = '../../../_base_/default_runtime.py'

# model_cfg
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=28,
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=52,
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1., 1.],
    average_clips='prob')
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))
model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

dataset_type = 'PoseDataset'
data_root = './data/ma52/raw_videos/'
ann_file = './data/ma52/MA-52_openpose_28kp/MA52_train.pkl'
ann_file_val = './data/ma52/MA-52_openpose_28kp/MA52_val.pkl'
ann_file_test = './data/ma52/MA-52_openpose_28kp/MA52_test.pkl'


left_kp = [2, 3, 4, 8, 9, 10, 14, 16]
right_kp = [5, 6, 7, 11, 12, 13, 15, 17]


skeletons = ((4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
             (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
             (14, 0), (17, 15), (16, 14),
             (23, 4), (24, 4), (25, 4), (26, 4), (27, 4),
             (18, 7), (19, 7), (20, 7), (21, 7), (22, 7))
left_limb = (0, 1, 6, 7, 9, 11, 14, 16, 17, 18, 19, 20, 21)
right_limb = (2, 3, 4, 5, 8, 10, 13, 15, 22, 23, 24, 25, 26)

train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]
val_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]
test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=10,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[10,20],
        gamma=0.1)
]

load_from = './pretrained/rgbpose_conv3d_init.pth'  # noqa: E501

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=10)
