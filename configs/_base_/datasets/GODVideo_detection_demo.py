# dataset settings
dataset_type = 'GODVideoDataset'
data_root = 'data/DEMO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImageFromFile', n_segment = 8),
    dict(type='LoadMultiAnnotations', with_bbox=True),
    dict(type='MultiExpand',
         mean=img_norm_cfg['mean'],
         to_rgb=img_norm_cfg['to_rgb'],
         ratio_range=(1, 4)),
    dict(type='MinIoURandomCrop',
         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
         min_crop_size=0.3),
    dict(type='Resize', img_scale=(72, 72), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='MultiTemporalShift', mean=img_norm_cfg['mean'],to_rgb=True, shift_range=4.0),
    dict(
        type='MultiPhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='MultiFramesNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='MultiCollect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadMultiImageFromFile',n_segment = 8),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(288, 288),#(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),#True
            dict(type='RandomFlip'),
            dict(type='MultiFramesNormalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,#16, #2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/GOD_val_demo.json',#'annotations/instances_train2017.json',
        img_prefix=data_root + 'demo_vis/',#'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/GOD_val_demo.json',#'annotations/instances_val2017.json',
        img_prefix=data_root + 'demo_vis/',#'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/GOD_val_demo.json',#'annotations/instances_val2017.json',
        img_prefix=data_root + 'demo_vis/',#'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')#2
