model = dict(
    type='FasterRCNNParallel',
    backbone=dict(
        type='AdaptiveResNet',
        depth=101,
        img_scale=960,
        adaptive_kernel=16,
        hidden_dim=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        num_blocks=12,
        zero_head=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=9,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
dataset_type = 'KittiDatasetLP'
data_root = '../stereo_datasets'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileLP'),
    dict(type='LoadAnnotationsLP', with_bbox=True),
    dict(type='ResizeLP', img_scale=(960, 960), keep_ratio=False),
    dict(type='RandomFlipLP', flip_ratio=0.5),
    dict(
        type='NormalizeLP',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        mean_lp=[118, 118, 100],
        std_lp=[89, 90, 85],
        to_rgb=True),
    dict(type='PadLP', size_divisor=32),
    dict(type='DefaultFormatBundleLP'),
    dict(type='CollectLP', keys=['img', 'lp', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFileLP'),
    dict(
        type='MultiScaleFlipAugLP',
        img_scale=(960, 960),
        flip=False,
        transforms=[
            dict(type='ResizeLP', keep_ratio=False),
            dict(type='RandomFlipLP'),
            dict(
                type='NormalizeLP',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                mean_lp=[118, 118, 100],
                std_lp=[89, 90, 85],
                to_rgb=True),
            dict(type='PadLP', size_divisor=32),
            dict(type='ImageToTensorLP', keys=['img', 'lp']),
            dict(type='CollectLP', keys=['img', 'lp']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='KittiDatasetLP',
        ann_file='train.txt',
        img_prefix='training/image_2',
        lp_prefix='training/lp_image',
        pipeline=[
            dict(type='LoadImageFromFileLP'),
            dict(type='LoadAnnotationsLP', with_bbox=True),
            dict(type='ResizeLP', img_scale=(960, 960), keep_ratio=False),
            dict(type='RandomFlipLP', flip_ratio=0.5),
            dict(
                type='NormalizeLP',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                mean_lp=[118, 118, 100],
                std_lp=[89, 90, 85],
                to_rgb=True),
            dict(type='PadLP', size_divisor=32),
            dict(type='DefaultFormatBundleLP'),
            dict(type='CollectLP', keys=['img', 'lp', 'gt_bboxes', 'gt_labels'])
        ],
        data_root='../stereo_datasets'),
    val=dict(
        type='KittiDatasetLP',
        ann_file='val.txt',
        img_prefix='training/image_2',
        lp_prefix='training/lp_image',
        pipeline=[
            dict(type='LoadImageFromFileLP'),
            dict(
                type='MultiScaleFlipAugLP',
                img_scale=(960, 960),
                flip=False,
                transforms=[
                    dict(type='ResizeLP', keep_ratio=False),
                    dict(type='RandomFlipLP'),
                    dict(
                        type='NormalizeLP',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        mean_lp=[118, 118, 100],
                        std_lp=[89, 90, 85],
                        to_rgb=True),
                    dict(type='PadLP', size_divisor=32),
                    dict(type='ImageToTensorLP', keys=['img', 'lp']),
                    dict(type='CollectLP', keys=['img', 'lp'])
                ])
        ],
        data_root='../stereo_datasets'),
    test=dict(
        type='KittiDatasetLP',
        ann_file='val.txt',
        img_prefix='training/image_2',
        lp_prefix='training/lp_image',
        pipeline=[
            dict(type='LoadImageFromFileLP'),
            dict(
                type='MultiScaleFlipAugLP',
                img_scale=(960, 960),
                flip=False,
                transforms=[
                    dict(type='ResizeLP', keep_ratio=True),
                    dict(type='RandomFlipLP'),
                    dict(
                        type='NormalizeLP',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        mean_lp=[118, 118, 100],
                        std_lp=[89, 90, 85],
                        to_rgb=True),
                    dict(type='PadLP', size_divisor=32),
                    dict(type='ImageToTensorLP', keys=['img', 'lp']),
                    dict(type='CollectLP', keys=['img', 'lp'])
                ])
        ],
        data_root='../stereo_datasets'))
evaluation = dict(interval=12, metric='mAP')

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './Adaptive_result'
seed = 0
gpu_ids = range(0, 1)