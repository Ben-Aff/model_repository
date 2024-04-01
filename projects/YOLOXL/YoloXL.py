default_scope = 'mmyolo'

file_client_args = dict(backend='disk')
backend_args = None
_file_client_args = dict(backend='disk')

data_root = '/home/renweilun/project/mmdetection/data/EXP1analysis/'
train_ann_file = 'annotations/instances_train2017.json'
val_ann_file = 'annotations/instances_val2017.json'
num_classes = 1
class_name = ('aokeng', )
train_batch_size_per_gpu = 16



model_test_cfg = dict(
    yolox_style=True,
    multi_label=True,
    score_thr=0.001,
    max_per_img=300,
    nms=dict(type='nms', iou_threshold=0.65))
img_scale = ( 640,640,  )



deepen_factor = 1.0
widen_factor = 1.0
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
batch_augments_interval = 10
weight_decay = 0.0005
loss_cls_weight = 1.0
loss_bbox_weight = 5.0
loss_obj_weight = 1.0
loss_bbox_aux_weight = 1.0
center_radius = 2.5
num_last_epochs = 50
random_affine_scaling_ratio_range = ( 0.1, 2,)
mixup_ratio_range = (0.8,1.6,)

save_epoch_intervals = 5
ema_momentum = 0.0001


# model settings
model = dict(
    type='YOLODetector',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,  # math.sqrt(5)
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    # TODO: Waiting for mmengine support
    use_syncbn=False,
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='YOLOXBatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=batch_augments_interval)
        ]),
    backbone=dict(
        type='YOLOXCSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOXHead',
        head_module=dict(
            type='YOLOXHeadModule',
            num_classes=num_classes,
            in_channels=256,
            feat_channels=256,
            widen_factor=widen_factor,
            stacked_convs=2,
            featmap_strides=(8, 16, 32),
            use_depthwise=False,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='mmdet.IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=loss_bbox_weight),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=loss_obj_weight),
        loss_bbox_aux=dict(
            type='mmdet.L1Loss',
            reduction='sum',
            loss_weight=loss_bbox_aux_weight)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.SimOTAAssigner',
            center_radius=center_radius,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
    test_cfg=model_test_cfg)



pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline_stage1 = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=random_affine_scaling_ratio_range,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        ratio_range=mixup_ratio_range,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='mmdet.PackDetInputs')
]



test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))]


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline_stage1))


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))

# # A-general测试
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#          type='YOLOv5CocoDataset',
#          data_root="/home/renweilun/project/mmdetection/data/EXP1baseline/test/test1/",
#         ann_file='annotations/instances_test2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline,))


# B-serious测试
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root="/home/renweilun/project/mmdetection/data/EXP1baseline/test/test2/",
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline))






# c-small的测试
test_dataloader  = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/home/renweilun/project/mmdetection/data/EXP1baseline/test/test3/',
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017'),
        test_mode=True,
        pipeline=test_pipeline,))

# #tool测试
test_dataloader = dict(
     batch_size=1,
    num_workers=4,
    persistent_workers=True,
     drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
     dataset=dict(
         type='YOLOv5CocoDataset',
         data_root='/home/renweilun/project/mmdetection/data/EXP1baseline/test/test_total/',
         ann_file='annotations/instances_test2017.json',
         data_prefix=dict(img='test2017/'),
         test_mode=True,
         pipeline=test_pipeline,
         backend_args=None))


val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')


# A-general的测试
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
    ann_file ='/home/renweilun/project/mmdetection/data/EXP1baseline/test/test1/annotations/instances_test2017.json',
    metric='bbox')

#B-serious的测试
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
ann_file = '/home/renweilun/project/mmdetection/data/EXP1baseline/test/test2/annotations/instances_test2017.json',
    metric='bbox')





# c-small的测试
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
    ann_file='/home/renweilun/project/mmdetection/data/EXP1baseline/test/test3/annotations/instances_test2017.json',
    metric='bbox')


# tool的测试
test_evaluator = dict(                                              ### 用于测试时候的评测
    type='mmdet.CocoMetric',
    ann_file= '/home/renweilun/project/mmdetection/data/EXP1baseline/test/test_total/annotations/instances_test2017.json',      ### 测试时的标注文件路径
    metric='bbox',                                                ### 需要计算的评价指标，'bbox'用于检测,'segm'用于实例分割
    format_only=False, )                                              ### 只将模型输出转换为 coco 的 JSON 格式并保存



base_lr = 0.01
max_epochs = 500

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(base_batch_size=2 * train_batch_size_per_gpu)





optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))


# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]





###default_runtime
# 运行设置(运行钩子设置)
# 默认钩子设置
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=ema_momentum,
        update_buffers=True,
        strict_load=False,
        priority=49)]



default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best=[ 'coco/bbox_mAP_50',  'coco/bbox_mAP',  ]),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='mmdet.DetVisualizationHook', draw=False, interval=1, show=False))



vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
        init_kwargs=dict(  project='bearinganalysis',
            entity='r770529885',
            name='YOLOXL_analysis_1122__500',
            tags=[  'A6000',  'num2', 'analysis', 'YOLOXL',  ],)),]



# # #测试用
vis_backends = [dict(type='LocalVisBackend')]



visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    save_dir='/home/renweilun/project/mmdetection/project/EXP1_analysis/YOLOX/train1122/visualizer')


randomness=dict(seed=0,diff_rank_seed=True,deterministic=True)




env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))




log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False




load_from = '/home/renweilun/project/mmdetection/project/EXP1_baseline/YOLOX/pretrain/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
