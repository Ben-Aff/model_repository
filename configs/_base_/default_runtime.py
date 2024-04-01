default_scope = 'mmdet'

# 用户自定义钩子设置(按需求注册)
custom_hooks = [dict(type='NumClassCheckHook')]    # 检查在head中和 dataset中的classes的长度是否相匹配的钩子。


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best=[ 'coco/bbox_mAP_50',  'coco/bbox_mAP',  ]),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='mmdet.DetVisualizationHook', draw=False, interval=1, show=False))





env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),)

randomness=dict(seed=0,diff_rank_seed=True,deterministic=True)


#训练
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
        init_kwargs=dict(  project='bearinganalysis',
            entity='r770529885',
            name='YOLOXL_analysis_1122__500',
            tags=[  'A6000',  'num2', 'analysis', 'YOLOXL',  ],)),]



#测试
vis_backends = [dict(type='LocalVisBackend')]


visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    save_dir='-------------------------/visualizer')


log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)




log_level = 'INFO'
load_from = None
resume = False
