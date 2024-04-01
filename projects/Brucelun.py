default_scope = 'mmdet'  # 默认的注册器域名，默认从此注册器域中寻找模块。请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html
pretrained = '------------------------'          # 预训练模型

#集成文件
#configs/_base_/datasets/coco_detection.py
#configs/_base_/default_runtime.py
#configs/_base_/schedules/schedule_1x.py




# 项目结构

'''
 -(project name)_box/seg                              #  项目文件夹名_检测/分割
        -model  算法                                                 #   所用的模型算法文件名
        -pretrain
            -预训练的权重文件
        -配置文件
          - 型号-time                                                    #  所使用的模型算法-时间文件夹名
                -config                                                     #  配置文件
                -train
                    -数字文件夹
                        -训练日志
                    -训练保存的权重文件
                    -visualizer                              # 可视化后端存放文件夹
                -test
                    -数字文件夹
                        -训练日志
                    -(project name).pkl             # pkl文件,可用于分析test后续指标
                    -display_dir                            # 展示测试集预测结果和GT的对比
                    -Metric.json                                      # 测试评估指标文件(json)的存放目录

'''


# 训练及测试路径设置
'''
train.py --config ../projects/-----/-----.py 
            --work-dir ../projects/-----/train                      # 训练日志文件夹



test.py --config ../projects/-----/-----.py 
        --checkpoint ../projects/-----/train/-----.pth
        --work_dir ../projects/-----/test                           # 测试日志文件夹
        --out ../projects/-----/test/-----.pkl                      # pkl文件输出位置
        --show-dir ../projects/-----/test/display_dir               # 测试集预测结果和GT的对比
'''


# 数据存放结构
'''
-data
    -(project name)_box/seg                                     ### 训练数据存放
        -annotations
            --instances_train2017.json
            --instances_test2017.json
            --instances_val2017.json
        -train2017
        -test2017
        -val2017
    -(projcet name)_original                                    ### 原始数据存放
'''



# 模型配置
model = dict(
    type='YOLOX',                                                                       # 检测头名称
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(type='BatchSyncRandomResize',
                 random_size_range=(480, 512),
                 size_divisor=32,
                 interval=10)]),
    backbone=dict(                                                                      # backbone设置
        type='SwinTransformer',                                                         # backbone类型
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),                       #权重初始化
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=3,
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
            type='YOLOXHead',
            num_classes=1,
            in_channels=192,
            feat_channels=192,
            stacked_convs=2,
            strides=(8, 16, 32),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))





###coco_detection
# 数据集配置
dataset_type = 'CocoDataset'
###
data_root = '/home/renweilun/project/mmdetection/data/..........'      




###coco_detection
# 数据增强配置，包括两部分,图像增强 (mmdet/datasets/transforms/transforms.py) 和 Bbox增强 (mmdet/datasets/custom.py)
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),    ### 第 1 个流程，从文件路径里加载图像
    dict(type='LoadAnnotations',                                  # 第 2 个流程，对于当前图像，加载它的注释信息。
        with_bbox=True,                                           # 是否使用标注框(bounding box)，目标检测需要设置为True,分割设置为False。
        with_mask=True,                                           # 是否使用 instance mask，实例分割需要设置为 True,目标检测设置为False。
        poly2mask=True),                                          ### 是否将多边形掩码转换为实例掩码
    dict(type='Resize',                                           ##### 变化图像和其标注（bbox或者segment）
         scale=(1333, 800),                                       # 图像调整的最大尺寸(int or tuple)
        #scale_factor=1.0,                                        # (float or tuple[float]):缩放因子，默认为NONE，如果scale和scale_factor都设置了，将使用scale调整大小。
        #interpolation='bilinear'),                               # (str): 插值方法,包括 "nearest", "bilinear", "bicubic", "area", cv2用"lanczos" ，pillow用"nearest", "bilinear" for 'pillow'。默认为'bilinear'.
        keep_ratio=True),                                         # 是否保持图像的长宽比的形象
#  dict(type='RandomFlip',                                        # 翻转图像和其标注的数据增广流程（bbox或者segment）
#       prob=0.5,                                                 ##### 翻转的概率，默认为NONE
#       direction = 'horizontal'),                                #(str|list[str]):翻转方向.如果输入是一个列表（多个方向），长度必须等于prob，即指定每一个方向的概率。默认为'horizontal'。
#  dict(type='Expand',                                            ### 随机扩增图像/标注/mask.将原始图像随机放置在填充平均值的“ratio”x原始图像大小的画布上
#       mean=[0, 0, 0],  ### [sequence]: 数据填充的平均值
#       to_rgb=True,  ### 如果需要转换为RGB
#       ratio_range=(1, 2),  ### (sequence): expand的比例范围(要大于1)
#       prob=0.5),  # (float): 使用此增强方式的概率,默认为0.5
#  dict(type='MinIoURandomCrop',  ### 随机剪裁图像/标注/mask
#       min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),  ### (Sequence[float]): 最小的IoU阈值对于所有带有标注框的交叉点.
#       min_crop_size=0.3,  ### 最小裁剪大小,( h,w := a*h, a*w, where a >= min_crop_size).
#       bbox_clip_border=True),  # 是否裁剪图像边框外的对象,默认为True
#  dict(type='PhotoMetricDistortion',  ### 扭曲/光度失真
#       brightness_delta=32,  # (int),增加亮度，默认值为32.
#       contrast_range=(0.5, 1.5),  # (sequence)，对比度范围,默认范围为(0.5,1.5)
#       saturation_range=(0.5, 1.5),  # (sequence), 饱和度范围，默认范围为(0.5，1.5)
#       hue_delta=18),  # (int),色度的增加，默认值为18.
    dict(type='PackDetInputs')]                                 # 将数据转换为检测器输入格式的流程




test_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=backend_args),
    dict(type='Resize',
         scale=(1333, 800),
         keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),              # 如果测试集中没有标注，则删除此管道
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))]










###coco_detection
# 数据加载
train_dataloader = dict(                                                            ### 训练数据集设置
    batch_size=8,                                                                   ### batch size设置
    num_workers=10,                                                                 ### num_workers设置
    persistent_workers=True,                                                        ### 如果设置为True，dataloader在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(                                                                   ### 训练数据的采样器
        type='DefaultSampler',                                                      ### 默认的采样器，同时支持分布式和非分布式设置
        shuffle=True),                                                              ### 训练机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'),                             ### 默认的batch采样器，用于保证 batch 中的图片具有相似的长宽比，从而可以更好地利用显存
    dataset=dict(
                type=dataset_type,                                                  ### 需要载入数据集设置
                data_root=data_root,                                                ### 数据集根目录
                ann_file='annotations/instances_train2017.json',                    ### 标注文件根目录
                data_prefix=dict(img='im/'),                                 ### 训练图片路径
                filter_cfg=dict(filter_empty_gt=True, min_size=32),                 ### 图片和标注的过滤配置
                pipeline=train_pipeline,                                            ### 训练数据增强策略
                backend_args=backend_args))





val_dataloader = dict(                                                              ### 验证数据集设置
    batch_size=4,                                                                   ### batch size 设置,验证时基本设置为1
    num_workers=10,                                                                 ### num_workers设置
    persistent_workers=True,                                                        ### 如果设置为True，dataloader在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,                                                                ### 是否丢弃最后未能组成一个批次的数据
    sampler=dict(                                                                   ### 验证数据的采样器
        type='DefaultSampler',                                                      ### 默认的采样器，同时支持分布式和非分布式设置
        shuffle=False),                                                             ### 验证时不打乱数据顺序
    dataset=dict(
        type=dataset_type,                                                          ### 需要载入数据集设置
        data_root=data_root,                                                        ### 数据集根目录
        ann_file='annotations/instances_val2017.json',                              ### 标注文件根目录
        data_prefix=dict(img='val2017/'),                                           ### 验证图片路径
        test_mode=True,                                                             ### 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline,                                                     ### 验证数据增强策略
        backend_args=backend_args))





test_dataloader = dict(                                                             ### 测试数据集设置
    batch_size=4,                                                                   ### batch size 设置
    num_workers=10,                                                                 ### num_workers设置
    persistent_workers=True,                                                       ### 如果设置为True，dataloader在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,                                                                ### 是否丢弃最后未能组成一个批次的数据
    sampler=dict(                                                                   ### 测试数据的采样器
        type='DefaultSampler',                                                      ### 默认的采样器，同时支持分布式和非分布式设置
        shuffle=True),                                                              ### 测试时打乱数据顺序
    dataset=dict(
        type=dataset_type,                                                          ### 需要载入数据集设置
        data_root=data_root,                                                        ### 数据集根目录
        ann_file=data_root+'annotations/instances_test2017.json',                   ### 标注文件根目录
        data_prefix=dict(img='test2017/'),                                          ### 测试图片路径
        test_mode=True,                                                             ### 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline))                                                    ### 测试数据增强策略

###coco_detection
# 模型评测器设置（mmdet/evaluation/metrics/coco_metric.py）
val_evaluator = dict(                                               ### 用于验证时的评测
    type='CocoMetric',                                              ### 评测方式:'CocoDataset'/'VOCMetric'
    ann_file=data_root + 'annotations/instances_val2017.json',      ### 验证时的标注文件路径
    metric=['bbox'],                                                ### 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    # classwise (bool):                                           ###是否按类别评价指标，默认为否
    format_only=True,                                           ### 只将模型输出转换为coco的json格式并进行保存，同时输出评估文件
    outfile_prefix='projects/........test/valMetric'    )

test_evaluator = dict(                                              ### 用于测试时候的评测
    type='CocoMetric',
    ann_file=data_root + 'annotations/inference_test2017.json',      ### 测试时的标注文件路径
    metric=['bbox'],                                                ### 需要计算的评价指标，'bbox'用于检测,'segm'用于实例分割
    format_only=True,                                               ### 将模型输出转换为 coco 的 JSON 格式并保存，同时输出评估文件
    outfile_prefix='projects/........test/testMetric')                                              ### 评估文件夹Metric的存放位置,projects/project name/test/Metric









### schedule_1x
# 训练和测试循环方式配置，可以设置最大训练轮次和验证间隔。
train_cfg = dict(                                            ### 训练循环方式设置
    type='EpochBasedTrainLoop',                              ### 训练循环的类型，默认EpochBasedTrainLoop，基于epoch的循环,其他可选'IterBasedTrainLoop'-基于迭代的循环训练和'_InfiniteDataloaderIterator'-无限数据加载器迭代器包装。
    max_epochs=400,                                          ### max_epochs(int):总的训练epoch数.
    val_begin=1,                                             # val_begin(int):开始验证的起始epoch，默认为1
    val_interval=1)                                          ### 验证间隔，默认每个1 epoch 验证一次
# train_cfg = dict(
#     type = 'IterBasedTrainLoop',                             # 基于迭代的循环训练
#     max_iters = 10000,                                       # int，总的训练迭代次数
#     val_begin = 1 ,                                          # int, 开始验证的起始迭代，默认是1
#     val_interval = 1000)                                     # int,验证间隔，默认每1000次迭代验证一次
val_cfg = dict(type='ValLoop')                               # 验证循环的类型
test_cfg = dict(type='TestLoop')                             # 测试循环的类型





### schedule_1x
# 优化器设置
optim_wrapper = dict(
    type='OptimWrapper',                                            ### 优化器封装的类型，可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(                                                 ### 优化器配置。支持 PyTorch的各种优化器
        type='SGD',                                                 ### 随机梯度优化器的类型
        lr=0.02,                                                    ### 基础学习率
        momentum=0.9,                                               ### 带动量的随机梯度下降
        weight_decay=0.0001),                                       ### 权重衰减
    clip_grad=None)                                                 ### 梯度裁剪的配置，设置为 None 关闭梯度裁剪。




### schedule_1x
# 学习率设置
param_scheduler = [
    dict(
        type='LinearLR',                                            ### 使用线性学习率预热
        start_factor=0.001,                                         ### 学习率预热的系数
        by_epoch=False,                                             ### 按 iteration 更新预热学习率
        begin=0,                                                    ### 从第一个 iteration 开始
        end=500),                                                   ### 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',                                         ### 在训练过程中使用 multi step 学习率策略
        by_epoch=True,                                              ### 按 epoch 更新学习率
        begin=0,                                                    ### 从第一个 epoch 开始
        end=12,                                                     ### 到第 12 个 epoch 结束
        milestones=[8, 11],                                         ### 在哪几个 epoch 进行学习率衰减
        gamma=0.1)                                                  ### 学习率衰减系数
]


### schedule_1x
auto_scale_lr = dict(enable=False,                                  ### 是否启用自适应学习率，默认不启用
                     base_batch_size=16)                            ### base_batch_size = (8 GPUs) x (2 samples per GPU)














###default_runtime
# 运行设置(运行钩子设置)
# 默认钩子设置
default_hooks = dict(
checkpoint=dict(type='CheckpointHook',             ### 定期保存检查点的钩子。
                by_epoch=True,                     #by_epoch (bool): 以epoch/迭代周期的形式保存关键点,默认为True，保存方式为epoch
                interval=1,                        ### interval(int),保存周期为interval，如果by_epoch=True, interval表示epoch，否则表示迭代周期
                max_keep_ckpts=5,                  ### 只保留最新的max_keep_ckpts个，权重文件数超过max_keep_ckpts时，前面的权重会被删除。
                # save_optimizer=True,               #是否将优化器state_dict保存在检查点信息上。它通常用于恢复实验。默认为True。
                # save_param_scheduler=True,         #是否保存param_scheduler state_dict在关键点信息上。它通常用于恢复实验。默认为True。
                #save_best=['coco/bbox_mAP_50']), ###save_best (str, List[str], optional),如果指定了一个标准，它将在评估最佳标准检查点如果传递了一个指标列表，它将度量与传递的指标相对应的一组最佳检查点。如果是'auto '，返回的第一个键结果将被使用。默认为None。
                ),
timer=dict(type='IterTimerHook'),                  ### 记录 “data_time” 用于加载数据和 “time” 用于模型训练步骤的钩子,该钩子没有可配置的参数，因此不需要配置它。
logger=dict(type='LoggerHook',                     ### 日志d打印的钩子,从Runner的不同组件收集日志，并将其写入终端、JSON文件、tensorboard和wandb等。
            interval=50),                          ### 间隔 50 个 iteration 打印一次日志
param_scheduler=dict(type='ParamSchedulerHook'),   ### 遍历Runner的所有优化器参数调度器，并调用它们的step方法按顺序更新优化器参数。该钩子没有可配置的参数，因此不需要配置它。
sampler_seed=dict(type='DistSamplerSeedHook'),     ### 为采样器和批处理采样器设置种子的钩子。
visualization=dict(type='DetVisualizationHook',    ### 设置用于可视化验证和测试过程预测结果的钩子。
                   draw=True,                      ### 是否将验证和测试时结果绘制出来
                   interval=1,                     ### 控制在DetVisualizationHook启用时存储或显示验证或测试结果的间隔，单位为迭代次数。
                   show=False)                      ### 控制是否展示可视化验证或测试的结果。
)



### 可视化设置
# 可视器后端设置,如TensorBoard或WandB，从而方便用户使用这些可视化工具来分析和监控训练过程。
vis_backends = [
    dict(type='LocalVisBackend'),                                                           ### 使用本地可视化,将所有训练信息保存在本地文件夹中，默认不用配置
    #dict(type='TensorboardVisBackend')                                                     ### 使用tensorboard可视化
    #dict(type='NeptuneLoggerHook',                                                         ### 使用NeptuneLoggerHook
#         init_kwarhs={
#           'project':'<YOUR_WORKSPACE/YOUR_PROJECT>'})       ###YOUR_WORKSPACE 是账号名，YOUR_PROJECT 是项目名
     dict(  type='WandbVisBackend',
                init_kwargs={'project': 'bearing_aigc_yoloxs_20230817'},)]




visualizer = dict(                                                                              ### 可视化设置，写入vis_backends设置的后端，
    type='DetLocalVisualizer',                                                                  ### 直接使用DetLocalVisualizer来支持任务，默认不用配置
    vis_backends=vis_backends,                                                                  ### 可视化后端设置
    name='visualizer',                                                                          ### 默认，不用配置
    save_dir='./train/visualizer')                                                   # 可视化后端存放位置








# 用户自定义钩子设置(按需求注册)
custom_hooks = [dict(type='NumClassCheckHook'),    # 检查在head中和 dataset中的classes的长度是否相匹配的钩子。
#               dict(type='EMAHook', ema_type='StochasticWeightAverage')
#在训练过程中对模型进行指数移动平均运算，目的是提高模型的鲁棒性.由指数移动平均生成的模型仅用于验证和测试，并不影响训练。
#EMAHook默认使用ExponentialMovingAverage，可选值为stochasticweightaaverage和MomentumAnnealingEMA。其他平均策略可以通过设置ema类型来使用。
#               dict(type='EmptyCacheHook', after_epoch=True),
#调用torch.cuda.empty cache()来释放所有未占用的GPU缓存内存。释放内存的时间可以通过设置before epoch、after iter和after epoch等参数来控制，
#分别表示每个epoch开始之前、每次迭代之后和每个epoch之后。
#                dict(type='SyncBuffersHook')
# 在分布式训练的每个epoch结束时同步模型的缓冲区，例如BN层的running mean和running var。
]




env_cfg = dict(
                cudnn_benchmark=False,                         ### 启用后算法的前期会比较慢，但算法跑起来以后会非常快。
                mp_cfg=dict(mp_start_method='fork',            ### 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能存在隐患。
                opencv_num_threads=0),                         ### 关闭 opencv 的多线程以避免系统超负荷
                dist_cfg=dict(backend='nccl')                 ### 分布式相关设置
)



log_processor = dict(
    type='LogProcessor',                                       ### 日志处理器用于处理运行时日志
    window_size=50,                                            ### 日志数值的平滑窗口
    by_epoch=True)                                             # 是否使用 epoch 格式的日志。需要与训练循环的类型保存一致。
log_level = 'INFO'                                             ### 日志等级



load_from = r"---------------------------------------------"   # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False                                                 # 是否从检查点恢复训练.如果指定,若load_from为 None,它将恢复`work_dir`中的最新检查点。









