# 使用现有模型进行推理
from mmpretrain import get_model
from mmpretrain import list_models
from mmpretrain import inference_model
from mmpretrain import ImageClassificationInferencer

'''展示如何使用以下API：
list_models: 列举 MMPretrain 中所有可用模型名称
 get_model: 通过模型名称或模型配置文件获取模型
inference_model: 使用与模型相对应任务的推理器进行推理。主要用作快速 展示。如需配置进阶用法，还需要直接使用下列推理器。
推理器:
(1) ImageClassificationInferencer: 对给定图像执行图像分类,
输入参数(模型名称/配置文件路径，pretrained检查点的路径)，
输出参数
 'pred_scores'：预测得分数组。
'pred_label'：预测标签的索引。
'pred_score'：最高预测得分。
'pred_class'：预测类别。
 ImageRetrievalInferencer: 从给定的一系列图像中，检索与给定图像最相似的图像。
ImageCaptionInferencer: 生成给定图像的一段描述。
VisualQuestionAnsweringInferencer: 根据给定的图像回答问题。
VisualGroundingInferencer: 根据一段描述，从给定图像中找到一个与描述对应的对象。
TextToImageRetrievalInferencer: 从给定的一系列图像中，检索与给定文本最相似的图像。
ImageToTextRetrievalInferencer: 从给定的一系列文本中，检索与给定图像最相似的文本。
NLVRInferencer: 对给定的一对图像和一段文本进行自然语言视觉推理（NLVR 任务）。
FeatureExtractor: 通过视觉主干网络从图像文件提取特征。'''''

'列举可用的模型(列出 MMPreTrain 中的所有已支持的模型)'
# list_models()
'list_models 支持 Unix 文件名风格的模式匹配，可以使用 ** * ** 匹配任意字符'
# print(list_models("*clip"))
'可以使用推理器的 list_models 方法获取对应任务可用的所有模型'
from mmpretrain import ImageCaptionInferencer
#print(ImageCaptionInferencer.list_models())



'设置推理模型'
# model1 = get_model("vit-base-p32_clip-openai-in12k-pre_3rdparty_in1k-384px", pretrained=True)#加载官方的预训练模型
Test_Model = get_model("vit-base-p32_clip-openai-in12k-pre_3rdparty_in1k-384px", \
                       # 加载制定的权重文件
                       pretrained="/Users/ferraari/project/mmdetection/projects/CLIP/Pretrain/clip-vit-base-p32_openai-in12k-pre_3rdparty_in1k-384px_20221220-dc2e49ea.pth")
#使用几个head进行分类
# model = get_model("convnext-base_in21k-pre_3rdparty_in1k", head=dict(num_classes=10))
#移除模型的 neck，head 模块，直接从 backbone 中的 stage 1, 2, 3 输出
# model_headless = get_model("resnet18_8xb32_in1k", head=None, neck=None, backbone=dict(out_indices=(1, 2, 3)))
#构建推理器,其输出始终为一个结果列表,每个样本的结果都是一个字典。比如图像分类的结果是一个包含了pred_label、pred_score、pred_scores、pred_class
inferencer = ImageClassificationInferencer(Test_Model)



'单张图片可视化推理'
# image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/dog.jpg'
# result = inference_model(Test_Model, image, show=True)
# print(result['pred_class'])



"批量快速推理数据加载设置"
image_list = ['/Users/ferraari/project/mmdetection/mmpretrain/demo/dog.jpg',\
              '../demo/bird.JPEG'] * 16
results = inferencer(image_list, batch_size=8)#results内的元素个数为16➗2✖️4
print(len(results))
result1 = results[0]
result2 = result1[1]

'多张图像批量推理'
image_list = ['/Users/ferraari/project/mmdetection/mmpretrain/demo/dog.jpg',\
              '../demo/bird.JPEG'] * 16
results = inferencer(image_list, batch_size=8)
label = results[0]['pred_label']
classes = results[0]['pred_class']
score = results[0]['pred_score']
scores = results[0]['pred_scores']
print(f'The category of the image with label number {label} is {classes},score is {score}.' )




# '为推理器配置额外的参数，比如使用自己的配置文件和权重文件，在 指定设备上 上进行推理'
# image = ''
# config = ''
# checkpoint = ''
# inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='')
# result = inferencer(image)[0]
