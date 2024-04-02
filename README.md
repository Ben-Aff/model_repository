## 📊 Overview of Benchmark and Model Zoo(Detection)
<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Panoptic Segmentation</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/fast_rcnn">Fast R-CNN (ICCV'2015)</a></li>
            <li><a href="configs/faster_rcnn">Faster R-CNN (NeurIPS'2015)</a></li>
            <li><a href="configs/rpn">RPN (NeurIPS'2015)</a></li>
            <li><a href="configs/ssd">SSD (ECCV'2016)</a></li>
            <li><a href="configs/retinanet">RetinaNet (ICCV'2017)</a></li>
            <li><a href="configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a></li>
            <li><a href="configs/yolo">YOLOv3 (ArXiv'2018)</a></li>
            <li><a href="configs/cornernet">CornerNet (ECCV'2018)</a></li>
            <li><a href="configs/grid_rcnn">Grid R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/guided_anchoring">Guided Anchoring (CVPR'2019)</a></li>
            <li><a href="configs/fsaf">FSAF (CVPR'2019)</a></li>
            <li><a href="configs/centernet">CenterNet (CVPR'2019)</a></li>
            <li><a href="configs/libra_rcnn">Libra R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/tridentnet">TridentNet (ICCV'2019)</a></li>
            <li><a href="configs/fcos">FCOS (ICCV'2019)</a></li>
            <li><a href="configs/reppoints">RepPoints (ICCV'2019)</a></li>
            <li><a href="configs/free_anchor">FreeAnchor (NeurIPS'2019)</a></li>
            <li><a href="configs/cascade_rpn">CascadeRPN (NeurIPS'2019)</a></li>
            <li><a href="configs/foveabox">Foveabox (TIP'2020)</a></li>
            <li><a href="configs/double_heads">Double-Head R-CNN (CVPR'2020)</a></li>
            <li><a href="configs/atss">ATSS (CVPR'2020)</a></li>
            <li><a href="configs/nas_fcos">NAS-FCOS (CVPR'2020)</a></li>
            <li><a href="configs/centripetalnet">CentripetalNet (CVPR'2020)</a></li>
            <li><a href="configs/autoassign">AutoAssign (ArXiv'2020)</a></li>
            <li><a href="configs/sabl">Side-Aware Boundary Localization (ECCV'2020)</a></li>
            <li><a href="configs/dynamic_rcnn">Dynamic R-CNN (ECCV'2020)</a></li>
            <li><a href="configs/detr">DETR (ECCV'2020)</a></li>
            <li><a href="configs/paa">PAA (ECCV'2020)</a></li>
            <li><a href="configs/vfnet">VarifocalNet (CVPR'2021)</a></li>
            <li><a href="configs/sparse_rcnn">Sparse R-CNN (CVPR'2021)</a></li>
            <li><a href="configs/yolof">YOLOF (CVPR'2021)</a></li>
            <li><a href="configs/yolox">YOLOX (CVPR'2021)</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR (ICLR'2021)</a></li>
            <li><a href="configs/tood">TOOD (ICCV'2021)</a></li>
            <li><a href="configs/ddod">DDOD (ACM MM'2021)</a></li>
            <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
            <li><a href="configs/conditional_detr">Conditional DETR (ICCV'2021)</a></li>
            <li><a href="configs/dab_detr">DAB-DETR (ICLR'2022)</a></li>
            <li><a href="configs/dino">DINO (ICLR'2023)</a></li>
            <li><a href="configs/glip">GLIP (CVPR'2022)</a></li>
            <li><a href="configs/ddq">DDQ (CVPR'2023)</a></li>
            <li><a href="projects/DiffusionDet">DiffusionDet (ArXiv'2023)</a></li>
            <li><a href="projects/EfficientDet">EfficientDet (CVPR'2020)</a></li>
            <li><a href="projects/ViTDet">ViTDet (ECCV'2022)</a></li>
            <li><a href="projects/Detic">Detic (ECCV'2022)</a></li>
            <li><a href="projects/CO-DETR">CO-DETR (ICCV'2023)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/mask_rcnn">Mask R-CNN (ICCV'2017)</a></li>
          <li><a href="configs/cascade_rcnn">Cascade Mask R-CNN (CVPR'2018)</a></li>
          <li><a href="configs/ms_rcnn">Mask Scoring R-CNN (CVPR'2019)</a></li>
          <li><a href="configs/htc">Hybrid Task Cascade (CVPR'2019)</a></li>
          <li><a href="configs/yolact">YOLACT (ICCV'2019)</a></li>
          <li><a href="configs/instaboost">InstaBoost (ICCV'2019)</a></li>
          <li><a href="configs/solo">SOLO (ECCV'2020)</a></li>
          <li><a href="configs/point_rend">PointRend (CVPR'2020)</a></li>
          <li><a href="configs/detectors">DetectoRS (ArXiv'2020)</a></li>
          <li><a href="configs/solov2">SOLOv2 (NeurIPS'2020)</a></li>
          <li><a href="configs/scnet">SCNet (AAAI'2021)</a></li>
          <li><a href="configs/queryinst">QueryInst (ICCV'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
          <li><a href="configs/condinst">CondInst (ECCV'2020)</a></li>
          <li><a href="projects/SparseInst">SparseInst (CVPR'2022)</a></li>
          <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
          <li><a href="configs/boxinst">BoxInst (CVPR'2021)</a></li>
          <li><a href="projects/ConvNeXt-V2">ConvNeXt-V2 (Arxiv'2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/panoptic_fpn">Panoptic FPN (CVPR'2019)</a></li>
          <li><a href="configs/maskformer">MaskFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
          <li><a href="configs/XDecoder">XDecoder (CVPR'2023)</a></li>
        </ul>
      </td>
      <td>
        </ul>
          <li><b>Contrastive Learning</b></li>
        <ul>
        <ul>
          <li><a href="configs/selfsup_pretrain">SwAV (NeurIPS'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCo (CVPR'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCov2 (ArXiv'2020)</a></li>
        </ul>
        </ul>
        </ul>
          <li><b>Distillation</b></li>
        <ul>
        <ul>
          <li><a href="configs/ld">Localization Distillation (CVPR'2022)</a></li>
          <li><a href="configs/lad">Label Assignment Distillation (WACV'2022)</a></li>
        </ul>
        </ul>
          <li><b>Semi-Supervised Object Detection</b></li>
        <ul>
        <ul>
          <li><a href="configs/soft_teacher">Soft Teacher (ICCV'2021)</a></li>
        </ul>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>VGG (ICLR'2015)</li>
        <li>ResNet (CVPR'2016)</li>
        <li>ResNeXt (CVPR'2017)</li>
        <li>MobileNetV2 (CVPR'2018)</li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/empirical_attention">Generalized Attention (ICCV'2019)</a></li>
        <li><a href="configs/gcnet">GCNet (ICCVW'2019)</a></li>
        <li><a href="configs/res2net">Res2Net (TPAMI'2020)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/pvt">PVT (ICCV'2021)</a></li>
        <li><a href="configs/swin">Swin (CVPR'2021)</a></li>
        <li><a href="configs/pvt">PVTv2 (ArXiv'2021)</a></li>
        <li><a href="configs/resnet_strikes_back">ResNet strikes back (ArXiv'2021)</a></li>
        <li><a href="configs/efficientnet">EfficientNet (ArXiv'2021)</a></li>
        <li><a href="configs/convnext">ConvNeXt (CVPR'2022)</a></li>
        <li><a href="projects/ConvNeXt-V2">ConvNeXtv2 (ArXiv'2023)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/pafpn">PAFPN (CVPR'2018)</a></li>
        <li><a href="configs/nas_fpn">NAS-FPN (CVPR'2019)</a></li>
        <li><a href="configs/carafe">CARAFE (ICCV'2019)</a></li>
        <li><a href="configs/fpg">FPG (ArXiv'2020)</a></li>
        <li><a href="configs/groie">GRoIE (ICPR'2020)</a></li>
        <li><a href="configs/dyhead">DyHead (CVPR'2021)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/ghm">GHM (AAAI'2019)</a></li>
          <li><a href="configs/gfl">Generalized Focal Loss (NeurIPS'2020)</a></li>
          <li><a href="configs/seesaw_loss">Seasaw Loss (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/faster_rcnn/faster-rcnn_r50_fpn_ohem_1x_coco.py">OHEM (CVPR'2016)</a></li>
          <li><a href="configs/gn">Group Normalization (ECCV'2018)</a></li>
          <li><a href="configs/dcn">DCN (ICCV'2017)</a></li>
          <li><a href="configs/dcnv2">DCNv2 (CVPR'2019)</a></li>
          <li><a href="configs/gn+ws">Weight Standardization (ArXiv'2019)</a></li>
          <li><a href="configs/pisa">Prime Sample Attention (CVPR'2020)</a></li>
          <li><a href="configs/strong_baselines">Strong Baselines (CVPR'2021)</a></li>
          <li><a href="configs/resnet_strikes_back">Resnet strikes back (ArXiv'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 📊 Overview of Benchmark and Model Zoo(Pretrain)
<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Supported Backbones</b>
      </td>
      <td>
        <b>Self-supervised Learning</b>
      </td>
      <td>
        <b>Multi-Modality Algorithms</b>
      </td>
      <td>
        <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="configs/vgg">VGG</a></li>
        <li><a href="configs/resnet">ResNet</a></li>
        <li><a href="configs/resnext">ResNeXt</a></li>
        <li><a href="configs/seresnet">SE-ResNet</a></li>
        <li><a href="configs/seresnet">SE-ResNeXt</a></li>
        <li><a href="configs/regnet">RegNet</a></li>
        <li><a href="configs/shufflenet_v1">ShuffleNet V1</a></li>
        <li><a href="configs/shufflenet_v2">ShuffleNet V2</a></li>
        <li><a href="configs/mobilenet_v2">MobileNet V2</a></li>
        <li><a href="configs/mobilenet_v3">MobileNet V3</a></li>
        <li><a href="configs/swin_transformer">Swin-Transformer</a></li>
        <li><a href="configs/swin_transformer_v2">Swin-Transformer V2</a></li>
        <li><a href="configs/repvgg">RepVGG</a></li>
        <li><a href="configs/vision_transformer">Vision-Transformer</a></li>
        <li><a href="configs/tnt">Transformer-in-Transformer</a></li>
        <li><a href="configs/res2net">Res2Net</a></li>
        <li><a href="configs/mlp_mixer">MLP-Mixer</a></li>
        <li><a href="configs/deit">DeiT</a></li>
        <li><a href="configs/deit3">DeiT-3</a></li>
        <li><a href="configs/conformer">Conformer</a></li>
        <li><a href="configs/t2t_vit">T2T-ViT</a></li>
        <li><a href="configs/twins">Twins</a></li>
        <li><a href="configs/efficientnet">EfficientNet</a></li>
        <li><a href="configs/edgenext">EdgeNeXt</a></li>
        <li><a href="configs/convnext">ConvNeXt</a></li>
        <li><a href="configs/hrnet">HRNet</a></li>
        <li><a href="configs/van">VAN</a></li>
        <li><a href="configs/convmixer">ConvMixer</a></li>
        <li><a href="configs/cspnet">CSPNet</a></li>
        <li><a href="configs/poolformer">PoolFormer</a></li>
        <li><a href="configs/inception_v3">Inception V3</a></li>
        <li><a href="configs/mobileone">MobileOne</a></li>
        <li><a href="configs/efficientformer">EfficientFormer</a></li>
        <li><a href="configs/mvit">MViT</a></li>
        <li><a href="configs/hornet">HorNet</a></li>
        <li><a href="configs/mobilevit">MobileViT</a></li>
        <li><a href="configs/davit">DaViT</a></li>
        <li><a href="configs/replknet">RepLKNet</a></li>
        <li><a href="configs/beit">BEiT</a></li>
        <li><a href="configs/mixmim">MixMIM</a></li>
        <li><a href="configs/efficientnet_v2">EfficientNet V2</a></li>
        <li><a href="configs/revvit">RevViT</a></li>
        <li><a href="configs/convnext_v2">ConvNeXt V2</a></li>
        <li><a href="configs/vig">ViG</a></li>
        <li><a href="configs/xcit">XCiT</a></li>
        <li><a href="configs/levit">LeViT</a></li>
        <li><a href="configs/riformer">RIFormer</a></li>
        <li><a href="configs/glip">GLIP</a></li>
        <li><a href="configs/sam">ViT SAM</a></li>
        <li><a href="configs/eva02">EVA02</a></li>
        <li><a href="configs/dinov2">DINO V2</a></li>
        <li><a href="configs/hivit">HiViT</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="configs/mocov2">MoCo V1 (CVPR'2020)</a></li>
        <li><a href="configs/simclr">SimCLR (ICML'2020)</a></li>
        <li><a href="configs/mocov2">MoCo V2 (arXiv'2020)</a></li>
        <li><a href="configs/byol">BYOL (NeurIPS'2020)</a></li>
        <li><a href="configs/swav">SwAV (NeurIPS'2020)</a></li>
        <li><a href="configs/densecl">DenseCL (CVPR'2021)</a></li>
        <li><a href="configs/simsiam">SimSiam (CVPR'2021)</a></li>
        <li><a href="configs/barlowtwins">Barlow Twins (ICML'2021)</a></li>
        <li><a href="configs/mocov3">MoCo V3 (ICCV'2021)</a></li>
        <li><a href="configs/beit">BEiT (ICLR'2022)</a></li>
        <li><a href="configs/mae">MAE (CVPR'2022)</a></li>
        <li><a href="configs/simmim">SimMIM (CVPR'2022)</a></li>
        <li><a href="configs/maskfeat">MaskFeat (CVPR'2022)</a></li>
        <li><a href="configs/cae">CAE (arXiv'2022)</a></li>
        <li><a href="configs/milan">MILAN (arXiv'2022)</a></li>
        <li><a href="configs/beitv2">BEiT V2 (arXiv'2022)</a></li>
        <li><a href="configs/eva">EVA (CVPR'2023)</a></li>
        <li><a href="configs/mixmim">MixMIM (arXiv'2022)</a></li>
        <li><a href="configs/itpn">iTPN (CVPR'2023)</a></li>
        <li><a href="configs/spark">SparK (ICLR'2023)</a></li>
        <li><a href="configs/mff">MFF (ICCV'2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="configs/blip">BLIP (arxiv'2022)</a></li>
        <li><a href="configs/blip2">BLIP-2 (arxiv'2023)</a></li>
        <li><a href="configs/ofa">OFA (CoRR'2022)</a></li>
        <li><a href="configs/flamingo">Flamingo (NeurIPS'2022)</a></li>
        <li><a href="configs/chinese_clip">Chinese CLIP (arxiv'2022)</a></li>
        <li><a href="configs/minigpt4">MiniGPT-4 (arxiv'2023)</a></li>
        <li><a href="configs/llava">LLaVA (arxiv'2023)</a></li>
        <li><a href="configs/otter">Otter (arxiv'2023)</a></li>
        </ul>
      </td>
      <td>
      Image Retrieval Task:
        <ul>
        <li><a href="configs/arcface">ArcFace (CVPR'2019)</a></li>
        </ul>
      Training&Test Tips:
        <ul>
        <li><a href="https://arxiv.org/abs/1909.13719">RandAug</a></li>
        <li><a href="https://arxiv.org/abs/1805.09501">AutoAug</a></li>
        <li><a href="mmpretrain/datasets/samplers/repeat_aug.py">RepeatAugSampler</a></li>
        <li><a href="mmpretrain/models/tta/score_tta.py">TTA</a></li>
        <li>...</li>
        </ul>
      </td>
  </tbody>
</table>

## 📊 Overview of Benchmark and Model Zoo (YOLO)
<details open>
<summary><b>Supported Algorithms</b></summary>

- [x] [YOLOv5](configs/yolov5)
- [ ] [YOLOv5u](configs/yolov5/yolov5u) (Inference only)
- [x] [YOLOX](configs/yolox)
- [x] [RTMDet](configs/rtmdet)
- [x] [RTMDet-Rotated](configs/rtmdet)
- [x] [YOLOv6](configs/yolov6)
- [x] [YOLOv7](configs/yolov7)
- [x] [PPYOLOE](configs/ppyoloe)
- [x] [YOLOv8](configs/yolov8)

<details open>
<div align="center">
  <b>Module Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>YOLOv5CSPDarknet</li>
        <li>YOLOv8CSPDarknet</li>
        <li>YOLOXCSPDarknet</li>
        <li>EfficientRep</li>
        <li>CSPNeXt</li>
        <li>YOLOv7Backbone</li>
        <li>PPYOLOECSPResNet</li>
        <li>mmdet backbone</li>
        <li>mmcls backbone</li>
        <li>timm</li>
      </ul>
      </td>
      <td>
      <ul>
        <li>YOLOv5PAFPN</li>
        <li>YOLOv8PAFPN</li>
        <li>YOLOv6RepPAFPN</li>
        <li>YOLOXPAFPN</li>
        <li>CSPNeXtPAFPN</li>
        <li>YOLOv7PAFPN</li>
        <li>PPYOLOECSPPAFPN</li>
      </ul>
      </td>
      <td>
        <ul>
          <li>IoULoss</li>
          <li>mmdet loss</li>
        </ul>
      </td>
      <td>
        <ul>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

</details>

