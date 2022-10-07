# Object Detection

Object Localization is simple case of Object Detection.

> Object <span style="color:Green">Localization</span> is finding <span style="color:DarkOrange">what</span> and <span style="color:DarkOrange">where</span> a (single) object exists in an image;

> Object <span style="color:Green">Detection</span> is finding <span style="color:DarkOrange">what</span> and <span style="color:DarkOrange">where</span> a (multiple) object exists in an image;


## Overview
1. how do we do Object Localization?
    - fisrtly image classification(cat or dog)
    - localization to add four additional points that correspond to the bounding box for that particular object(bounding box via [x1,y1] and [x2,y2] coordinates)
    - two common ways to defind BBOXES: 1) (x1, y1) is the uppper left corner point, (x2, y2) is bottom right corner point; 2) tow points define a corner point, and two points to define height and width
2. localization is no biggie! how do we **generalize** this for multiple objects in an image?
    - many approaches to solving object detection and each is in many ways unique
    - the predict bounding box via **sliding windows** with particular step-size and particular stride(sliding window is usually a square)
3. potential problems in this way?
    - a lot of computation?
    - "OverFeat" paper on the ImageNet Large Scale Visual Recognition Challenge 2013 (ILSVRC2013)
    - many bouding boxes for same object?
    - non-max suppression method to solve second problem
    - regional based networks to solve first problem
    - [regional-based methods:R-CNN,Fast R-CNN,Faster R-CNN,Mask R-CNN](https://d2l.ai/chapter_computer-vision/rcnn.html)
4. regional-based still slow(real-time) and complicated two-stage process?
    - first-step to have some region proposal
    - second-step to determine for each of those regions is the bouding box
    - one-single step(stage) method end to end
    - YOLO(You Only Look Once) algorithm
    - SxS grid on input ---> Bounding boxes + confidence; Class probability map ---> Final detections
    - SSD model and YOLOs models paper
    - [d2l book by Mu Li](https://d2l.ai/chapter_computer-vision/)
    or [动手学深度学习——李沫](https://zh.d2l.ai/)
5. how to evaluate bounding box prediction and target bounding box for an object?
    - way of quantifying or measure for prediction and target
    - how do we measure how good a bounding box is?
    - Intersection over Union(IoU) $IoU = \frac{Area~~of~~Intersection}{Area~~Union} \in(0, 1)$
    - IoU > 0.5 ---> "decent"; IoU > 0.7 ---> "pretty good"; IoU > 0.9 ---> "almost perfect";
    - how do we get the intersection? max and min for prediction_box and target_box
6. how to clean bounding boxes for an object?
    - cleaning up Bounding Boxes via Non-Max  Suppression
    - prediction bounding boxes from model have a probability score
    - what if multiple classes?
7. Mean Average Precision(mAP)
    - understand and implement the most common metric used in Deep Learning to evaluate object detection models
    - "mAP@0.5:0.05:0.95" in research paper should know <span style="color:DarkTurquoise">exactly</span> what that means, and know how to do that evaluation on our own model
    - confusion matrix(True Positives,TP; False Positives,FP; False Negatives, FN; True Negatives,TN)
    - Precision and Recall(PR curve or graph) and Area under(AP) for single-object
    - All this was calculated given a <span style="color:DarkOrange">specific IoU</span> threshold of 0.5, we need to redo all computations for many IoUs, example(step=0.05): 0.5, 0.55, 0.6, ... , 0.95. Then <span style="color:DarkGreen">average this</span> and this will be our final result. This is what is meant by <span style="color:DodgerBlue">mAP@0.5:0.05:0.95</span>
    - implement this from scratch in <span style="color:DarkOrange">PyTorch</span> to make sure we get <span style="color:DodgerBlue">all the details</span>
8. YOLOv1 algorithm
9. YOLOv3 algorithm


## Source Reference Link
- [Deep Learing to Computer Vision by Mu Li](https://d2l.ai/chapter_computer-vision/)
- [YOLO serial algorithm](https://appsilon.com/object-detection-yolo-algorithm/)
- [YOLO v1 and v3 simple-scratch PyTorch code](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection)


## Overview of YOLO algorithm

<center class="center">
    <img src="./images/YOLOv1.png" />
</center>

### idea behind algorithm
- split an image into SxS(e.g. 7) grids(patchs/tokens/cells)
- each cell will output a prediction with a corresponding bounding box
- one cell can contains the object's midpoint(bounding box)
- how will labels look like? each output and label will be relative to the cell
- each bounding box for each cell will have $[x, y, w, h] \in (0, 1)$

### Model architecture
- model architecture in Figure 3 in original paper

### Loss Function
- MSE loss for bounding boxes coord error
- MSE loss for bounding boxes width and height error
- MSE loss for bounding boxes object and class error

### PASCAL VOC Dataset with 20 classes
- [PASCAL Visual Object Classes Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Common Objects in Context Dataset](https://cocodataset.org/#home)
- [useful Dataset on Kaggle](https://www.kaggle.com/code/mamun18/yolov1-implementation-with-pytorch)

## Overview of YOLOv3 algorithm

<center class="center">
    <img src="./images/YOLOv3.png" />
</center>

- Quick recap of YOLO algorithm
    - [YOLOv3:Implementation with Training setup from Scratch](https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0)
- Differences between YOLO and YOLOv3
    - network architecture(multi-scale bounding boxes)
    - Anchor boxes conception from YOLOv2
    - implement Anchor Boxes to understand all the details
- Implement it from scratch
    - [Pascal VOC dataset for YOLOv3 on Kaggle](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video)
    - [Microsoft COCO dataset for YOLOv3 on Kaggle](https://www.kaggle.com/datasets/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e)
    - [pretrained weights on Pascal-VOC on Kaggle](https://www.kaggle.com/datasets/1cf520aba05e023f2f80099ef497a8f3668516c39e6f673531e3e47407c46694)



------------------------
## Reference
[1] Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun, "OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks," arXiv'2013

[OverFeat ILSVRC'2013 on arXiv](https://arxiv.org/abs/1312.6229)
&emsp;&emsp;[OverFeat ILSVRC'2013](https://arxiv.org/abs/1312.6229)
&emsp;&emsp;[OverFeat code reference](https://paperswithcode.com/paper/overfeat-integrated-recognition-localization)

[2] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation," CVPR'2014

[R-CNN paper on arXiv](https://arxiv.org/abs/1311.2524)
&emsp;&emsp;[R-CNN paper on IEEE CVPR](https://ieeexplore.ieee.org/document/6909475)
&emsp;&emsp;[R-CNN paper on CVPR](https://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)
&emsp;&emsp;[R-CNN code reference](https://paperswithcode.com/paper/rich-feature-hierarchies-for-accurate-object)

[3] Ross Girshick, "Fast R-CNN," ICCV'2015

[Fast R-CNN paper on arXiv](https://arxiv.org/abs/1504.08083)
&emsp;&emsp;[Fast R-CNN paper on IEEE ICCV](https://ieeexplore.ieee.org/document/7410526)
&emsp;&emsp;[Fast R-CNN paper on ICCV](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html)
&emsp;&emsp;[Fast R-CNN code](https://github.com/rbgirshick/fast-rcnn)
&emsp;&emsp;[Fast R-CNN code reference](https://paperswithcode.com/paper/fast-r-cnn)

[4] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS'2015

[Faster R-CNN paper on arXiv](https://arxiv.org/abs/1506.01497)
&emsp;&emsp;[Faster R-CNN paper on NeurIPS](https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html)
&emsp;&emsp;[Faster R-CNN code](https://github.com/ShaoqingRen/faster_rcnn)

[5] Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, "Mask R-CNN," ICCV'2017

[Mask R-CNN paper on arXiv](https://arxiv.org/abs/1703.06870)
&emsp;&emsp;[Mask R-CNN paper on ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
&emsp;&emsp;[Mask R-CNN code on GitHub](https://github.com/matterport/Mask_RCNN)
&emsp;&emsp;[Mask R-CNN code reference](https://paperswithcode.com/paper/mask-r-cnn)

[6] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," CVPR'2016

[YOLOv1 paper on arXiv](https://arxiv.org/abs/1506.02640)
&emsp;&emsp;[YOLOv1 paper on CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)
&emsp;&emsp;[YOLOv1 code reference](https://paperswithcode.com/paper/you-only-look-once-unified-real-time-object)

[7] Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement," arXiv'2018

[YOLOv3 paper on arXiv](https://arxiv.org/abs/1804.02767)
&emsp;&emsp;[YOLOv3 code reference](https://paperswithcode.com/paper/yolov3-an-incremental-improvement)
