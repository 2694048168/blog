# Visual Interpretation of Deep Convolution Neural Networks

- &ensp;<span style="color:MediumPurple">Title</span>: Visual Interpretation of Deep Convolution Neural Networks
- &ensp;<span style="color:Moccasin">Tags</span>: Attention Mechanism; Visual Interpretation; Deep Convolution Neural Networks;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Mini-Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-09


## Overview of Visual Interpretation
- ZF-Net
- CAM
- Grad-CAM
- 可视化特征图
- 可视化卷积核


### Source Reference Link
- [Advanced AI explainability for PyTorch](https://github.com/jacobgil/pytorch-grad-cam)
- [YOLOv3 code](https://github.com/ultralytics/yolov3)
- [YOLOv4 code](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- [YOLOv5 code](https://github.com/ultralytics/yolov5)
- [YOLOv7 code](https://github.com/wongkinyiu/yolov7)
- [YOLOv7 + Grad-CAM](https://blog.csdn.net/weixin_43799388/article/details/126190981)
- [YOLOv5 + Grad-CAM](https://github.com/pooya-mohammadi/yolov5-gradcam)
- [TV-norm](https://www.zhihu.com/question/24049207)




### Reference
----------------------------

[1] Matthew D Zeiler, Rob Fergus, "Visualizing and Understanding Convolutional Networks," ECCV'2014

[ZF-Net Paper on ECCV'2014](https://proceedings.mlr.press/v37/ioffe15.html)
&emsp;&emsp;[Paper on arXiv'2014](https://arxiv.org/abs/1311.2901v3)
&emsp;&emsp;[code implementation](https://paperswithcode.com/paper/visualizing-and-understanding-convolutional)

[2] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba, "Learning Deep Features for Discriminative Localization," CVPR'2016

[CAM Paper on CVPR'2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html)
&emsp;&emsp;[Paper on arXiv'2016](https://arxiv.org/abs/1512.04150v1)
&emsp;&emsp;[code implementation](https://paperswithcode.com/paper/learning-deep-features-for-discriminative)

[3] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV'2017 and IJCV'2019

[Grad-CAM Paper on ICCV'2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)
[Paper on IJCV'2020](https://link.springer.com/article/10.1007/s11263-019-01228-7)
&emsp;&emsp;[Paper on arXiv'2017](https://arxiv.org/abs/1610.02391)
&emsp;&emsp;[Original code](https://github.com/ramprs/grad-cam)
&emsp;&emsp;[code implementation](https://paperswithcode.com/paper/grad-cam-visual-explanations-from-deep)

[4] Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian, "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks," WACV'2018

[Grad-CAM++ Paper on WACV'2018](https://ieeexplore.ieee.org/abstract/document/8354201)
&emsp;&emsp;[Paper on arXiv'2017](https://arxiv.org/abs/1710.11063)
&emsp;&emsp;[Original code](https://github.com/adityac94/Grad_CAM_plus_plus)
&emsp;&emsp;[code implementation](https://paperswithcode.com/paper/grad-cam-improved-visual-explanations-for)

[5] Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, Xia Hu, "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks," CVPR'2020

[Score-CAM Paper on CVPR'2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.html)
&emsp;&emsp;[Paper on arXiv'2020](https://arxiv.org/abs/1910.01279)
&emsp;&emsp;[Original code](https://github.com/haofanwang/Score-CAM)
&emsp;&emsp;[code implementation](https://paperswithcode.com/paper/score-camimproved-visual-explanations-via)

[6] Peng-Tao Jiang, Chang-bin Zhang, Qibin Hou, Ming-Ming Cheng, Yunchao Wei, "LayerCAM: Exploring Hierarchical Class Activation Maps for Localization," TIP'2021

[Layer-CAM Paper on TIP'2021](https://ieeexplore.ieee.org/abstract/document/9462463)
&emsp;&emsp;[Paper on person page](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf)
&emsp;&emsp;[Paper Project](https://mmcheng.net/layercam/)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/1910.01279)
&emsp;&emsp;[Original code](https://github.com/PengtaoJiang/LayerCAM-jittor)

[7] Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement," Tech. Report'2018

[YOLO3 Paper on arXiv'2018](https://arxiv.org/abs/1804.02767)
&emsp;&emsp;[code implementation](https://paperswithcode.com/paper/yolov3-an-incremental-improvement)

[8] Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection," arXiv'2020

[YOLO4 Paper on arXiv'2020](https://arxiv.org/abs/2004.10934v1)
&emsp;&emsp;[Original code](https://github.com/AlexeyAB/darknet)
&emsp;&emsp;[YOLO5 source code](https://github.com/ultralytics/yolov5)

[9] Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao, "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," arXiv'2022

[YOLO7 Paper on arXiv'2022](https://arxiv.org/abs/2207.02696)
&emsp;&emsp;[Original code](https://github.com/wongkinyiu/yolov7)
&emsp;&emsp;[code implementation](https://paperswithcode.com/paper/yolov7-trainable-bag-of-freebies-sets-new)
