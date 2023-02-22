# Neural Radiance Fields (NeRF) technique

- &ensp;<span style="color:MediumPurple">Title</span>: Neural Radiance Fields (NeRF) technique
- &ensp;<span style="color:Moccasin">Tags</span>: NeRF; View Synthesis; 3D Reconstruction; Neural Rendering;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Mini-Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">Revision of DateTime</span>: 2022-09-05;


---------------------

<center class="center">
    <img src="./images/full_pipeline_dark_light.svg" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 1. The standard NeRF training pipeline (a) takes in LDR images that have been sent through a camera processing pipeline, reconstructing the scene and rendering new views in LDR color space. As such, its renderings are  effectively already postprocessed and cannot be significantly retouched. In contrast, our method RawNeRF (b) modifies  NeRF to train directly on linear raw HDR input data. The resulting scene representation produces novel views that can be edited like any raw photograph. (Image source from Oral paper, CVPR'2022)</div>
</center>


## **Overview and Source**
- Introduction to the Neural Radiance Fields, NeRF
- Foundation of the NeRF technique
- Application of the NeRF technique
- [Awesome NeRF paper list](https://github.com/yenchenlin/awesome-NeRF)
- [Neural Fields in Computer Vision CVPR'2022 Toturial](https://neuralfields.cs.brown.edu/cvpr22)
- [Neural Fields in Computer Vision CVPR'2022 Toturial bilibili vedio](https://www.bilibili.com/video/BV1he411u7rS/)
- [Neural Fields CVPR'2022 paper](http://blog.leanote.com/post/wuvin/CVPR2022-NeRF)
- [Awesome NeRF papers](https://github.com/awesome-NeRF/awesome-NeRF)


**基于 NeRF 的三维内容生成**
1. why 3D content creation from images
    - what are 3D contents
    - rendering and inverse rendering
    - 3 key factors in inverse rendering
2. why NeRF is a big thing
    - break NeRF into 3 components
    - take-home messages from NeRF
    - unbounded scene and anti-aliasing
3. editable 2D contents
    - relinghting, material re-touching
    - stylized apprearance
    - other editings




### Reference
----------------------------
[1] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, "NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images," CVPR'2022 </br>
[Oral Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Mildenhall_NeRF_in_the_Dark_High_Dynamic_Range_View_Synthesis_From_CVPR_2022_paper.html)
&emsp;&emsp;[Paper on arXiv'2022](https://arxiv.org/abs/2111.13679)
&emsp;&emsp;[Paper Project](https://bmild.github.io/rawnerf/)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/google-research/multinerf)

[2] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng, "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV'2020, Oral and Best Paper Honorable Mention </br>
[Oral and Best Paper on ECCV'2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1473_ECCV_2020_paper.php)
&emsp;&emsp;[Paper on arXiv'2020](https://arxiv.org/abs/2003.08934)
&emsp;&emsp;[Paper Project](https://www.matthewtancik.com/nerf)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/bmild/nerf)
&emsp;&emsp;[Implementation Code](https://paperswithcode.com/paper/nerf-representing-scenes-as-neural-radiance)

[3] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, Daniel Duckworth, "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collection," CVPR'2021 </br>
[Oral Paper on CVPR'2021](https://openaccess.thecvf.com/content/CVPR2021/html/Martin-Brualla_NeRF_in_the_Wild_Neural_Radiance_Fields_for_Unconstrained_Photo_CVPR_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2020](https://arxiv.org/abs/2008.02268)
&emsp;&emsp;[Paper Project](https://nerf-w.github.io/)
&emsp;&emsp;[Implementation Code](https://paperswithcode.com/paper/nerf-in-the-wild-neural-radiance-fields-for)

[4] Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen Lu, Jie Zhou, "NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor Multi-view Stereo," ICCV'2021 </br>
[Oral Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wei_NerfingMVS_Guided_Optimization_of_Neural_Radiance_Fields_for_Indoor_Multi-View_ICCV_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2109.01129)
&emsp;&emsp;[Paper Project](https://weiyithu.github.io/NerfingMVS/)
&emsp;&emsp;[Paper Code](https://github.com/weiyithu/nerfingmvs)

[5] Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Animashree Anandkumar, Minsu Cho, Jaesik Park, "Self-Calibrating Neural Radiance Fields," ICCV'2021 </br>
[Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Jeong_Self-Calibrating_Neural_Radiance_Fields_ICCV_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2108.13826)
&emsp;&emsp;[Paper Project](https://postech-cvlab.github.io/SCNeRF/)
&emsp;&emsp;[Paper Code](https://github.com/postech-cvlab/scnerf)

[6] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, Andrew J. Davison, "In-Place Scene Labelling and Understanding with Implicit Scene Representation," ICCV'2021 </br>
[Oral Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhi_In-Place_Scene_Labelling_and_Understanding_With_Implicit_Scene_Representation_ICCV_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2108.13826)
&emsp;&emsp;[Paper Project](https://shuaifengzhi.com/Semantic-NeRF/)
&emsp;&emsp;[Paper Code](https://github.com/Harry-Zhi/semantic_nerf/)

[7] Kai Zhang, Gernot Riegler, Noah Snavely, Vladlen Koltun, "NeRF++: Analyzing and Improving Neural Radiance Fields," arXiv'2020 </br>
[NeRF++ Paper on arXiv'2021](https://arxiv.org/abs/2010.07492)
&emsp;&emsp;[Paper Code](https://github.com/Kai-46/nerfplusplus)

[8] Norman Müller, Andrea Simonelli, Lorenzo Porzi, Samuel Rota Bulò, Matthias Nießner, Peter Kontschieder, "AutoRF: Learning 3D Object Radiance Fields from Single View Observations," CVPR'2022 </br>
[AutoRF Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Muller_AutoRF_Learning_3D_Object_Radiance_Fields_From_Single_View_Observations_CVPR_2022_paper.html)
&emsp;&emsp;[AutoRF Paper on arXiv'2022](https://arxiv.org/abs/2204.03593)
&emsp;&emsp;[Paper Project and Code](https://sirwyver.github.io/AutoRF/)
&emsp;&emsp;[Reference Code](https://github.com/google/nerfies)

[9] Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, Song-Hai Zhang, "NeRFReN: Neural Radiance Fields With Reflections," CVPR'2022 </br>
[NeRFReN Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_NeRFReN_Neural_Radiance_Fields_With_Reflections_CVPR_2022_paper.html)
&emsp;&emsp;[NeRFReN Paper on arXiv'2022](https://arxiv.org/abs/2111.15234)
&emsp;&emsp;[Paper Project](https://bennyguo.github.io/nerfren/)
&emsp;&emsp;[Code](https://github.com/bennyguo/nerfren)

[10] Chung-Yi Weng, Brian Curless, Pratul P. Srinivasan, Jonathan T. Barron, Ira Kemelmacher-Shlizerman, "HumanNeRF:Free-viewpoint Rendering of Moving People from Monocular Video," CVPR'2022 Oral </br>
[HumanNeRF Paper on CVPR'2022 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Weng_HumanNeRF_Free-Viewpoint_Rendering_of_Moving_People_From_Monocular_Video_CVPR_2022_paper.html)
&emsp;&emsp;[HumanNeRF Paper on arXiv'2022](https://arxiv.org/abs/2201.04127)
&emsp;&emsp;[Paper Project](https://grail.cs.washington.edu/projects/humannerf/)
&emsp;&emsp;[Code](https://github.com/chungyiweng/humannerf)

[11] Yudong Guo, Keyu Chen, Sen Liang, Yong-Jin Liu, Hujun Bao, Juyong Zhang, "AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis," ICCV'2021 </br>
[AD-NeRF Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_AD-NeRF_Audio_Driven_Neural_Radiance_Fields_for_Talking_Head_Synthesis_ICCV_2021_paper.html)
&emsp;&emsp;[AD-NeRF Paper on arXiv'2021](https://arxiv.org/abs/2103.11078)
&emsp;&emsp;[Paper Project](https://yudongguo.github.io/ADNeRF/)
&emsp;&emsp;[Code](https://github.com/YudongGuo/AD-NeRF)

[12] Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh, "On the Versatile Uses of Partial Distance Correlation in Deep Learning," ECCV'2022 </br>
[Paper on ECCV'2022 Best Paper Award](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3213_ECCV_2022_paper.php)
&emsp;&emsp;[Paper on arXiv'2022](https://arxiv.org/abs/2207.09684)
&emsp;&emsp;[Paper Project](https://zhenxingjian.github.io/DC_webpage/)
&emsp;&emsp;[Code](https://github.com/zhenxingjian/partial_distance_correlation)

[13] Yunzhi Lin, Thomas Müller, and Jonathan Tremblay et al. "Parallel Inversion of Neural Radiance Fields for Robust Pose Estimation," NVIDIA Corporation'2022 </br>
[Paper on arXiv'2022](https://arxiv.org/abs/2210.10108)
&emsp;&emsp;[Paper Project](https://pnerfp.github.io/)

[14] Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall, "DreamFusion: Text-to-3D using 2D Diffusion," ICLR'2023 </br>
[DreamFusion paper ICLR'2023](https://openreview.net/forum?id=FjNys5c7VyY)
&emsp;&emsp;[arXiv](https://arxiv.org/abs/2209.14988)
&emsp;&emsp;[Project page](https://dreamfusion3d.github.io/)
&emsp;&emsp;[Reference Code](https://github.com/ashawkey/stable-dreamfusion)

[15] Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, Mohammad Norouzi, "Novel View Synthesis with Diffusion Models," Google Research'2022 </br>
[paper arXiv'2022](https://arxiv.org/abs/2210.04628)
&emsp;&emsp;[Project page](https://3d-diffusion.github.io/)

[16] Tao Hu, Shu Liu, Yilun Chen, Tiancheng Shen, Jiaya Jia, "EfficientNeRF: Efficient Neural Radiance Fields," CVPR'2022 </br>
[EfficientNeRF paper CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_EfficientNeRF__Efficient_Neural_Radiance_Fields_CVPR_2022_paper.html)
&emsp;&emsp;[EfficientNeRF paper arXiv'2022](https://arxiv.org/abs/2206.00878)
&emsp;&emsp;[Code](https://github.com/dvlab-research/EfficientNeRF)

[17] Ali Tourani, Hriday Bavle, Jose Luis Sanchez-Lopez, Holger Voos, "Visual SLAM: What are the Current Trends and What to Expect?," arXiv'2022 </br>
[paper arXiv'2022](https://arxiv.org/abs/2210.10491)
&emsp;&emsp;[reference SLAM blog](https://mp.weixin.qq.com/s/CV68ZfFHCJJnKsngxfGpxg)

[18] Yongwei Chen, Rui Chen, Jiabao Lei, Yabin Zhang, Kui Jia, "TANGO: Text-driven Photorealistic and Robust 3D Stylization via Lighting Decomposition," NeurIPS'2022 </br>
[TANGO paper NeurIPS'2022 on openreview](https://openreview.net/forum?id=zbuq101sCNV)
&emsp;&emsp;[TANGO paper arXiv'2022](https://arxiv.org/abs/2210.11277)
&emsp;&emsp;[Project page](https://cyw-3d.github.io/tango/)
&emsp;&emsp;[code](https://github.com/Gorilla-Lab-SCUT/tango)

[19] Andreas Kurz, Thomas Neff, Zhaoyang Lv, Michael Zollhöfer, Markus Steinberger, "AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields," ECCV'2022 </br>
[AdaNeRF paper ECCV'2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6513_ECCV_2022_paper.php)
&emsp;&emsp;[paper arXiv'2022](https://arxiv.org/abs/2207.10312)
&emsp;&emsp;[Project page](https://thomasneff.github.io/adanerf/)
&emsp;&emsp;[code](https://github.com/thomasneff/AdaNeRF)

[20] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, Tsung-Yi Lin, "Magic3D: High-Resolution Text-to-3D Content Creation," NVIDIA'2022 </br>
[Magic3D paper arXiv'2022](https://arxiv.org/abs/2211.10440)
&emsp;&emsp;[Project](https://deepimagination.cc/Magic3D/)

[21] Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller, "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding," SIGGRAPH'2022 Best Paper </br>
[Instant-NGP SIGGRAPH'2022](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)
&emsp;&emsp;[arXiv](https://arxiv.org/abs/2201.05989)
&emsp;&emsp;[Project](https://nvlabs.github.io/instant-ngp/)
&emsp;&emsp;[Code](https://github.com/NVlabs/instant-ngp)

[22] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, Hao Su, "TensoRF: Tensorial Radiance Fields," ECCV'2022 </br>
[TensorRF ECCV'2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3555_ECCV_2022_paper.php)
&emsp;&emsp;[arXiv](https://arxiv.org/abs/2203.09517)
&emsp;&emsp;[Project](https://apchenstu.github.io/TensoRF/)
&emsp;&emsp;[Code](https://github.com/apchenstu/TensoRF)

[23] Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang, Yu-Wing Tai, Shi-Min Hu, "NeRF-SR: High-Quality Neural Radiance Fields using Supersampling," ACMMM'2022 </br>
[NeRF-SR ACMMM'2022](https://dl.acm.org/doi/abs/10.1145/3503161.3547808)
&emsp;&emsp;[arXiv](https://arxiv.org/abs/2112.01759)
&emsp;&emsp;[Project](https://cwchenwang.github.io/NeRF-SR/)
&emsp;&emsp;[Code](https://github.com/cwchenwang/NeRF-SR)

[24] Jian Zhang, Yuanqing Zhang, Huan Fu, Xiaowei Zhou, Bowen Cai, Jinchi Huang, Rongfei Jia, Binqiang Zhao, Xing Tang, "Ray Priors through Reprojection: Improving Neural Radiance Fields for Novel View Extrapolation," CVPR'2022 </br>
[RapNeRF CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Ray_Priors_Through_Reprojection_Improving_Neural_Radiance_Fields_for_Novel_CVPR_2022_paper.html)
&emsp;&emsp;[arXiv](https://arxiv.org/abs/2205.05922)

[25] Wojciech Zielonka, Timo Bolkart, Justus Thies, "Instant Volumetric Head Avatars," arXiv'2022 </br>
[arXiv](https://arxiv.org/abs/2211.12499)
&emsp;&emsp;[Project](https://zielon.github.io/insta/)

[26] Ayush Tewari, Justus Thies, Ben Mildenhall et. al "Advances in Neural Rendering," EuroGRAPHICS'2022 </br>
[arXiv](https://arxiv.org/abs/2111.05849)

[26] Ayush Tewari, Ohad Fried, Justus Thies et. al "State of the Art on Neural Rendering," EuroGRAPHICS'2020 </br>
[arXiv](https://arxiv.org/abs/2004.03805)
