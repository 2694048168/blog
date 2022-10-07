# Neural Radiance Fields (NeRF) technique

- &ensp;<span style="color:MediumPurple">Title</span>: Neural Radiance Fields (NeRF) technique
- &ensp;<span style="color:Moccasin">Tags</span>: NeRF; 神经辐射场; 3D重建; View Synthesis; 视角合成;
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




### Reference
----------------------------

[1] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, "NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images," CVPR'2022

[Oral Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Mildenhall_NeRF_in_the_Dark_High_Dynamic_Range_View_Synthesis_From_CVPR_2022_paper.html)
&emsp;&emsp;[Paper on arXiv'2022](https://arxiv.org/abs/2111.13679)
&emsp;&emsp;[Paper Project](https://bmild.github.io/rawnerf/)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/google-research/multinerf)

[2] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng, "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV'2020, Oral and Best Paper Honorable Mention

[Oral and Best Paper on ECCV'2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1473_ECCV_2020_paper.php)
&emsp;&emsp;[Paper on arXiv'2020](https://arxiv.org/abs/2003.08934)
&emsp;&emsp;[Paper Project](https://www.matthewtancik.com/nerf)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/bmild/nerf)
&emsp;&emsp;[Implementation Code](https://paperswithcode.com/paper/nerf-representing-scenes-as-neural-radiance)

[3] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, Daniel Duckworth, "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collection," CVPR'2021

[Oral Paper on CVPR'2021](https://openaccess.thecvf.com/content/CVPR2021/html/Martin-Brualla_NeRF_in_the_Wild_Neural_Radiance_Fields_for_Unconstrained_Photo_CVPR_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2020](https://arxiv.org/abs/2008.02268)
&emsp;&emsp;[Paper Project](https://nerf-w.github.io/)
&emsp;&emsp;[Implementation Code](https://paperswithcode.com/paper/nerf-in-the-wild-neural-radiance-fields-for)

[4] Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen Lu, Jie Zhou, "NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor Multi-view Stereo," ICCV'2021

[Oral Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wei_NerfingMVS_Guided_Optimization_of_Neural_Radiance_Fields_for_Indoor_Multi-View_ICCV_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2109.01129)
&emsp;&emsp;[Paper Project](https://weiyithu.github.io/NerfingMVS/)
&emsp;&emsp;[Paper Code](https://github.com/weiyithu/nerfingmvs)

[5] Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Animashree Anandkumar, Minsu Cho, Jaesik Park, "Self-Calibrating Neural Radiance Fields," ICCV'2021

[Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Jeong_Self-Calibrating_Neural_Radiance_Fields_ICCV_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2108.13826)
&emsp;&emsp;[Paper Project](https://postech-cvlab.github.io/SCNeRF/)
&emsp;&emsp;[Paper Code](https://github.com/postech-cvlab/scnerf)

[6] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, Andrew J. Davison, "In-Place Scene Labelling and Understanding with Implicit Scene Representation," ICCV'2021

[Oral Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhi_In-Place_Scene_Labelling_and_Understanding_With_Implicit_Scene_Representation_ICCV_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2108.13826)
&emsp;&emsp;[Paper Project](https://shuaifengzhi.com/Semantic-NeRF/)
&emsp;&emsp;[Paper Code](https://github.com/Harry-Zhi/semantic_nerf/)

[7] Kai Zhang, Gernot Riegler, Noah Snavely, Vladlen Koltun, "NeRF++: Analyzing and Improving Neural Radiance Fields," arXiv'2020

[NeRF++ Paper on arXiv'2021](https://arxiv.org/abs/2010.07492)
&emsp;&emsp;[Paper Code](https://github.com/Kai-46/nerfplusplus)

[8] Norman Müller, Andrea Simonelli, Lorenzo Porzi, Samuel Rota Bulò, Matthias Nießner, Peter Kontschieder, "AutoRF: Learning 3D Object Radiance Fields from Single View Observations," CVPR'2022

[AutoRF Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Muller_AutoRF_Learning_3D_Object_Radiance_Fields_From_Single_View_Observations_CVPR_2022_paper.html)
&emsp;&emsp;[AutoRF Paper on arXiv'2022](https://arxiv.org/abs/2204.03593)
&emsp;&emsp;[Paper Project and Code](https://sirwyver.github.io/AutoRF/)
&emsp;&emsp;[Reference Code](https://github.com/google/nerfies)
