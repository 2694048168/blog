# Graph Neural Networks and Non-Euclidean Convolution

- &ensp;<span style="color:MediumPurple">Title</span>: Graph Neural Networks and Non-Euclidean Convolution
- &ensp;<span style="color:Moccasin">Tags</span>: Euclidean Space; Non-Euclidean; Convolution; Spectrum Convolution; Graph Neural Networks; Graph Convolution;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Mini-Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-09

## Overview
- Euclidean and non-Euclidean Domains
- Convolution Operations
- Data Information = Features + Structures(relationships)
- Advantages of Graph Representation of images from ViG paper
- [图神经网络的应用场景](https://mp.weixin.qq.com/s/C10Tl2SGgU6zN-7GKfrrzw)


**Reference Source Linking**
- [GNN conference paper](https://github.com/naganandy/graph-based-deep-learning-literature)
- [GNN paper list](https://github.com/thunlp/GNNPapers/blob/master/README.md)
- [Graph Data Augmentation Papers](https://github.com/zhao-tong/graph-data-augmentation-papers)
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)
- [GNN on wikipedia](https://en.wikipedia.org/wiki/Graph_neural_network)
- [Graph Convolutional Networks GCN](https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b)
- [GNN tutorial slides on KDD'2022 and IJCAI'2022](https://graph-neural-networks.github.io/tutorial_ijcai22.html)
- [PyG:Graph Neural Network Library for PyTorch](https://pytorch-geometric.readthedocs.io/en/latest/)
- [DGL:Deep Grahp Library](https://www.dgl.ai/)


### Euclidean and non-Euclidean Domains

> [From Euclidean to non-Euclidean Domains](https://www.bilibili.com/video/BV1W14y147ps?p=37&vd_source=8a8b4e22f7e4a8ca15c0705739d772cb); the development of Deep Neural Networks and Deep Learning.

**Convolutional Neural Networks**
- Data embedding on regular grids (Euclidean or grid-like structure)
    - Speech: translation, speech processing; 1-dimension tensor or vector
    - Image: object detection, object segmentation; 2-dimension tensor or matrix
    - Video: video understanding, object detection; 3-dimension tensor
- Convolution: sharing parameters at each location
    - (多维)欧氏空间
    - 局部空间响应
    - 卷积参数共享

**Graph Convolutional Neural Networks**
- Data embedding on irregular grids
    - Social networks: Facebook, Twitter, Tencent
    - Biological networks: genes, molecules, brain connectivity
    - Infrastructure networks: transportation, Internet, radar
- Graph Convolution: sharing parameters at each vertex(node)

### Convolution Operations

**Convolution**
- Euclidean or grid-like structures
- non-Euclidean or graph structures
- Convolution Neural Networks
    - Sharing parameters at each location
    - Tensor versus Graph
    - $\mathcal{y}_{n} = \mathcal{F}(x_{n})$ whatever for Euclidean or non-Euclidean
- Convolution Operations on Euclidean
    - Basic convolution
        - Important assumptions on the statistical properties
        - Euclidean or grid-like structure: speeches, images(kernel-fixed), videos
    - Active convolution CVPR'2017
        - Bilinear interpolation
    - Deformable convolution ICCV'2017
        - $\mathcal{y}(p_{0}) = \sum_{p_{n} \in \mathcal{R}} w(p_{n}) \cdot x(p_{0} + p_{n})$
        - $\mathcal{y}(p_{0}) = \sum_{p_{n} \in \mathcal{R}} w(p_{n}) \cdot x(p_{0} + p_{n} + \bigtriangleup p_{n})$; offsets
        - $\mathcal{R} = \{ (-1, -1), (1, 0), \dots, (0, 1),(1, 1) \}$
        - 3*3 deformable convolution ---> N = 9
    - Limitations: hardly be applied to non-Euclidean domains
        - The number of neighbors is **unfixed** and the neighbors are **unordered**
        - It is difficult to execute the bilinear interpolation
        - Ordered permutation and Equal dimension in Euclidean 
        - versus Unordered permutation and Unequal dimension in non-Euclidean
- Convolution Operations on non-Euclidean
    - Graph convolution: Spectral Convolution and Spatial Convolution
    - From the spectral graph theory: **Graph Laplacian** and Fourier transform
        - Graph Laplacian: 包含图上所有节点的结构信息
        - Matrix decomposition ---> time-consuming
        - $L = U \wedge U^{T}$; $x \ast y = U ( (U^{T} \cdot x) \odot (U^{T} \cdot y))$
        - Relationship is hard to obtained (edge-weight coeff.)
        - Processing in frequency domain
    - Spectral methods NeurIPS'2016(time-consuming)
        - L-decomposition into Affine Function with $\theta$
        - $y = g_{\theta}(L)x = \sum_{k=0}^{k-1} \theta_{k} T_{k}(\tilde{L})x$
        - GCN paper ICLR'2017 ---> $K=2$; 基函数只需要两个; 局部节点只考虑二阶度
        - Limitation: these methods will be degenerated on images, videos; 3*3 parameters ---> 3 parameters
    - Analogizing the classical convolutional **Aggregation**
        - Aggregating local inputs with a sharable local manifold (spatial)
        - 从离散的数据点到流线曲面, 对应的流线曲面做点乘后积分即可
        - $D_{j}(x) f = \int_{\mathcal{x}} f(\acute{x}) v_{j}(x, \acute{x}) \mathrm{d} \acute{x}, j = 1, \dots, J, $
        - $(f \star g)(x) = \sum_{j} g_{j} D_{j}(x) f$
        - How to express **manifold** for data; 不同的基函数选择
        - Diffusion CNNs NeurIPS'2016; Anisotropic CNNS NeurIPS'2016; Geodesic CNNs ICCV'2015; Mixture-model CNNs CVPR'2017
        - Limitations: images and videos results; fixed or undirected graph structure

### Data Information = Features + Structures(relationships)

**Structure-Aware Convolution**
- Rethinking convolution
    - Stationarity is a nice or necessary property? (局部平稳性)
    - Just a **local parameter sharing operation**
    - Examplars: active conv. /deformable conv. /structure-aware conv.
- Understanding data
    - Euclidean space(计算机存储) and non-Euclidean space(现实场景)
    - 真实场景的数据如何更好的表征; 计算机内部如何更好的存储; 折中平衡
    - Data = feature + structures(relationships)
- Classical convolution to Structure-Aware convolution
    - Structure-Aware Convolution paper NeurIPS'2018
    - Discrete-Continuous and Finite-Infinite
    - Structure-aware convolution in **Paper**
    - Structure-aware convolutional neural networks in **Paper**
        - Polynomial parametrization for functional filters
        - Local structure representations learning
        - Understanding the structure-aware convolution
    - Functional filters may be used in theoretical tasks
        - Model compression and Few-shot learning
        - Continual learning with a specific basis function
        - Explaining the learning processing of deep learning (泛函分析网络学到什么)
    - Functional filters may be used in practical tasks
        - Classifying the radar data from automatic driving
        - Solving the problem of real-time path planning
        - Discovering potential user (Amazon, Taobao, JD)

<center class="center">
    <img src="./images/gnn_1.png" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 1.  A structure-aware convolutional layer. For clarity of exposition, the input $x$ has $c=2$ channels with $n = 6$ vertices, the output $y$ has a single channel, and $\bar{x}_{j}$, $\bar{x}_{j} \in \mathcal{R}^{c}$ indicate the j-th and i-th rows of the input $x$, respectively. For each vertex $i$, its local structure  representation is first captured from the input and represented as $\mathcal{R}_{i}$, which is identically shared for  each channel of the input $x$. Afterwards, the local inputs in the first and second channels are aggregated via the  first filter $f1(·)$ and the second filter $f2(·)$ respectively, with the same $\mathcal{R}_{i}$. Note that $f1(·)$ and $f2(·)$ are shared for every location in the first and second channels, respectively. (Image source from Structure-Aware Convolution paper NeurIPS'2018)</div>
</center>

<center class="center">
    <img src="./images/gnn_2.png" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 2. Illustration of the grid, sequence and graph representation of the image. In the grid structure, the pixels or patches are ordered only by the spatial position. In the sequence structure, the 2D image is transformed  to a sequence of patches. In the graph structure, the nodes are linked by its content and are not constrained by the  local position. (Image source from ViG paper'2022)</div>
</center>


### Advantages of Graph Representation of images

**Advantages of graph representation of the image**
1. graph is a generalized data structure that grid and sequence can be viewed as a special case of graph
2. graph is more flexible than grid or sequence to model the complex object as an object in the image is usually not quadrate whose shape is irregular
3. an object can be viewed as a composition of parts (e.g., a human can be roughly divided into head, upper body, arms and legs), and graph structure can construct the connections among those parts
4. the advanced research on GNN can be transferred to address visual tasks
5. Additional statistical, causal or logical relationships among the so-abstracted discrete concepts can be further modeled parsimoniously as a compact and structured (say sparse) graph, with each node representing a subspace/category, e.g.,  (Bear et al., 2020). We believe such a graph can be  and should be learned via a closedloop transcription to ensure self- consistency. **Quote from Prof. Yi Ma paper**


#### **Definition of Sign Convolution**

$$ \int_{- \infty}^{\infty} f(\tau) g(x - \tau) \mathrm{d}\tau $$

$$ y_{n} = x * w = \sum_{k=1}^{K} w_{k} x_{n - k} $$

$$ f_{1}(t) \star f_{2}(t) = \mathcal{F}^{-1} [ F_{1}(\omega) \cdot F_{2}(\omega) ]$$




### Reference
----------------------------

[1] Kai Han, Yunhe Wang, Jianyuan Guo, Yehui Tang, Enhua Wu, "Vision GNN: An Image is Worth Graph of Nodes," arXiv'2022,Tech. Report Huawei Noah

[ViG Paper on arXiv'2022](https://arxiv.org/abs/2206.00272)
&emsp;&emsp;[ViG code](https://github.com/huawei-noah/CV-backbones)
&emsp;&emsp;[ViG code](https://github.com/huawei-noah/efficient-ai-backbones)

[2] Yi Ma, Doris Tsao, Heung-Yeung Shum, "On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence," arXiv'2022

[Principles Paper on arXiv'2022](https://arxiv.org/abs/2207.04630)

[3] Ladislav Rampášek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini, "Recipe for a General, Powerful, Scalable Graph Transformer," arXiv'2022

[GraphGPS Paper on arXiv'2022](https://arxiv.org/abs/2206.00272)
&emsp;&emsp;[GraphGPS code](https://github.com/rampasek/GraphGPS)

[4] Thomas N. Kipf, Max Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR'2017

[GCN Paper on ICLR'2017](https://openreview.net/forum?id=SJU4ayYgl)
&emsp;&emsp;[GCN Paper on arXiv'2017](https://arxiv.org/abs/1609.02907)
&emsp;&emsp;[GCN Paper blog](http://tkipf.github.io/graph-convolutional-networks/)
&emsp;&emsp;[GCN implementation code](https://paperswithcode.com/paper/semi-supervised-classification-with-graph)

[5] Qimai Li, Zhichao Han, Xiao-Ming Wu, "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning," AAAI'2018

[Deeper GCN on AAAI'2018 Oral](https://ojs.aaai.org/index.php/AAAI/article/view/11604)
&emsp;&emsp;[Deeper GCN on arXiv'2018](https://arxiv.org/abs/1801.07606)
&emsp;&emsp;[Deeper GCN implementation code](https://paperswithcode.com/paper/deeper-insights-into-graph-convolutional)

[6] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio, "Graph Attention Networks," ICLR'2018

[GAT Paper on ICLR'2018](https://openreview.net/forum?id=rJXMpikCZ)
&emsp;&emsp;[GAT Paper on arXiv'2018](https://arxiv.org/abs/1710.10903)
&emsp;&emsp;[GAT code](https://github.com/PetarV-/GAT)
&emsp;&emsp;[GAT implementation code](https://paperswithcode.com/paper/graph-attention-networks)

[7] Shangchen Zhou, Jiawei Zhang, Wangmeng Zuo, Chen Change Loy, "Cross-Scale Internal Graph Neural Network for Image Super-Resolution," NeurIPS'2020

[IGNN Paper on NeurIPS'2020](https://proceedings.neurips.cc/paper/2020/hash/23ad3e314e2a2b43b4c720507cec0723-Abstract.html)
&emsp;&emsp;[IGNN on arXiv'2020](https://arxiv.org/abs/2006.16673)
&emsp;&emsp;[IGNN code](https://github.com/sczhou/IGNN)

[8] Riccardo de Lutio, Alexander Becker, Stefano D'Aronco, Stefania Russo, Jan D. Wegner, Konrad Schindler, "Learning Graph Regularisation for Guided Super-Resolution," CVPR'2022

[Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/de_Lutio_Learning_Graph_Regularisation_for_Guided_Super-Resolution_CVPR_2022_paper.html)
&emsp;&emsp;[paper on arXiv'2022](https://arxiv.org/abs/2203.14297)
&emsp;&emsp;[paper code](https://github.com/prs-eth/graph-super-resolution)

[9] Zipeng Liu, Yang Wang, Jürgen Bernard, Tamara Munzner, "Visualizing Graph Neural Networks with CorGIE: Corresponding a Graph to Its Embedding," TVCG'2022

[Paper on TVCG'2022](https://www.computer.org/csdl/journal/tg/2022/06/09705082/1AIIcR14E4o)
&emsp;&emsp;[paper on arXiv'2022](https://arxiv.org/abs/2106.12839)
&emsp;&emsp;[paper code](https://github.com/zipengliu/CorGIE)

[10] Mingqi Yang, Yanming Shen, Rui Li, Heng Qi, Qiang Zhang, Baocai Yin, "A New Perspective on the Effects of Spectrum in Graph Neural Networks," ICML'2022

[Paper on ICML'2022](https://proceedings.mlr.press/v162/yang22n.html)
&emsp;&emsp;[paper on arXiv'2022](https://arxiv.org/abs/2112.07160)
&emsp;&emsp;[paper code](https://github.com/qslim/gnn-spectrum)

[11] Dan Xu, Xavier Alameda-Pineda, Wanli Ouyang, Elisa Ricci, Xiaogang Wang, Nicu Sebe, "Probabilistic Graph Attention Network with Conditional Kernels for Pixel-Wise Prediction," accepted at TPAMI'2020 and published at TPAMI'2022

[Paper on TPAMI'2020](https://ieeexplore.ieee.org/document/9290049)
&emsp;&emsp;[paper on arXiv'2022](https://arxiv.org/abs/2101.02843)

[12] Pál András Papp, Roger Wattenhofer, "A Theoretical Comparison of Graph Neural Network Extensions," ICML'2022

[Paper on ICML'2022](https://proceedings.mlr.press/v162/papp22a.html)
&emsp;&emsp;[paper on arXiv'2022](https://arxiv.org/abs/2201.12884)

[13] Asiri Wijesinghe, Qing Wang, "A New Perspective on 'How Graph Neural Networks Go Beyond Weisfeiler-Lehman?'," ICLR'2022

[Paper on ICLR'2022 Oral](https://iclr.cc/virtual/2022/oral/6437)
&emsp;&emsp;[Paper Poster at ICLR'2022](https://graphlabanu.github.io/website/downloads/ICLR2022_Poster.pdf)

[14] Yao Li, Xueyang Fu, Zheng-Jun Zha, "Cross-Patch Graph Convolutional Network for Image Denoising," ICCV'2021

[Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Cross-Patch_Graph_Convolutional_Network_for_Image_Denoising_ICCV_2021_paper.html)

[15] Jianlong Chang, Jie Gu, Lingfeng Wang, GAOFENG MENG, SHIMING XIANG, Chunhong Pan, "Structure-Aware Convolutional Neural Networks," NeurIPS'2018

[Structure-Aware Paper on NeurIPS'2018](https://papers.nips.cc/paper/2018/hash/182be0c5cdcd5072bb1864cdee4d3d6e-Abstract.html)
&emsp;&emsp;[Original code](https://github.com/vector-1127/SACNNs)

[16] Weiyi Xie, Colin Jacobs, Jean-Paul Charbonnier, Bram van Ginneken, "Structure and position-aware graph neural network for airway labeling," arXiv'2022

[Paper on arXiv'2022](https://arxiv.org/abs/2201.04532)
&emsp;&emsp;[Original code](https://github.com/diagnijmegen/spgnn)
