# Principles of Parsimony and Self-Consistency for Intelligence

- &ensp;<span style="color:MediumPurple">Title</span>: Principles of Parsimony and Self-Consistency for Intelligence
- &ensp;<span style="color:Moccasin">Tags</span>: Parsimony; Self-Consistency; Artificial Intelligence(AI); Deep Learning(DL);
- &ensp;<span style="color:PaleVioletRed">Type</span>: Research Paper
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-08

> Quote learning from Professor [Yi Ma](https://people.eecs.berkeley.edu/~yima/)

<center class="center">
    <img src="./images/principles_mayi.png">
</center>

## **On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence**
**马毅老师对摘要的中文翻译**

> &ensp;&ensp;深度学习重振人工智能十年后的今天，我们提出一个**理论框架**来帮助理解深度神经网络在整个智能系统里面扮演的角色。我们引入了两个基本原则：**简约与自洽** -- 分别解释智能系统要学习什么以及如何学习。我们认为这两个原则是人工智能和自然智能之所以产生和发展的基石。虽然这两个原则的雏形早已出现在前人的经典工作里，但是我们对这些原则的重新表述使得它们变得可以**精准度量与计算**。确切地来说，简约与自洽这两个原则能自然地演绎出了一个高效计算框架：**压缩闭环转录**。这个框架统一并解释了现代深度神经网络以及众多人工智能实践的演变和进化。尽管本文主要用视觉数据建模作为例子，我们相信这两个原则将会有助于统一对各种自动智能系统的理解，并且提供了一个帮助理解**大脑工作机理**的框架。


### Overview
1. **Context and Motivation** 背景和动机
    - Intelligent agent and Brain 智能体和人脑的感知&预测&决策
    - Difficult to interpret 深度神经网络的黑盒(不可解释性)
    - Neural collapse and Mode collapse 神经元坍塌和模型坍塌
    - Catastrophic forgetting 灾难性遗忘
    - Robustness to deformations or adversarial attacks 形变和对抗性攻击的鲁棒性
    - A principled and unifying approach? 控制论的开环系统和闭环系统
    - Parsimony: what to learn 简约原则: 学习什么
    - Self-Consistency: how to learn 自洽原则: 如何学习
    - Reunite and Restate || Measurable and Computationally tractable 统一和重申 || 可精准测度和可解析计算
    - Information/Coding Theory & Control/Game Theory 信息论(编码理论) & 控制论(博弈论)
2. **Two Principles for Intelligence** 智能的两个原则
    - The Principle of Parsimony 简约原则
        - What to learn
        - Deep Networks from First Principle 深度神经网络的第一性原理
        - Maximizing Coding Rate Reduction, MCR2 最大化编码衰减
        - Modeling and Computing parsimony 模型设计和计算的简约性
        - Compression & Linearization & Sparsification 压缩化 & 线性化 & 稀疏化
        - Maximizing rate reduction(MR2) and Linear discriminative representation(LDR) 如何优化和学到什么
        - White-box deep networks from unrolling optimization versus backward propagation 前向展开和反向传播优化策略
        - CNN derived from shift-invariance and nonlinearity 卷积神经网络的归纳偏置(平移不变性和非线性)
        - Artificial selection and evolution of neural networks 网络模型的选择和进化(MLPs, CNNs, ResNets, Transformer, GNNs, NAS, AutoML)
    - The Principle of Self-Consistency 自洽原则
        - How to learn
        - Auto-encoding and its caveats with computability 自编码和可计算性
        - AE --> VAE --> GAN --> Flow-based --> DDPMs --> SDEs 生成模型的发展和可解析计算
        - Closed-loop data transcription for self-consistency 数据的闭环转录(数据自洽/无监督/有监督)
        - Self-learning through a self-critiquing game 通过自我评价完成自我学习
        - Self-consistent incremental and unsupervised learning 自洽的增量学习和无监督范式
3. **Universal Learning Engines** 统一的学习架构
    - First Principles and Deductive method 第一性原理和演绎法
    - “Unite and build” versus “divide and conquer” 从分而治之到统一联合
    - 3D Perception: Closing the Loop for Vision and Graphics 计算机视觉和计算机图形学的联合统一感知
    - Decision Making: Closing the Loop for Perception, Learning, and Action 智能体的决策和强化学习的高效范式
4. **A Broader Program for Intelligence** 普世的智能框架
    - Neuroscience of Intelligence 神经科学角度阐述
    - Mathematics of Intelligence 数学和可计算角度阐述
    - Toward Higher-Level Intelligence 符号等更高级智能角度阐述
    - Intelligence as interpretable and computable systems 智能系统的可解释性和可计算性
    - Symbolic or logic inference and Graph(Graph Neural Networks) 符号人工智能和图结构(图神经网络)
    - Intelligence as interpretable and computable systems 人工智能的可解释性和可计算性(正本清源-科学原则)
5. Conclusion 总结
6. Afterword and Acknowledgment 后记
7. References 参考文献

### A picture is worth a thousand words

<center class="center">
    <img src="./images/mayi_1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 1. Overall framework for a universal learning engine: seeking a compact and structured model for sensory data via a compressive closed-loop transcription: a (nonlinear) mapping $f(·; \theta) : x \mapsto z$ that maps high-dimensional sensory data with complicated lowdimensional structures to a compact structured representation. The model needs to be self-consistent, i.e., it can regenerate the original data via a map $g(·; \eta) : z \mapsto \hat{x}$ such that $f$ cannot distinguish despite its best effort. (Often, for simplicity, we omit the dependency of the mappings $f$ and $g$ on their parameters $\theta$ and $\eta$, respectively.) (Image source from Yi Ma)</div>
</center>

<center class="half">
    <img src="./images/mayi_2.png", width=50% /><img src="./images/mayi_3.png", width=50% />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 2. (Left)Seeking a linear and discriminative representation: mapping high-dimensional sensory data, typically distributed on many nonlinear low-dimensional submanifolds, onto a set of independent linear subspaces of the same dimensions as the submanifolds. (Right)Rate of all features $R = \log \#$(<span style="color:green">green spheres</span> + <span style="color:blue">blue spheres</span>); average rate of features on the two subspaces $R^{c} = \log \#$(<span style="color:green">green spheres</span>); rate reduction is the difference between the two rates: $\bigtriangleup R = R − R^{c}$. (Image source from Yi Ma)</div>
</center>

<center class="center">
    <img src="./images/mayi_4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 3. A basic way to construct the nonlinear mapping $f$: following the local gradient flow $\frac{\partial \bigtriangleup R(Z) }{\partial Z}$ of the rate reduction $\bigtriangleup R$, we incrementally linearize and compress features on nonlinear submanifolds and separate different submanifolds to respective orthogonal subspaces (the two dotted lines). (Image source from Yi Ma)</div>
</center>

<center class="center">
    <img src="./images/mayi_5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 4. Building blocks of the nonlinear mapping $f$. Left: one layer of the ReduNet as one iteration of projected gradient ascent, which precisely consists of expansive or compressive linear operators, a nonlinear softmax, plus a skip connection, and normalization. Middle and Right: one layer of ResNet and ResNeXt, respectively. (Image source from Yi Ma)</div>
</center>

<center class="center">
    <img src="./images/mayi_6.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 5. A compressive closed-loop transcription of nonlinear data submanifolds to an LDR, by comparing and minimizing the difference in $z$ and $\hat{z}$, internally. This leads to a natural pursuit-evasion game between the encoder/sensor $f$ and the decoder/controller $g$, allowing the distribution of the decoded $\hat{x}$ (the dotted blue curves) to chase and match that of the observed data $x$ (the solid black curves). (Image source from Yi Ma)</div>
</center>

<center class="center">
    <img src="./images/mayi_7.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 6. Incremental learning via a compressive closed-loop transcription. For a new data class $X_{new}$, a  new LDR memory $Z_{new}$ is learned via a constrained minimax game between the encoder and decoder subject to a constraint that memory of past classes $Z_{old}$ is preserved, as a “fixed point” of the closed loop. (Image source from Yi Ma)</div>
</center>

<center class="half">
    <img src="./images/mayi_8.png", width=75% /><img src="./images/mayi_9.png", width=25% />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 7. Left: comparison between $x$ and the corresponding decoded $\hat{x}$ of the auto-encoding learned in the unsupervised setting for the CIFAR-10 dataset (with 50,000 images in ten classes). Right: t-SNE of unsupervised-learned features of the ten classes and visualization of several neighborhoods with their associated images. Notice the local thin (nearly 1-D) structures in the visualized features, projected from a feature space of hundreds of dimensions. Correlations between unsupervised-learned features for 50,000 images that belong to ten classes (CIFAR-10) by the closedloop transcription. Block-diagonal structures consistent with the classes emerge without any supervision.(Image source from Yi Ma)</div>
</center>


### Understanding of Professional Terms for Intelligence

**tractable and intractable**
- [Why is the marginal likelihood difficult/intractable to estimate?](https://stats.stackexchange.com/questions/246179/why-is-the-marginal-likelihood-difficult-intractable-to-estimate)
- [What does a 'tractable' distribution mean?](https://stackoverflow.com/questions/43820240/what-does-a-tractable-distribution-mean)
- [What are the factors that cause the posterior distributions to be intractable?](https://stats.stackexchange.com/questions/4417/what-are-the-factors-that-cause-the-posterior-distributions-to-be-intractable)


**efficient(ly) and effective(ly)**
> 两个单词源于拉丁词根(单词) effect; efficient(ly) 在人工智能领域强调达到预期的效果(结果); effective(ly) 强调最终实现的效率(Inference time/Parameters/FLOPs)很高。也就是一项研究工作，首先需要 work, 然后需要 effective.

**neural collapse and mode collapse**
> **neural collapse** refers to the final representation for each class collapsing to a one-hot vector that carries no  information about the input except its class label. Richer features might be learned inside the networks, but their  structures are unclear and remain largely hidden. 简单理解就是学到神经元特征不够好，所表征的信息量有限，所导致的神经元(参数)冗余或者无效。**mode collapse** 主要是指代 GANs 一类方法，由于生成/对抗范式所引起的模型坍塌，即中间隐变量无法覆盖(cover)到所有的数据分布(mode/distribution)的问题，Denoising Diffusion Probability Models, DDPMs 这一类生成模型可以 cover all dist. than better GANs.

**catastrophic forgetting**
> **catastrophic forgetting** 灾难性遗忘，指的是


### Reference
----------------------------

[1] Yi Ma, Doris Tsao, Heung-Yeung Shum, "On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence," arXiv'2022

[Principles Paper on arXiv'2022](https://arxiv.org/abs/2207.04630)

[2] John Wright and Yi Ma, "High-Dimensional Data Analysis with Low-Dimensional Models: Principles, Computation, and Applications," book from Cambridge University Press, 2022

[Book on WebSite'2021](https://book-wright-ma.github.io/)
&emsp;&emsp;[Book Sources on CSDN'2021](https://bbs.csdn.net/forums/high-Dimensional-DA)

[3] Kwan Ho Ryan Chan, Yaodong Yu, Chong You, Haozhi Qi, John Wright, Yi Ma, "ReduNet: A White-box Deep Network from the Principle of Maximizing Rate Reduction," JMLR'2022

[ReduNet Paper on JMLR'2022](https://jmlr.org/papers/v23/21-0631.html)
&emsp;&emsp;[ReduNet Paper on arXiv'2021](https://arxiv.org/abs/2105.10446)
&emsp;&emsp;[ReduNet code on GitHub](https://github.com/Ma-Lab-Berkeley/ReduNet)
&emsp;&emsp;[MCR2 code on GitHub](https://github.com/Ma-Lab-Berkeley/MCR2)

[3] Xili Dai, Shengbang Tong, Mingyang Li, Ziyang Wu, Michael Psenka, Kwan Ho Ryan Chan, Pengyuan Zhai, Yaodong Yu, Xiaojun Yuan, Heung Yeung Shum, Yi Ma, "Closed-Loop Data Transcription to an LDR via Minimaxing Rate Reduction," published on Entropy, March 2022

[LDR Paper on Entropy'2022](https://www.mdpi.com/1099-4300/24/4/456)
&emsp;&emsp;[LDR Paper on arXiv'2022](https://arxiv.org/abs/2111.06636)
&emsp;&emsp;[LDR code on GitHub](https://github.com/delay-xili/ldr)

[4] Shengbang Tong, Xili Dai, Ziyang Wu, Mingyang Li, Brent Yi, Yi Ma, "Incremental Learning of Structured Memory via Closed-Loop Transcription," arXiv'2022

[Paper on arXiv'2022](https://arxiv.org/abs/2202.05411)

[5] Zepeng Zhang, Ziping Zhao, "Towards Understanding Graph Neural Networks: An Algorithm Unrolling Perspectiv,"  IEEE Transactions on Signal Processing'2022

[Unrolling GNNs Paper on arXiv'2022](https://arxiv.org/abs/2206.04471)

[6] Yongyi Yang, Zengfeng Huang, David Wipf, "Transformers from an Optimization Perspective,"  arXiv'2022

[Paper on arXiv'2022](https://arxiv.org/abs/2205.13891)

[7] Mary Phuong, Marcus Hutter, "Formal Algorithms for Transformers,"  arXiv'2022 from DeepMind

[Paper on arXiv'2022](https://arxiv.org/abs/2207.09238)

[8] Shengbang Tong, Xili Dai, Ziyang Wu, Mingyang Li, Brent Yi, Yi Ma, "Incremental Learning of Structured Memory via Closed-Loop Transcription,"  arXiv'2022

[Paper on arXiv'2022](https://arxiv.org/abs/2202.05411)
