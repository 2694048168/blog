# Foundation and Application of the Variational AutoEncoder

- &ensp;<span style="color:MediumPurple">Title</span>: Foundation and Application of the Variational AutoEncoder
- &ensp;<span style="color:Moccasin">Tags</span>: VAE; Flow; Deep Generative Models;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-08

---------------------
## Overview
- **Variational AutoEncoder**
    - Variational Lower Bound 变分下界推导
    - Conditional Variational Lower Bound 条件变分下界推导
    - Variational dequantitative 变分反量化
    - Variational data augmentation 变分数据增广
- **Flow Generative**
    - Flow 推导
- **Generative GAN**
    - GAN 推导


<center class="center">
    <img src="./images/Conditional_variational_autoencoder.png" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 1. Framework of Variational Autoencoder and Conditional Variational Autoencoder. (Image source from Stanislav S. Borysov, and Rich J. paper'2021)</div>
</center>


### 变分下界推导 Variational Autoencoder

&ensp;&ensp;&ensp;&ensp;VAEs 的需求(目的/目标)就是从概率论角度出发，对数据的建模都是建立在概率分布之上，其基本假设就是认为所需要生成的数据(或原始数据-text/audio/image)为 $P(X)$，根据中间变量(隐变量Z/中间分布/Gaussian dist.) 进行如下积分即可，

$$P(x) = \int_{z} P(x \mid z)P(z) \mathrm{d}z $$

&ensp;&ensp;&ensp;&ensp;假设 $x$ 为目标数据分布的随机变量，$z$ 为隐变量 latent，对 $x$ and $z$ 的联合概率分布进行以 $z$ 的边缘化操作即可得到关于 $x$ 的边缘概率分布，那么就可以根据 $P(x)$ 概率分布进行目标数据的生成(采样 sample)

$$
\begin{aligned}
P(x)
&= \int_{z} P(x,z) \mathrm{d}z \\
&= \int_{z} P(x \mid z)P(z) \mathrm{d}z \\
\end{aligned}
$$

&ensp;&ensp;&ensp;&ensp;显然，对于隐(随机)变量 $z$，其样本空间很有可能无限大(因为没有任何约束)，这样导致的情况就是对于绝大部分 $z$ 空间中的样本而言，$P(x \mid z) = 0$, 即对于目标数据分布 $x$ 没有任何关联/关系(完全独立的两个随机变量，相互之间没有提供任何信息量)。因此，需要对 $z$ 的样本空间进行限制和约束，一个直观且简单的技巧(trick) 就是构造一个新的概率分布 $Q(z \mid x)$, 以此来增大从 $z$ 样本空间到 $x$ 样本空间的可能性(概率)；即在约束样本空间 $z \sim Q(z \mid x)$ 前提下，计算 $P(x \mid z)$ 的期望(目标数据):

> [如何简单易懂理解变分推断](https://www.zhihu.com/question/41765860)

> ELBO, Evidence Lower Bound, 即证据下界; 这里的证据是指数据或可观测变量的概率密度。<br> how to understand the variational inference ? try to search with Google ! <br> 这就是 research paper 里面提到的 Variational Inference or Bayesian Inference or  [Variational Bayesian Inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods), 其实这一系列的名词本质上指代的是同一个思想内容。

<center class="half">
    <img src="./images/variational_inference.png", width=50% /><img src="./images/bayesian_formula.png", width=50% />
    <img src="./images/CAVI_algorithm_explain.jpg" />
</center>

> how to measure the **Exception of $P(x \mid z)$** ? &ensp;<span style="color:Red"> relative entropy or Kullback-Leibler divergence</span>

$$
\begin{aligned}
D_{KL}[ Q(z \mid x) || P(z \mid x) ]
&= E_{z \sim Q(z \mid x)} [ \log \frac{Q(z \mid x)}{P(z \mid x)} ] \\
&= E_{z \sim Q(z \mid x)} [ \log Q(z \mid x) - \log P(z \mid x) ] \\
&= E_{z \sim Q(z \mid x)} [ \log Q(z \mid x) - \log \frac{P(z, x)}{P(x)} ] \\
&= E_{z \sim Q(z \mid x)} [ \log Q(z \mid x) - \log \frac{P(x \mid z)P(z)}{P(x)} ] \\
&= E_{z \sim Q(z \mid x)} [ \log Q(z \mid x) - \log P(x \mid z)P(z) + \log {P(x)} ] \\
&= E_{z \sim Q(z \mid x)} [ \log Q(z \mid x) - \log P(x \mid z) - \log P(z) + \log P(x) ] \\
&= E_{z \sim Q(z \mid x)} [ \log Q(z \mid x) - \log P(x \mid z) - \log P(z)] + \log P(x) \\
&= E_{z \sim Q(z \mid x)} [ \log Q(z \mid x) - \log P(z)] - E_{z \sim Q(z \mid x)} [\log P(x \mid z)] + \log P(x) \\
&= \log P(x) + D_{KL}[ Q(z \mid x) || P(z) ] - E_{z \sim Q(z \mid x)} [\log P(x \mid z)] \\
\end{aligned}
$$

&ensp;&ensp;&ensp;&ensp;根据詹森不等式 (Jensen's inequality) 可知 KL 散度的非负性，同时结合以上等式便可以推导出(对数形式)变分下限 (variational lower bound, VLB)

$$ D_{KL}[ Q(z \mid x) || P(z \mid x) ]
= \log P(x) + D_{KL}[ Q(z \mid x) || P(z) ] - E_{z \sim Q(z \mid x)} [\log P(x \mid z)] $$

$$ \Rightarrow \log P(x) - D_{KL}[ Q(z \mid x) || P(z \mid x) ] 
= E_{z \sim Q(z \mid x)} [\log P(x \mid z)] - D_{KL}[ Q(z \mid x) || P(z) ]$$

$$ 
\begin{aligned}
\Rightarrow \log P(x)
&= D_{KL}[ Q(z \mid x) || P(z \mid x) ] + E_{z \sim Q(z \mid x)} [\log P(x \mid z)] - D_{KL}[ Q(z \mid x) || P(z) ] \\
&\ge E_{z \sim Q(z \mid x)} [\log P(x \mid z)] - D_{KL}[ Q(z \mid x) || P(z) ] \\
&= E_{z \sim Q(z \mid x)} [\log P(x \mid z) - \log \frac{Q(z \mid x)}{P(z)} ] \\
\end{aligned}
$$

&ensp;&ensp;&ensp;&ensp;当且仅当 $Q(z \mid x)$ 可以逼近 $P(x \mid z)$, 上述等式成立。其中，上述等式右边的第一项可以视为解码器(decoder)/重构部分; 第二项可以视为对(随机)变量 $Z$ 的先验分布 $P(z)$ 和后验分布 $Q(z \mid x)$ 之间距离度量。这样若希望对数似然 $P(x)$ 越大越好，即就等价于希望解码器基于 $z$ 分布预测 $X$ 的概率越大越好，同时保证先验分布 $P(z)$ 和后验分布 $Q(z \mid x)$ 间距离测度越小越好。这里对后验分布 $Q(z \mid x)$ 的构造(参数化)即可以视为编码器(encoder)，这里的参数化意味着可以是高斯分布或混合高斯分布这种直接可解析计算的方式(需要进行学习的参数, 无参)，也可以是通过神经网络(Neural Networks)进行拟合(有可学习的参数，有参)。

> VAE 的目标(objective) 就是最大化观测数据的 $X$ 的对数似然函数，$\mathcal{L}=\sum_{x} \log P(x)$

$$
\begin{aligned}
\log P(x)
&= \int q(z \mid x) \log p(x) \mathrm{d}z \\
&= \int q(z \mid x) \log \frac{p(x, z)}{p(z \mid x)} \mathrm{d}z \\
&= \int q(z \mid x) \log \frac{p(x, z)}{q(z \mid x)} \frac{q(z \mid x)}{p(z \mid x)} \mathrm{d}z \\
&= \int q(z \mid x) \log \frac{p(x, z)}{q(z \mid x)} \mathrm{d}z + \int q(z \mid x) \log \frac{q(z \mid x)}{p(z \mid x)} \mathrm{d}z \\
&= \int q(z \mid x) \log \frac{p(x, z)}{q(z \mid x)} \mathrm{d}z + D_{KL}(q(z \mid x) || p(z \mid x)) \\
&\ge \int q(z \mid x) \log \frac{p(x, z)}{q(z \mid x)} \mathrm{d}z \\
\end{aligned}
$$

$$
\begin{aligned}
\mathcal{L}_{ELBO} 
&= \mathcal{L}_{VLB} \\
&= \int q(z \mid x) \log \frac{p(x, z)}{q(z \mid x)} \mathrm{d}z \\
&= \int q(z \mid x) \log \frac{p(x \mid z) p(z)}{q(z \mid x)} \mathrm{d}z \\
&= \int q(z \mid x) \log \frac{p(z)}{q(z \mid x)} \mathrm{d}z + \int q(z \mid x) \log p(x \mid z) \mathrm{d}z \\
&= - \int q(z \mid x) \log \frac{q(z \mid x)}{p(z)} \mathrm{d}z + \int q(z \mid x) \log p(x \mid z) \mathrm{d}z \\
&= - D_{KL}(q(z \mid x) || p(z)) + \int q(z \mid x) \log p(x \mid z) \mathrm{d}z \\
&= \int q(z \mid x) \log p(x \mid z) \mathrm{d}z - D_{KL}(q(z \mid x) || p(z)) \\
&= E_{z \sim q(z \mid x)} [\log p(x \mid z) - \log \frac{q(z \mid x)}{p(z)} ] \\
\end{aligned}
$$


### 条件变分下界推导 Conditional Variational Autoencoder

> VAE 是个贝叶斯模型(Bayesian model), 其条件概率版本根据取条件概率的不同形式，自然会出现多种多样的条件变分模型(CVAE)。可以将条件(conditions c) 视为对 $Z$ 的一些先验补充信息(归纳偏置)，或者约束生成目标数据的方向(郎之万动力学, Langevin dynamics)，$P(x \mid c)$ 保持推导和理解不变, 直接套用原始的 VAE 模型，如下推导：

$$
\begin{aligned}
p(x \mid c)
&= \int q(z \mid x, c) \log \frac{p(x, z \mid c)}{q(z \mid x, c)} \mathrm{d}z + D_{KL}(q(z \mid x, c) || p(z \mid x, c)) \\
&\ge \int q(z \mid x, c) \log \frac{p(x, z \mid c)}{q(z \mid x, c)} \mathrm{d}z \\
&= \mathcal{L}_{ELBO} = \mathcal{L}_{VLB} \\
&= \int q(z \mid x, c) \log \frac{p(x \mid z, c) p(z \mid c)}{q(z \mid x, c)} \mathrm{d}z \\
&= \int q(z \mid x, c) \log \frac{p(z \mid c)}{q(z \mid x, c)} \mathrm{d}z + \int q(z \mid x, c) \log p(x \mid z, c) \mathrm{d}z \\
&= - D_{KL}(q(z \mid x, c) || p(z \mid c)) + \int q(z \mid x, c) \log p(x \mid z, c) \mathrm{d}z \\
&= \int q(z \mid x, c) \log p(x \mid z, c) \mathrm{d}z - D_{KL}(q(z \mid x, c) || p(z \mid c)) \\
&= E_{z \sim q(z \mid x, c)} [ \log p(x \mid z, c) - \log \frac{q(z \mid x, c)}{p(z \mid c)} ] \\
\end{aligned}
$$

> CVAE Case 1: 假设额外条件信息 $c$ 与隐变量 $z$ 没有直接关系(马尔可夫性质), 因此条件概率 $p(z \mid c) = p(z)$, 则条件变分推断下界可以变换如下： (该情况的前提假设和 CGAN paper 一样)

$$
\begin{aligned}
p(x \mid c) \ge
\mathcal{L}_{VLB}
&= \int q(z \mid x, c) \log p(x \mid z, c) \mathrm{d}z - D_{KL}(q(z \mid x, c) || p(z \mid c)) \\
&= \int q(z \mid x, c) \log p(x \mid z, c) \mathrm{d}z - D_{KL}(q(z \mid x, c) || p(z)) \\
&= E_{z \sim q(z \mid x, c)} [ \log p(x \mid z, c) - \log \frac{q(z \mid x, c)}{p(z)} ] \\
\end{aligned}
$$

> CVAE Case 2: 根据 CMMA(conditional multimodal autoencoder) paper 的前提假设, $p(x \mid c, z) = p(x \mid z)$, 则条件变分推断下界可以变换如下：

$$
\begin{aligned}
p(x \mid c) \ge
\mathcal{L}_{VLB}
&= \int q(z \mid x, c) \log p(x \mid z, c) \mathrm{d}z - D_{KL}(q(z \mid x, c) || p(z \mid c)) \\
&= \int q(z \mid x, c) \log p(x \mid z) \mathrm{d}z - D_{KL}(q(z \mid x, c) || p(z \mid c)) \\
&= E_{z \sim q(z \mid x, c)} [ \log p(x \mid z) - \log \frac{q(z \mid x, c)}{p(z \mid c)} ] \\
\end{aligned}
$$

> CVAE Case 3: 根据 VITS(conditional multimodal autoencoder) paper 的前提假设, 从 $D_{KL}[ q(z \mid x) || p(z \mid x, c) ]$ 进行推导, 则最终条件变分推断下界可以变换如下：

$$
\begin{aligned}
p(x \mid c) \ge
\mathcal{L}_{VLB}
&= \int q(z \mid x) \log p(x \mid z) \mathrm{d}z - D_{KL}(q(z \mid x) || p(z \mid c)) \\
&= E_{z \sim q(z \mid x)} [ \log p(x \mid z) - \log \frac{q(z \mid x)}{p(z \mid c)} ] \\
\end{aligned}
$$

&ensp;&ensp;&ensp;&ensp;From now on, we will refer to our method as Variational Inference with adversarial learning for end-to-end Text-to-Speech (VITS). 在 VITS(text-to-speech) ICML'2021 paper 中，$x$ 表示音频的梅尔频谱; $c$ 表示文本信息和对齐信息, 对齐信息是一个硬对齐单调矩阵(hard monotonic attention matrix), shape:[|text|, |z|]; $z$ 表示隐变量; 论文提到发现 $p(z \mid c)$ 表示为高斯函数分布后加上一个 flow 变换, 最终的音质效果更好. VITS paper 做的是一个端到端(stage  gap)的文本-语音的任务(single-stage: text-waveform VS two stage: text-spectrum-waveform), 其中涉及的技术点(GAN, VAE, CVAE, Flow, Stochastic duration predictor, Variational dequantitative, Variational data augmentation)


> [variational dequantizer](https://mtskw.com/posts/variational-dequantizer/)


### Reference
----------------------------

[1] Diederik P. Kingma, Max Welling, "Auto-Encoding Variational Bayes," ICLR'2014

[VAE Paper on ICLR'2014](https://openreview.net/forum?id=33X9fd2-9FyZd)
&emsp;&emsp;[VAE Paper on arXiv'2013](https://arxiv.org/abs/1312.6114)
&emsp;&emsp;[VAE Code on GitHub](https://github.com/AntixK/PyTorch-VAE)

[2] Diederik P. Kingma, Prafulla Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions," NeurIPS'2018

[Glow Paper on NeuriPS'2018](https://papers.nips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html)
&emsp;&emsp;[Glow Paper on arXiv'2018](https://arxiv.org/abs/1807.03039)
&emsp;&emsp;[Glow Original Code on GitHub](https://github.com/openai/glow)

[3] Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel, "Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design," ICML'2019

[Flow++ Paper on ICML'2019](http://proceedings.mlr.press/v97/ho19a.html)
&emsp;&emsp;[Flow++ Paper on arXiv'2019](https://arxiv.org/abs/1902.00275)
&emsp;&emsp;[Flow++ Code on GitHub](https://github.com/aravindsrinivas/flowpp)

[4] Stanislav S. Borysov, and Rich J. "Introducing synthetic pseudo panels: application to transport behaviour dynamics," Transportation Springer, vol. 48, pp. 1-28, 2021

[5] Jaehyeon Kim, Jungil Kong, Juhee Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," ICML'2021

[VITS Paper on ICML'2021](https://icml.cc/virtual/2021/spotlight/9246)
&emsp;&emsp;[VITS Paper on arXiv'2021](https://arxiv.org/abs/2106.06103)
&emsp;&emsp;[VITS Code on GitHub](https://github.com/jaywalnut310/vits)

[6] Jianfei Chen, Cheng Lu, Biqi Chenli, Jun Zhu, Tian Tian, "VFlow: More Expressive Generative Flows with Variational Data Augmentation," ICML'2020

[VFlow Paper on ICML'2020](https://icml.cc/virtual/2021/spotlight/9246)
&emsp;&emsp;[VFlow Paper v2 on arXiv'2022](https://arxiv.org/abs/2002.09741)
&emsp;&emsp;[VFlow Code on GitHub](https://github.com/thu-ml/vflow)

[7] Danilo Jimenez Rezende, Shakir Mohamed, "Variational Inference with Normalizing Flows," ICML'2015

[Normalizing Flows on ICML'2015](https://proceedings.mlr.press/v37/rezende15.html)
&emsp;&emsp;[Normalizing Flows on arXiv'2015](https://arxiv.org/abs/1505.05770)
&emsp;&emsp;[Normalizing Flows Implemented Code](https://paperswithcode.com/paper/variational-inference-with-normalizing-flows)