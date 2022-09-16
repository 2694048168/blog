# Variational AutoEncoder and and Flow-based and GANs

- &ensp;<span style="color:MediumPurple">Title</span>: Variational AutoEncoder and and Flow-based and GANs
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
- **MCMC Algorithrm**


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

### Flow Generative


### Generative GAN


### MCMC Algorithm

> [Markov Chain Monte Carlo, MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) 马尔科夫链蒙特卡洛方法; <br> MCMC 是一组利用马尔科夫链从随机分布中取样的算法，生成的马氏链即是对于目标分布的近似估计。常见算法：Metropolis-Hastings 算法；Gibbs 取样算法；Hamiltonian Monte Carlo 算法；都是通过随机游走对目标分布进行采样。

some **Python** packages for MCMC
- [emcee](https://emcee.readthedocs.io/en/stable/)
- [ParaMonte](https://www.cdslab.org/paramonte/notes/installation/python/)
- [ParaMonte](https://en.wikipedia.org/wiki/PyMC)
- [MCMC-刘建平](https://www.cnblogs.com/pinard/p/6625739.html)
- [马尔可夫链蒙特卡洛方法](https://www.jiqizhixin.com/articles/2017-12-24-6)
- [马尔可夫链蒙特卡罗算法 MCMC](https://zhuanlan.zhihu.com/p/37121528)


#### **蒙特卡罗方法**
&ensp;&ensp;&ensp;&ensp;蒙特卡罗是一个赌场的名称，统计学中蒙特卡罗方法是一种随机模拟的方法，这很像赌博场里面的扔骰子的过程。最早的蒙特卡罗方法都是为了求解一些不太好求解的求和或者积分问题, 难求解出 $f(x)$ 的原函数，那么这个积分比较难求解。

$$ \theta = \int_{a}^{b} f(x) \mathrm{d}x $$
$$ \theta = (b - a) f(x_{0}) ; x_{0} \sim [a, b] $$
$$ \theta = \frac{b - a}{n} \sum_{i=0}^{n-1} f(x_{i}) ; x_{i} \sim [a, b]$$
$$
\begin{aligned}
\theta
&= \int_{a}^{b} f(x) \mathrm{d}x \\
&= \int_{a}^{b} \frac{f(x)}{p(x)} p(x) \mathrm{d}x \\
&\approx \frac{1}{n} \sum_{i=0}^{n-1} \frac{f(x_{i})}{p(x_{i})} \\
\end{aligned}
$$

其中，简单的近似求解方法是在 $[a, b]$ 之间随机的采样一个点 $x_{0}$ , 然后用 $f(x_{0})$ 代表在 $[a, b]$ 区间上所有的 $f(x)$ 的值; 采样 $[a, b]$ 区间的 $n$ 个值：$x_{0}, x_{1}, ... x_{n−1}$ , 用它们的均值来代表 $[a, b]$ 区间上所有的 $f(x)$ 的值, 隐含假定 $x$ 在$[a, b]$ 之间是均匀分布的; 可以利用 $x$ 在 $[a, b]$ 的概率分布函数 $p(x)$, 进而更准确估计。这就是蒙特卡罗方法的一般形式，当然这里是连续函数形式的蒙特卡罗方法，但是在离散时一样成立。

**概率分布采样**
&ensp;&ensp;&ensp;&ensp;关键在于获取 $x$ 的概率分布 $p(x)$ 或者得到基于其概率分布的采样样本集合; 对于均匀采样, 可以通过同余弦发生器生成伪随机数样本, 而对于常见(概率论中 $\Beta$ dist.; $Gamma$ dist. )的连续和离散分布可以通过均匀采样的样本进行转换计算获得。

**接受-拒绝采样**
&ensp;&ensp;&ensp;&ensp;对于复杂未知的分布, 无法进行样本的采样; 可以用一个已知的分布 $q(x)$ (如高斯分布)去采样, 然后按照一定规则拒绝某些样本, 以达到不断逼近期望分布 $p(x)$ 的目标, 其中 $q(x)$ 也称之为 proposal distribution. 采用过程如下图，设定一个方便采样的常用概率分布函数 $q(x)$，以及一个常量 $k$，使得 $p(x)$ 总在 $k q(x)$ 的下方; 首先，通过概率采样方法得到 $q(x)$ 的一个样本 $z_{0}$ ，然后，从均匀分布 $(0, k q(z_{0}))$ 中采样得到一个值 $u$ , 如果 $u$ 落在了上图中的灰色区域，则拒绝这次抽样，否则接受这个样本 $z_{0}$; 不断重复以上过程得到 $n$ 个接受的样本 $z_{0}, z_{1}, ... z_{n−1}$.

<center class="center">
    <img src="./images/proposal_dist.png" />
</center>

**存在的问题**
- 对于一些二维分布 $p(x, y)$，有时候只能得到条件分布 $p(x|y)$ 和 $p(y|x)$ , 很难得到二维分布 $p(x,y)$ 一般形式，这时无法用接受-拒绝采样得到其样本集
- 对于一些高维的复杂非常见分布 $p(x{1}, x_{2}, ..., x_{n})$，要找到一个合适的 $q(x)$ 和 $k$ 非常困难
- 蒙特卡罗方法作为一个通用的采样模拟求和的方法，必须解决如何方便得到各种复杂概率分布的对应的采样样本集的问题
- 马尔科夫链就是帮助找到这些复杂概率分布的对应的采样样本集的一个最佳方法

#### **马尔科夫链**
&ensp;&ensp;&ensp;&ensp;马尔科夫链假设某一时刻状态转移的概率只依赖于它的前一个状态; 时刻 $X_{t+1}$ 的状态的条件概率仅仅依赖于时刻 $X_{t}$; 既然某一时刻状态转移的概率只依赖于它的前一个状态，那么只要能求出系统中任意两个状态之间的转换概率，这个马尔科夫链的模型就确定了, 同时根据马尔科夫链可以写出对应的马尔科夫链模型的状态转移矩阵.

$$ p(x_{t+1} | ... x_{t-2}, x_{t-1}, x_{t}) = p(x_{t}) $$

**状态转移矩阵性质**
- 马尔科夫链模型的状态转移矩阵收敛到的稳定概率分布与初始状态概率分布无关
- 非常好的性质，也就是说，如果得到了这个稳定概率分布对应的马尔科夫链模型的状态转移矩阵，则可以用任意的概率分布样本开始，带入马尔科夫链模型的状态转移矩阵，这样经过一些序列的转换，最终就可以得到符合对应稳定概率分布的样本
- 该性质对离散状态，连续状态都成立
- 对于一个确定的状态转移矩阵 $P$，它的 $n$ 次幂 $P^{n}$ 在当 $n$ 大于一定的值的时候也可以发现是确定的
- 可以查阅精准的数学语言描述
- 根据马尔科夫链的良好性质，可以基于马尔科夫链进行采样，得到样本集合, 就可以计算蒙特卡洛模拟求和了

**存在的问题**
- 假定可以得到需要采样样本的平稳分布所对应的马尔科夫链状态转移矩阵，那么就可以用马尔科夫链采样得到需要的样本集，进而进行蒙特卡罗模拟
- 一个重要的问题是，随意给定一个平稳分布 $π$, 如何得到它所对应的马尔科夫链状态转移矩阵 $P$
- MCMC 采样通过迂回的方式解决了这个大问题

#### **MCMC采样**
- 一般情况下，目标平稳分布 $π(x)$ 和某一个马尔科夫链状态转移矩阵 $Q$ 不满足细致平稳条件
- 引入一个 $α(i,j)$ 使细致平稳条件成立, 去等式成立的条件
- MCMC的采样算法过程中最难和最关键的一步就是 **接受率**这个数值过低的问题
- Metropolis-Hastings (M-H) 采样解决了 MCMC 算法中采样接受率过低的问题
- M-H 采样已经可以很好的解决蒙特卡罗方法需要的任意概率分布的样本集的问题
- 但 M-H 采样有两个缺点：
    - 需要计算接受率，在高维时计算量大
    - 并且由于接受率的原因导致算法收敛时间变长
    - 有些高维数据，特征的条件概率分布好求，但是特征的联合分布不好求
- 需要一个好的方法来改进 M-H 采样，这就是 Gibbs 采样
- Gibbs 采样通过重新寻找合适的细致平稳条件
    - 二维 Gibbs 采样
    - 多维 Gibbs 采样


<details>
<summary> <span style="color:Teal">MCMC Algorithm Example</span> </summary>

```python
import numpy as np
from scipy.special import beta
import scipy.stats

class MCMC:
    """Markov Chain Monte Carlo method class used by MCMC Algorithms functions (e.g. metropolis)

    Used to initialize instances for MCMC Algorithm functions

    Return an MCMC object

    Parameters
    ----------
    dfunc : function
        self-defined density function which take only one parameter (i.e. r.v. X)
    chain : int, optional
        the length of Markov Chain to be generated (default 5000)
    theta_init : float, optional
        initial value for the chain of θ (default 0.5)
    jumpdist : scipy.stats.rv_continuous or scipy.stats.rv_discrete, optional
        the distribution that proposed jump (Δθ) follows (default scipy.stats.norm(0,0.2))
    space : list, optional
        the accepted span of the parameter θ (default [-np.inf,np.inf])
    burnin : int, optional
        the length of the chain to be burned from the beginning (default 0)
    seed : int or None, optional
        random seed (default None)

    Usage:
    -----
    >>> import metropolis
    >>> density = lambda x: scipy.stats.gamma(2,loc=4,scale=5).pdf(x)
    >>> d = metropolis.MCMC(density, chain = 10000, jumpdist=scipy.stats.norm(loc=0,scale=2), space = [0,np.inf])
    >>> d
    <metropolis.MCMC object ...>
    """
    def __init__(self, 
                 dfunc, 
                 chain=5000, 
                 theta_init=0.5, 
                 jumpdist=scipy.stats.norm(0,0.2), 
                 space=[-np.inf,np.inf], 
                 burnin=0, 
                 seed=None) -> None:
        self.dfunc = dfunc
        self.chain = chain
        self.theta_init = theta_init
        self.jump = jumpdist
        self.space = space
        self.burnin = burnin
        self.seed = seed

    def metropolis(self, *arg, **kwarg):
        """
        function applying Metropolis Algorithm to MCMC objects

        Parameters
        ----------
        dfunc : function, *arg, optional (update)
            self-defined density function which take only one parameter (i.e. r.v. X)
        chain : int, optional (update)
            the length of Markov Chain to be generated
        theta_init : float, optional (update)
            initial value for the chain of θ (default 0.5)
        jumpdist : scipy.stats.rv_continuous or scipy.stats.rv_discrete, optional (update)
            the distribution that proposed jump (Δθ) follows (default scipy.stats.norm(0,0.2))
        space : list, optional (update)
            the accepted span of the parameter θ (default [-np.inf,np.inf])
        burnin : int, optional (update)
            the length of the chain to be burned from the beginning (default 0)
        seed : int or None, optional (update)
            random seed (default None)

        Examples
        --------
        >>> import metropolis
        >>> from scipy.special import beta
        >>> density = lambda x: x**14 * (1-x)**6 / beta(15,7) # Beta(15,7) distribution density function
        >>> d = metropolis.MCMC(density, space = [0,1], burnin = 5, seed = 72)
        >>> result = d.metropolis(chain = 50, theta_init = 0.1)
        >>> result
        [0.8176960093922774, 0.6965658096789994, 0.7918980615882665, 0.7742394795862185, 0.7742394795862185, 0.6215414421599866, 0.6215414421599866, 0.6468248283946754, 0.7523307146724492, 0.7523307146724492, 0.7680822171469903, 0.6717376155310788, 0.6717376155310788, 0.8189878864825765, 0.7533257832372013, 0.7648206889254298, 0.7648206889254298, 0.7648206889254298, 0.7648206889254298, 0.7648206889254298, 0.7967947965010628, 0.707139903476412, 0.707139903476412, 0.707139903476412, 0.707139903476412, 0.7949590848197712, 0.48018105229278457, 0.48018105229278457, 0.6163978253871556, 0.6163978253871556, 0.6163978253871556, 0.6241841537823003, 0.6241841537823003, 0.6241841537823003, 0.6241841537823003, 0.6922332333961135, 0.8204027203103916, 0.7521267229207833, 0.7521267229207833, 0.808566620237602, 0.808566620237602, 0.6460288022318678, 0.5452360032045938, 0.5452360032045938, 0.5934357613729606]
        """
        # update attributes
        for args in arg:
            self.dfunc = args
        
        for kw, value in kwarg.items():
            if kw == 'chain':
                self.chain = value
            elif kw == 'theta_init':
                self.theta_init = value
            elif kw == 'jumpdist':
                self.jump = value
            elif kw == 'space':
                self.space = value
            elif kw == 'burnin':
                self.burnin = value
            elif kw == 'seed':
                self.seed = value
            else:
                raise Exception(f'keyword argument "{kw}" not supported',)

        # check if dfunc callable
        if not callable(self.dfunc):
            raise Exception("dfunc must be a function. recreate the object with a valid density function")
        
        # Metropolis Algorithm
        theta_cur = self.theta_init
        theta_freq = [self.theta_init]
        
        rng = np.random.default_rng(self.seed)

        while True:
            Delta_theta = self.jump.rvs(random_state=rng)
            theta_pro = theta_cur + Delta_theta

            if theta_pro < self.space[0] or theta_pro > self.space[1]:
                pmoving = 0
            elif self.dfunc(theta_pro) == 0:
                pmoving = 1
            else:
                pmoving = min(1,self.dfunc(theta_pro)/self.dfunc(theta_cur))
            
            # np.random.rand()
            if scipy.stats.uniform().rvs(random_state=rng) <= pmoving:
                theta_freq.append(theta_pro)
                theta_cur = theta_pro
            else:
                theta_freq.append(theta_cur)

            if len(theta_freq) >= self.chain:
                break

        return theta_freq[self.burnin:]


# --------------------------
if __name__ == "__main__":
    # density = lambda x: x**14 * (1-x)**6 / beta(15, 7)
    density = lambda x: scipy.stats.gamma(2, loc=4, scale=5).pdf(x)
    d = MCMC(density, chain=10, jumpdist=scipy.stats.norm(loc=0, scale=2), space=[0, np.inf])
    print(d)
    result = d.metropolis(chain=100, seed=42)
    print(result)
```

</details>


-------------
Cited as:
```shell
@article{WeiLi2022VAE-Flow-GAN,
  title   = Variational AutoEncoder and and Flow-based and GANs,
  author  = Wei Li,
  journal = https://2694048168.github.io/blog/,
  year    = 2022-09,
  url     = https://2694048168.github.io/blog/#/PaperMD/vae
}
```


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

[8] Dinghuai Zhang, Ricky T. Q. Chen, Nikolay Malkin, Yoshua Bengio, "Unifying Generative Models with GFlowNets," Accepted to ICML 2022 Beyond Bayes workshop

[GFlowNets on arXiv'2022](https://arxiv.org/abs/2209.02606)