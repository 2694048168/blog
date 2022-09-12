# Image Generation via Diffusion Models and Scored-Matching and SDEs

- &ensp;<span style="color:MediumPurple">Title</span>: Image Generation via Diffusion Models and Scored-Matching and SDEs
- &ensp;<span style="color:Moccasin">Tags</span>: Generative Models; Denoising Diffusion Probability Models, DDPMs; Scored-Matching; Score-based Multi-level Noise Matching; Stochastic Differential Equations, SDEs;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">Revision of DateTime</span>: 2022-08-06; 2022-08-13; 2022-08-29; 2022-09-05;

> Deep Generative Learning: Learning to generate data

## Overview of Diffusion Models and Scored-Matching Models and SDE Methods
1. **Magical Incredible Images and some interesting demos**
2. **Research Paradigm of Image Generation and Deep Generative Learning**
3. **Mathematical deduction** 
4. **Coding implementation and a toy demo**
5. **Score and Naive Score-Based Models**
6. **Noise Conditional Score Networks (NCSN)**
7. **Stochastic Differential Equations (SDEs)**
8. **Coding implementation and MNIST demo via SDEs**
9. **Reference**

<center>
    <img src="./images/Diffusion_Models_Slide.png" />
</center>

**Application of Diffusion**

- [AI] Content Generation: <span style="color:Purple">StyleGAN3 example images</span>
- [ML] Representation Learning: <span style="color:Purple">Learning from limited labels</span>
- [AI] Artistic Tools: <span style="color:Purple">music, text, audio, painting</span>
- [CV] (Controllable) Image Generation: <span style="color:Purple">text-to-image(DALL-E 2), Image-to-image translation(SR)</span>
- [NLP] (Conditional) Text Generation: <span style="color:Purple">text-to-image(DALL-E 2), Image-to-image translation</span>
- [Graph] Molecule Generation: <span style="color:Purple">分子生成；抗癌药物研究；抗体分子研究</span>
- [Protein] Protein Design: <span style="color:Purple">蛋白质设计(Generating sequence and structure of amino acids)</span>

## Magical Images 魔法式图像

> [DeepFake tech.](https://en.wikipedia.org/wiki/Deepfake)  a  portmanteau word of "deep learning" and "fake" since 2016.

[Stable Diffusion Release](<https://stability.ai/blog/stable-diffusion-public-release> (Robin Rombach, Andreas Blattmann, Dominik Lorenz, et al. "High-Resolution Image Synthesis with Latent Diffusion Models," CPVR'2022 Oral)) : **Firstly the public release of stable diffusion for researchers and user**

<center class="center">
    <img src="./images/stable_diffusion.png" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 0. We condition LDMs either via concatenation or by a more general cross-attention mechanism from High-Resolution Image Synthesis with Latent Diffusion Models paper. (Image source from stable diffusion paper, CVPR'2022)</div>
</center>

> Some magical and interesting demos on InternetWeb for Stable-Diffusion and Disco-Diffusion. You can Google search on Web and try it with creative and generate ideas (music/image).

[DALLE 2](<https://openai.com/dall-e-2/> (Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents," OpenAI arXiv'2022)) : **a new AI system that can create realistic images and art from a description in natural language**

<center class="half">
    <img src="./images/DALLE2_0.png", width="50%" /><img src="./images/DALLE2_1.png", width="50%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 1. Some Samples from DELLE 2. (Image source from DALLE-2 paper, OpenAI)</div>
</center>

[Imagen](<https://imagen.research.google/> (Chitwan Saharia, William Chan, Saurabh Saxena, et al. "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding," Google arXiv'2022)) : **unprecedented photorealism x deep level of language understanding**

<center class="half">
    <img src="./images/Imagen_0.png", width="50%" /><img src="./images/Imagen_1.png", width="50%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 2. Some Samples from Imagen. (Image source from Imagen paper, Google)</div>
</center>

-------

## Image Generation Paradigm 图像生成研究范式

**Reference Blogs**

- [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
- [Yang Song publication paper: Score-based and SDEs](https://yang-song.net/publications/)
- [CVPR 2022 Tutorial: Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://cvpr2022-tutorial-diffusion-models.github.io/)
- [Diffusion Models for Deep Generative Learning](https://zaixiang.notion.site/Diffusion-Models-for-Deep-Generative-Learning-24ccc2e2a11e40699723b277a7ebdd64)
- [Computer Vison: Models, Learning, and Inference 中英版本图书](https://item.jd.com/12218342.html)
- [Awesome Diffusion Models](https://github.com/heejkoo/Awesome-Diffusion-Models)
- [The Annotated Diffusion Model on Hugging Face](https://huggingface.co/blog/annotated-diffusion)
- [IDDPM Code on GitHub from OpenAI](https://github.com/openai/improved-diffusion)
- [Stable Diffusion Demo at Hugging Face](https://huggingface.co/spaces/stabilityai/stable-diffusion)
- [Stable Diffusion rest at Hugging Face](https://huggingface.co/spaces/huggingface/diffuse-the-rest)
- [Exponentially Weighted Average for Deep Neural Networks](https://medium.datadriveninvestor.com/exponentially-weighted-average-for-deep-neural-networks-39873b8230e9)
- [Understanding “Exponential Moving Averages” in Data Science](https://medium.com/analytics-vidhya/understanding-exponential-moving-averages-e3f020d9d13b)

**Paradigm of Deep Generative Models**

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/generative-overview.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 3. Overview of different types of generative models. (Image source from Lilian Weng blog, that the Applied AI Research Manager, OpenAI)</div>
</center>
<!-- ![generative-overview](./Images/generative-overview.png) -->

**Quote from Yang Song (PhD, Stanford)**

> Existing generative modeling techniques can largely be grouped into two categories based on how they represent probability distributions. <br> (1) likelihood-based models, which directly learn the distribution’s probability density (or mass) function via (approximate) maximum likelihood. Typical likelihood-based models include autoregressive models,  normalizing flow models, energy-based models (EBMs), and variational auto-encoders (VAEs). <br> (2) implicit generative  models, where the probability distribution is implicitly represented by a model of its sampling process. The most prominent example is generative adversarial networks (GANs), where new samples from the data distribution are synthesized by  transforming a random Gaussian vector with a neural network. <br> Likelihood-based models and implicit generative models, however, both have significant limitations. Likelihood-based models either require strong restrictions on the model  architecture to ensure a tractable normalizing constant for likelihood computation, or must rely on surrogate objectives to approximate maximum likelihood training. Implicit generative models, on the other hand, often require adversarial training, which is notoriously unstable and can lead to mode collapse.

Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data). Diffusion Models 和其他生成模型最大的区别是它的 latent code(z) 和原图是同尺寸大小的; 当然也有基于压缩的 Latent Diffusion Model &ensp;[CVPR'2022](<https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html> (Robin Rombach, Andreas Blattmann, et al. "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR'2022))

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/2022_CVPR_DDPMs_tutorial_fig.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 4. The Landscape of Deep Generative Learning. (Image source from 2022-CVPR-Tutorial DDPMs slides)</div>
</center>

---------------------

## Elegant Mathematical 优雅的数学原理

### 1. 随机变量及其概率分布 Random Variables & Probability Distribution

> A random variable $x$ denote a quantity that is uncertain. This information is captured by the probability distribution $P_{r}(x)$ of the random variable. A random variable may be **discrete** or **continuous**.

离散随机变量的概率分布常用直方图或者 Hinton 图进行可视化；连续随机变量的概率分布常用概率密度函数 PDF(probability density function, pdf) 进行可视化。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/probability_visual.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 5. Visualization of PDF for discrete and continuous random variable. (Image source from the book, "Computer Vision: Models, Learning, and Inference")</div>
</center>

$$ \sum_{i=1}^{n} p_{r}(x) = 1 $$

$$ \int_{-\infty}^{\infty} p_{r}(x)\mathrm{d}x=1 $$

### 2. 联合概率 Joint Probability

> 度量多元随机变量组合出现的可能性。$ p_{r}(x, y) $ and $ p_{r}(x, y, z) $

$$ \int \int p_{r}(x, y)\mathrm{d}x\mathrm{d}y = 1$$
$$ \sum \sum p_{r}(x, y) = 1$$
$$ \sum \int p_{r}(x, y)\mathrm{d}y  = 1$$
$$ \int \sum p_{r}(x, y)\mathrm{d}x  = 1$$


$ p_{r}(x, y, z) $ 可以推广到多元随机变量的联合概率分布；同时也可以推广到多维随机变量 $ p_{r}(X) $ , 其中 $ X = [x_{1}, x_{1}, x_{1}, \dots, x_{k}]^T $; 同时也可以推广多元多维随机变量的联合概率分布 $ p_{r}(X, Y) $ and $ p_{r}(X, Y, Z) $

### 3. 边缘化 Marginalization

> We can recover the probability distribution of any single variable from a joint distribution by summing (discrete case) or integrating (continuous case) over all the other variables. In general, we can recover the joint probability of any subset of variables, by marginalizing over all of the others. 

任意单随机变量的概率分布可以通过其联合概率分布对其他随机变量进行积分或者求和计算得出，该计算过程称之为边缘化 (marginalization)，该过程计算结果称之为边缘分布 (marginal distribution)。


$$ p_{r}(x) = \int p_{r}(x, y) \mathrm{d}y $$
$$ p_{r}(y) = \int p_{r}(x, y) \mathrm{d}x $$
$$ p_{r}(z) = \int \int p_{r}(x, y, z) \mathrm{d}x \mathrm{d}y $$
$$ p_{r}(x, y) = \sum_{w} \int p_{r}(w, x, y, z) \mathrm{d}z $$


### 4. 条件概率 Conditional Probability

> The conditional probability of $ x $ given that $ y $ takes value $ y^{\ast} $ tells us the relative propensity of the random variable $ x $ to take different outcomes given that the random variable $ y $ is fixed to value $ y^{\ast} $ . 

$$ p_{r}(x|y=y^{\ast})=\frac{p_{r}(x,y=y^{\ast})}{\int p_{r}(x,y=y^{\ast})\mathrm{d}x}=\frac{p_{r}(x,y=y^{\ast})}{p_{r}(y=y^{\ast})} $$
$$ p_{r}(x|y) = \frac{p_{r}(x,y)}{p_{r}(y)} $$
$$ p_{r}(x,y) = p_{r}(x|y)p_{r}(y) $$
$$ p_{r}(x,y) = p_{r}(y|x)p_{r}(x) $$


**利用条件概率分布可以不断将联合概率分布分解为乘积的形式**


$$
\begin{aligned}
p_{r}(w,x,y,z)
&= p_{r}(w,x,y|z)p_{r}(z) \\
&= p_{r}(w,x|y,z)p_{r}(y|z)p_{r}(z) \\
&= p_{r}(w|x,y,z)p_{r}(x|y,z)p_{r}(y|z)p_{r}(z) \\
\end{aligned}
$$


**同时利用上马尔科夫链的条件独立性质 $x -> y -> z$**


$$
\begin{aligned}
p_{r}(x,y,z)
&= p_{r}(x,y|z)p_{r}(z) \\
&= p_{r}(x|y,z)p_{r}(y|z)p_{r}(z) \\
&= p_{r}(x|y)p_{r}(y|z)p_{r}(z) \\
\end{aligned}
$$


### 5. 贝叶斯公式 Bayes's Rule

<span style="color:Teal">**Thomas Bayes(1701-1761), 解决“逆概”问题**</span>

**贝叶斯要解决的问题：**

- 正向概率：假设袋子里面由N个白球，M个黑球，随机摸出一个球，是黑球的可能性是多少？
- 逆向概率：如果事先并不知道袋子里面黑白球的比例，而是闭着眼睛随机摸出一个(或好几个)球，观察这些取出来的球的颜色之后，那么可以就此对袋子里面的黑白球的比例作出什么样的推测？

> 例子：学校里面男生比例为60%；女生比例为40%；男生总是穿长裤；女生则是一半穿长裤，一半穿裙子；

- 正向概率：随机抽取一个学生，是穿长裤的概率和穿裙子的概率分别是多大？
- 逆向概率：抽取到一个穿长裤的学生，无法确定性别，推断其是女生的概率是多大？


$$
\begin{aligned}
& \text{假设学校学生的总人数为M} \\
& \text{穿长裤的男生为：} M\ast P(Boy)\ast P(Pants|Boy) \\
& \text{穿长裤的女生为：} M\ast P(Girl)\ast P(Pants|Girl) \\
& \text{穿长裤的学生总人数为：}  \\
& M\ast P(Boy)\ast P(Pants|Boy) + M\ast P(Girl)\ast P(Pants|Girl) \\
& \text{求解穿长裤的学生里面有多少女生：}\\
P(Girl|Pants)
&= \frac{M\ast P(Girl)\ast P(Pants|Girl)}{M\ast P(Boy)\ast P(Pants|Boy) + M\ast P(Girl)\ast P(Pants|Girl)} \\
&= \frac{P(Girl)\ast P(Pants|Girl)}{P(Boy)\ast P(Pants|Boy) + P(Girl)\ast P(Pants|Girl)}  & \text{;分母就是P(Pants)} \\
&= \frac{P(Pants|Girl) \ast P(Girl)}{P(Pants)} \\
\end{aligned}
$$


**贝叶斯公式 Bayes's Rule**


$$ P(A|B) = \frac{P(B|A) P(A)}{P(B)} $$


> we can expressed the joint probability in two ways. We can combine these formulations to find a relationship between $ P_{r}(x|y)$ and $P_{r}(y|x)$


$$ p_{r}(x, y) = p_{r}(y|x)p_{x} = p_{r}(x|y)p_{y} $$
$$ p_{r}(y|x) = \frac{p_{r}(x|y)p_{r}(y)}{p_{r}(x)} $$


$$
\begin{aligned}
{\color{DeepPink}p_{r}(y|x)}
&= \frac{{\color{cyan}p_{r}(x|y)} {\color{red}p_{r}(y)}} {\color{blue}p_{r}(x)} & \text{ ;origin Bayes's rule} \\
&= \frac{p_{r}(x|y)p_{r}(y)}{\int p_{r}(x,y) \mathrm{d}y} & \text{ ;marginal dist.}\\
&= \frac{p_{r}(x|y)p_{r}(y)}{\int p_{r}(x|y)p_{r}(t) \mathrm{d}y} & \text{ ;conditional dist.} \\
\end{aligned}
$$


其中 $p_{r}(y|x)$ 称之为后验概率 (posterior); $p_{r}(y)$ 称之为先验概率 (prior); $p_{r}(x)$ 称之为证据 (evidence); $p_{r}(x|y)$ 称之为似然性 (likelihood).

**多元变量的贝叶斯公式 Bayes's rule**


$$
\begin{aligned}
p_{r}(x_{t-1},x_{t},x_{0})  
&= p_{r}(x_{t-1}, x_{t}|x_{0})p_{r}(x_{0}) \\
&= p_{r}(x_{t-1}|x_{t},x_{0})p_{r}(x_{t}|x_{0})p_{r}(x_{0}) \\
\end{aligned}
$$


$$
\begin{aligned}
p_{r}(x_{t},x_{t-1},x_{0})  
&= p_{r}(x_{t}, x_{t-1}|x_{0})p_{r}(x_{0}) \\
&= p_{r}(x_{t}|x_{t-1},x_{0})p_{r}(x_{t-1}|x_{0})p_{r}(x_{0}) \\
\end{aligned}
$$


两种形式的联合概率分布表示的是同一个联合概率分布，因此是完全相等的，故此可以推导出：


$$
\begin{aligned}
p_{r}(x_{t-1}|x_{t},x_{0})
&= \frac{p_{r}(x_{t},x_{t-1},x_{0})}{p_{r}(x_{t}|x_{0})p_{r}(x_{0})} \\
&= \frac{p_{r}(x_{t}|x_{t-1},x_{0})p_{r}(x_{t-1}|x_{0})p_{r}(x_{0})}{p_{r}(x_{t}|x_{0})p_{r}(x_{0})} \\
&= \frac{p_{r}(x_{t}|x_{t-1},x_{0})p_{r}(x_{t-1}|x_{0})}{p_{r}(x_{t}|x_{0})} \\
\end{aligned}
$$


### 6. 独立性 Independence

> 在概率论和统计学中，独立同分布 (Independent and Identically Distributed, IID, iid, i.i.d.) 的假设指的是一组随机变量中每一个变量的概率分布相同，且这些随机变量互相独立。

若 $x$ and $y$ 互相独立，则条件概率分布和联合概率分布为，即独立随机变量的联合概率分布等于边缘概率分布的累计乘积


$$ p_{r}(x|y) = p_{r}(x) $$
$$ p_{r}(y|x) = p_{r}(y) $$
$$ p_{r}(x, y) = p_{r}(x|y)p_{r}(y) = p_{r}(x)p_{r}(y) = p_{r}(x|y)p_{r}(y|x)$$


**条件的独立性 Conditional Independence**

> Confusingly, the conditional independence of $x_{1}$ and $x_{3}$ given $x_{2}$ does not mean that $x_{1}$ and $x_{3}$ are themselves independent. It merely implies that if we know variable $x_{2}$, then $x_{1}$ provides no further information about $x_{3}$ and vice versa.

对于多元随机变量 $p_{r}(x_{1},x_{2},x_{3})$, 若在 $x_{2}$ 条件下，$x_{1}$ 和 $x_{3}$ 互相独立，这种情况称之为条件独立性; 注意，条件独立是对称的。可以将联合概率分布写成条件概率的乘积形式：


$$
\begin{aligned}
p_{r}(x_{1},x_{2},x_{3})
&= p_{r}(x_{1}, x_{2}|x_{3})p_{r}(x_{3}) \\
&= p_{r}(x_{1}|x_{2}, x_{3})p_{r}(x_{2}|x_{3})p_{r}(x_{3}) \\
&= p_{r}(x_{1}|x_{2})p_{r}(x_{2}|x_{3})p_{r}(x_{3}) & \text{ ;Conditional Independence} \\
\end{aligned}
$$


<span style="color:GoldenRod">**Note that conditional independence relations are always symmetric**</span>


$$
\begin{aligned}
p_{r}(x_{3},x_{2},x_{1})  
&= p_{r}(x_{3}, x_{2}|x_{1})p_{r}(x_{1}) \\
&= p_{r}(x_{3}|x_{2}, x_{1})p_{r}(x_{2}|x_{1})p_{r}(x_{1}) \\
&= p_{r}(x_{3}|x_{2})p_{r}(x_{2}|x_{1})p_{r}(x_{1}) & \text{ ;Conditional Independence}\\
\end{aligned}
$$


条件独立关系意味着对条件分布以一定的方式进行因子分解(并因此视为冗余)，这种冗余意味着可用更少量的参数来描述数据的概率分布，同时对含有大规模参数的模型更加易于处理。计算机视觉中常引入图模型来表示这种条件独立关系，如有向图模型(即贝叶斯网络)，链式模型(即马尔科夫链)和树模型。

> Reference chapter-10 in the book: "Computer Vision: Models, Learning, and Inference".

### 7. 期望 Expectation

> Given a function $f[\bullet]$ that returns a value for each possible value $x^{\ast}$ of the variable $x$ and a probability $P_{r}(x=x^{\ast})$ that each value of $x$ occurs, we sometimes wish to calculate the expected output of the function. 

随机变量 $x$ 在仿射变换函数 $f[\bullet]$ 下进行变换，需要计算对应 $f[\bullet]$ 变换后的期望输出结果；可以将这个问题转化为从随机变量 $x$ 的概率分布 $P_{r}(x=x^{\ast})$ 中抽取大量样本，计算对应的仿射变换的值，并计算一系列数值的均值，该均值就是期望。


$$E\left [f\left [ x \right ]  \right ] =\sum_{x}f\left [  x\right ]  p_{r}(x)=\int_{x} f\left [  x\right ]  p_{r}(x) \mathrm{d}x $$

$$E\left [f\left [ x,y \right ]  \right ] =\sum_{x} \sum_{y} f\left [ x,y\right ] p_{r}(x,y)=\int_{x} \int_{y} f\left [ x,y\right ]  p_{r}(x,y) \mathrm{d}x \mathrm{d}y $$

$$E\left [f\left [ x,y,z \right ] \right ] =\sum_{x} \sum_{y} \sum_{z} f\left [ x,y,z\right ] p_{r}(x,y,z)=\int_{x} \int_{y} \int_{z} f\left [ x,y,z\right ]  p_{r}(x,y,z) \mathrm{d}x \mathrm{d}y \mathrm{d}z $$

$$E\left [f\left [ x_{1},x_{2}, \dots, x_{k} \right ] \right ] =\sum_{i=1}^{k} \sum_{i} f\left [ x_{1},x_{2}, \dots, x_{k}\right ] p_{r}( x_{1},x_{2}, \dots, x_{k})=\prod_{i=1}^{k} \int_{i} f\left [ x_{1},x_{2}, \dots, x_{k}\right ]  p_{r}(x_{1},x_{2}, \dots, x_{k}) \mathrm{d}i $$


有一些特殊的仿射变换函数 $f[\bullet]$，其计算后的期望有一些特殊的名称，这些特殊的名称常用于量化概括复杂概率分布的性质。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/expectation.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 6.  Special cases of expectation. (Image source from the book, "Computer Vision: Models, Learning, and Inference")</div>
</center>

### **8. 重参数技巧 Reparameterization Trick**

如果从高斯分布中随机采样一个样本，这个过程不可微分的，即无法反传梯度的。通过**重参数 (reparameterization) 技巧** &ensp;[VAE](<https://arxiv.org/abs/1312.6114> (Diederik P. Kingma, Max Welling, "Auto-Encoding Variational Bayes," ICLR'2014)) &ensp;来使其可微。最通常的做法是把这种随机性通过一个独立的随机变量 $\epsilon$ 进行转移。举个例子，如果要从高斯分布 $z\sim \mathcal{N}\left ( z;\mu_{\theta},\sigma^{2}_{\theta} I\right ) $ 中采样一个 z，可以写成:


$$ z = \mu_{\theta} + \sigma_{\theta} \odot \epsilon , \epsilon \sim \mathcal{N}\left ( 0,I\right ) $$


上式的 z 依旧是有随机性的，且满足均值为 $\mu_{\theta}$, 方差为 $\epsilon_{\theta}$ 的高斯分布。这里的 $\mu_{\theta}$ 和 $\sigma_{\theta}$ 可以由参数化神经网络推断得到的。使得采样过程依旧梯度可导，随机性被转嫁到 $\epsilon$ 上。

### **9. 高斯函数形式 Notation of Gaussian dist. or Normal distribution**

> [Normal_distribution wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)

**1. Standard normal distribution** 

The simplest case of a normal distribution is known as the standard normal distribution or unit normal distribution. This is a special case when $\mu =0$ and $\sigma =1$, and it is described by this probability density function (or density):


$$\varphi(z)=\frac{e^{-z^{2}/2}}{\sqrt{2\pi}} $$


The variable $z$ has a mean of $0$ and a variance and standard deviation of $1$. The density $\varphi(z)$ has its peak $1/{\sqrt {2\pi}}$ at $z=0$ and inflection points at $z=+1$ and $z=-1$.

**2. General normal distribution** 

Every normal distribution is a version of the standard normal distribution, whose domain has been stretched by a factor $\sigma$ (the standard deviation) and then translated by $\mu$ (the mean value):


$$ f(x\mid \mu ,\sigma^{2}) = \frac{1}{\sigma} \varphi (\frac{x-\mu}{\sigma}) $$


The probability density must be scaled by $1/\sigma$ so that the integral is still 1.

If $Z$ is a standard normal deviate, then $X=\sigma Z+\mu$ will have a normal distribution with expected value $\mu$ and standard deviation $\sigma$. This is equivalent to saying that the "standard" normal distribution $Z$ can be scaled/stretched by a factor of $\sigma$ and shifted by $\mu$ to yield a different normal distribution, called $X$. Conversely, if $X$ is a normal deviate with parameters $\mu$ and $\sigma^{2}$, then this $X$ distribution can be re-scaled and shifted via the formula $Z=(X-\mu)/\sigma$ to convert it to the "standard" normal distribution. This variate is also called the standardized form of $X$.

**3. Notation**

The probability density of the standard Gaussian distribution (standard normal distribution, with zero mean and unit variance) is often denoted with the Greek letter $\phi$. The alternative form of the Greek letter phi, $\varphi$, is also used quite often.

The normal distribution is often referred to as $N(\mu ,\sigma^{2}$ or $\mathcal{N}(\mu,\sigma^{2}$. Thus when a random variable $X$ is normally distributed with mean $\mu$ and standard deviation $\sigma$, one may write:


$$ X \sim \mathcal{N}(\mu ,\sigma^{2}).$$


**4. Alternative parameterizations**

Some authors advocate using the precision $\tau$ as the parameter defining the width of the distribution, instead of the deviation $\sigma$ or the variance $\sigma^{2}$. The precision is normally defined as the reciprocal of the variance, $1/\sigma^{2}$. The formula for the distribution then becomes:


$$
\begin{aligned}
f(x)  
&= \sqrt{\frac{\tau}{2\pi}} e^{\frac{-\tau (x - \mu)^{2}}{2}} \\
&= \sqrt{\frac{1}{2\pi \sigma^{2}}} e^{- \frac{(x - \mu)^{2}}{2 \sigma^{2}}} \\
&= \sqrt{\frac{1}{2\pi \sigma^{2}}} exp(- \frac{(x - \mu)^{2}}{2 \sigma^{2}}) \\
&\propto exp(- \frac{(x - \mu)^{2}}{2 \sigma^{2}}) \\
&\propto exp(\frac{(x - \mu)^{2}}{\sigma^{2}}) \\
&\propto \frac{x^{2} - 2 \mu x + \mu^{2}}{\sigma^{2}} \\
&= \frac{1}{\sigma^{2}}x^{2} -  \frac{2 \mu}{\sigma^{2}}x + \frac{\mu^{2}}{\sigma^{2}}\\
\end{aligned}
$$


### **10. 信息论和概率模型**

**1. 拟合概率模型**

How to Fitting or Parameterization the probability models

- **最大似然法 Maximum likelihood, ML**
- **最大后验法 Maximum a posteriori, MAP**
- <span style="color:red">**贝叶斯方法 the Bayesian approach**</span>

**maximum likelihood, ML**

最大似然 ML 用来求数据 $x_{i}$ , $[i=1, 2, 3, \cdots, I]$ 最有可能的参数集合 $\mathbf{\hat{\theta}}$ 。为了计算在单个数据点 $x_{i}$ 处的似然函数 $P_{r}(x_{i} \mid \mathbf{\theta})$ , 只需要简单估计在 $x_{i}$ 处的概率密度函数 (probability density function, pdf) 。假设每一个数据点都是从分布中独立采样，点的集合的似然函数 $P_{r}(x_{1\cdots}I \mid \mathbf{\theta})$ 就是独立似然的乘积。因此，参数的最大似然估计如下：


$$
\begin{aligned}
\mathbf{\hat{\theta}}
&=\underset{\theta}{argmax}[P_{r}(x_{1\cdots}I \mid \mathbf{\theta})] \\
&=\underset{\theta}{argmax}[\prod_{i=1}^{I} P_{r}(x_{i} \mid \mathbf{\theta})] \\
\end{aligned}
$$

其中，$\underset{\theta}{argmax} f[\theta]$ 返回使得 $f[\theta]$ 最大化的 $\theta$ 数值


为了估计新的数据点 $x^{\ast}$ 的概率分布，其中计算 $x^{\ast}$ 属于拟合模型的概率，用最大似然拟合参数 $\mathbf{\hat{\theta}}$ 简单估计概率密度函数 $P_{r}(x^{\ast} \mid \mathbf{\hat{\theta}})$ 即可。

**Maximum a posteriori, MAP**

最大后验拟合 MAP 中，引入参数 $\theta$ 的先验 (prior) 信息。 From previous experience we may know something about the possible parameter values. For example, in a time-sequence  the values of the parameters at time $t$ tell us a lot about the possible values at time $t + 1$. 而且这个先验信息可以被先验分布所编码。

最大后验估计就是最大化参数的后验概率 $P_{r}(\mathbf{\theta} \mid x_{1 \cdots I})$


$$
\begin{aligned}
\mathbf{\hat{\theta}}
&=\underset{\theta}{argmax}[P_{r}(\mathbf{\theta} \mid x_{1 \cdots I})] \\
&=\underset{\theta}{argmax}[\frac{P_{r}(x_{1 \cdots I} \mid \mathbf{\theta}) P_{r}(\mathbf{\theta})}{P_{r}(x_{1 \cdots I})}] \\
&=\underset{\theta}{argmax}[\frac{\prod_{i=1}^{I} P_{r}(x_{i} \mid \mathbf{\theta}) P_{r}(\mathbf{\theta})}{P_{r}(x_{i \cdots I})}] \\
\end{aligned}
$$


其中，对前两行和随后的假设的独立性之间使用贝叶斯公式；实际上，可以忽略对于参数而言是常数项的分母 (即与参数 $\theta$ 无关) ，这样并不会影响最大值的位置，简化为：


$$\mathbf{\hat{\theta}} = \underset{\theta}{argmax} [\prod_{i=1}^{I} P_{r}(x_{i} \mid \mathbf{\theta}) P_{r}(\mathbf{\theta})]$$


将该式子与最大似然估计对比可知，除了先验部分之外完全一致；ML 是 MAP 在先验信息未知情况下的一个特例。对于概率密度 ($x^{\ast}$ 在拟合模型下的概率) 则可以通过新参数估计概率密度函数 $P_{r}(x^{\ast} \mid \mathbf{\hat{\theta}})$ 进行计算。

**Bayerian approach**

使用贝叶斯方法，不再试图估计具有单个数据点的参数 $\mathbf{\theta}$，即点估计方法；而且承认一个明显的事实：参数 $\theta$ 可能有多个与数据兼容的值。 (In the Bayesian approach we stop trying to estimate single fixed values (point estimates)
of the parameters θ and admit what is obvious; there may be many values of the parameters that are compatible with the data. ) 

使用贝叶斯公式在数据 $\{ x_{i} \}_{i=1}^{I}$ 上计算参数 $\mathbf{\theta}$ 的概率分布：


$$P_{r}(\mathbf{\theta} \mid x_{1 \cdots I})= \frac{\prod_{i=1}^{I} P_{r}(x_{r} \mid \mathbf{\theta}) P_{r}(\mathbf{\theta})}{P_{r}(x_{i \cdots I})}$$


估计预测分布对于贝叶斯的情况更加困难 (computationally untractable in DPM paper), 因为没有估计单一模型，而是通过概率模型来拟合一个概率分布；因此，计算：

$$P_{r}(x^{\ast} \mid x_{1 \cdots I}) = \int P_{r}(x^{\ast} \mid \mathbf{\theta}) P_{r}(\mathbf{\theta} \mid x_{i \cdots I}) \mathrm{d}\theta$$

可以这样理解: $P_{r}(x^{\ast} \mid \mathbf{\theta})$ 是一个给定 $\mathbf{\theta})$ 的预测；所以积分 (integral) 可以当作由不同参数 $\mathbf{\theta})$ 确定的预测的加权和，这里的加权系数由参数 $\theta$ 的后验概率分布 $P_{r}(\mathbf{\theta} \mid x_{i \cdots I})$ 所决定 (表示确保不同的参数值都是正确的)。

如果用 ML 和 MAP 估计密度都确定为 $\mathbf{\hat{\theta}}$ 前提的特殊概率分布，那么 ML、MAP、and Bayerian 中预测概率密度的估计可以统一起来。更加一般形式化，将三者当作中心在 $\mathbf{\hat{\theta}}$ 处的贝塔函数即可 (delta functions centered at $\mathbf{\hat{\theta}}$) ，设 $\delta[z]$ 是一个积分为 1， 而且除了 $z=0$ 处之外都为 $0$ 的函数 (信息与系统处理里面的冲激函数或脉冲函数)，那么则有如下式子：


$$
\begin{aligned}
P_{r}(x^{\ast} \mid x_{1 \cdots I}) 
&= \int P_{r}(x^{\ast} \mid \mathbf{\theta}) \delta[\mathbf{\theta} - \mathbf{\hat{\theta}}] \mathrm{d}\theta \\
&= P_{r}(x^{\ast} \mid \mathbf{\theta}) \\
\end{aligned}
$$

where $\int \delta[\mathbf{\theta} - \mathbf{\hat{\theta}}] \mathrm{d}\theta = 1$, which is exactly the calculation we originally prescribed:  we simply evaluate the probability of the data under the  model with the estimated parameters. 可以估计数据在参数模型下的概率。

**2. 信息熵，交叉熵和KL散度**

**信息论与编码**

> Information entropy; Entropy; Shannon entropy; Cross entropy; Relative entropy; Kullback–Leibler divergence ([KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence))

- 信息量：指用一个信息所需要的编码长度来定义，而一个信息的编码长度与其出现的概率呈负相关。

> 其实也就可以说一件事情发生概率很小的话，那么当其发生的时候所代表的信息量也更大


$$I = log_{2}(\frac{1}{p(x)}) = -log_{2}(p(x))$$


- 信息熵 (香农熵)：信息熵则代表一个分布的信息量,或者编码的平均长度。

> 信息熵度量的是随机变量 (<span style="color:green">**离散和连续**</span>) 或整个系统的不确定性，熵越大，随机变量或系统的不确定性就越大，也就是信息量的一个总期望值也叫均值；根据真实分布，能够找到一个最优策略，以最小的代价消除系统的不确定性，而这个代价大小就是信息熵；信息熵衡量了系统的不确定性，而要消除这个不确定性，所要付出的最小努力(猜题次数、编码长度等)的大小就是信息熵。


$$
\begin{aligned}
H(p) 
&= \sum_{x}p(x) log_{2}(\frac{1}{p(x)}) \\
&= -\sum_{x}p(x) log_{2}(p(x)) \\
&= E_{x \sim p(x)}[-log_{2}p(x)] \\
&= -\sum_{i=1}^{n} p(x) log_{2}(p(x)) \\
\end{aligned}
$$


$$
\begin{aligned}
H(p) 
&= \int p(x) log_{2}(\frac{1}{p(x)}) \mathrm{d}x \\
&= -\int p(x) log_{2}(p(x)) \mathrm{d}x \\
&= E_{x \sim p(x)}[-log_{2}p(x)] \\
&= -\int_{-\infty}^{\infty} p(x) log_{2}(p(x)) \\
\end{aligned}
$$


在信息论和编码中 log 的底数是 2，但一般在神经网络中，默认以 e (<span style="color:DeepPink">e = 2.73 magic number, such as 42 </span>) 为底，这样算出来的香农信息量虽然不是最小的可用于完整表示实践的比特数，但对于信息熵的含义来说是区别不大的，只要这个底数大于 1，就可以表达信息熵的大小。

- 交叉熵: 本质上可以视为用一个猜测(预估)的分布的编码方式去编码其真实的分布, 得到的平均编码长度或者信息量。

> 交叉熵，用来度量在给定的真实分布 $p$ 下，使用非真实分布 $q$ 所指定的策略消除系统的不确定性所需要付出的努力的大小; 交叉熵越低，这个策略就越好，最低的交叉熵也就是使用了真实分布所计算出来的信息熵，故此 “交叉熵 = 信息熵” ；这也是为什么在机器学习中的分类算法中，总是最小化交叉熵，因为交叉熵越低，就证明由算法所产生的策略最接近最优策略，也间接证明算法所算出的非真实分布越接近真实分布。


$$
\begin{aligned}
H_p(q) 
&= \sum_{x}p(x)log(\frac{1}{q(x)}) \\
&= - \sum_{x}p(x)log(q(x)) \\
&= - \sum_{i=1}^{n} p(x_{i})log(q(x_{i})) \\
&= - E_{x \sim p(x)} log(q(x)) & \text{; 离散和连续}\\
&= - \int_{-\infty}^{\infty} p(x)log(q(x)) \mathrm{d}x \\
\end{aligned}
$$


> 相对熵 (KL 散度)：KL散度或距离是度量两个分布的差异，KL 距离一般用 $D(p||q)$ 或 $D_{p}(q)$ 称之为 $p$ 对 $q$ 的相对熵。


$$
\begin{aligned}
D(p || q) = D_p(q) 
&= H_{p}(q) - H(P) & \text{; cross entropy minus information entropy} \\
&= \sum_{x}p(x)log(\frac{1}{q(x)}) - [- \sum_{i=1}^{n} p(x) log(p(x))] \\
&= \sum_{x}p(x)log(\frac{1}{q(x)}) + \sum_{i=1}^{n} p(x) log(p(x)) \\
&= \sum_{x}p(x) \log{\frac{p(x)}{q(x)}} \\
&= E_{x \sim p(x)} [\log{\frac{p(x)}{q(x)}}] \\
&= D_{KL}(p || q) \\
\end{aligned}
$$


在 $p$ and $q$ 满足可交换的条件下，交叉熵和 KL 散度相等。还有联合信息熵；条件信息熵；自信息；互信息等针对不同用途的度量形式。

> [Difference of KL divergence and cross entropy](https://stats.stackexchange.com/questions/357963/what-is-the-difference-cross-entropy-and-kl-divergence)

利用 詹森不等式 ([Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)) 可以推导出 KL 散度的非负性：

> 对数的期望大于等于期望的对数 $\Phi(E[X]) \le E[\Phi(X)]$


$$
\begin{aligned}
D_{KL}(p || q)
&= \sum_{x}p(x) \log (\frac{p(x)}{q(x)}) \\
&= \int_{-\infty}^{\infty} p(x)\mathrm{d}x \log(\frac{q(x)}{p(x)}) \\
&= E_{x \sim p(x)}[log(\frac{q(x)}{p(x)})] \\
&\ge \log E_{x \sim p(x)}[\frac{q(x)}{p(x)}] \\
&= \log \sum_{x} p(x) \frac{q(x)}{p(x)} \\
&= \log \sum_{x} q(x) \\
&= \log(1) \\
&= 0 \\
\end{aligned}
$$

> KL-divergence between two gaussians is tractable, having closed-form formula. Let’s consider the case of single variable Gaussians:

$$
\begin{aligned}
& D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) || \mathcal{N}(\mu_2, \sigma_2^2)) \\

= & \int dx \left[\log \mathcal{N}(\mu_1, \sigma_1^2) - \log \mathcal{N}(\mu_2, \sigma_2^2)\right] \mathcal{N}(\mu_1, \sigma_1^2) \\

= & \int dx  \left[ -\frac{1}{2} \log(2\pi) - \log \sigma_1 - \frac{1}{2} \left(\frac{x - \mu_1}{\sigma_1} \right)^2 \right. \\
&\left. ~~~~~~~~~~~~+ \frac{1}{2} \log(2\pi) + \log \sigma_2 + \frac{1}{2} \left(\frac{x - \mu_2}{\sigma_2}\right)^2 \right] \\
&~~~~~~~~~~~\times\frac{1}{\sqrt{2\pi\sigma_1}} \exp \left[ -\frac{1}{2}\left( \frac{x - \mu_1}{\sigma} \right)^2 \right] \\

= & \mathbb{E}_{1}  \left[ \log \frac{\sigma_2}{\sigma_1} + \frac{1}{2} \left[ \left(\frac{x - \mu_2}{\sigma_2} \right)^2 - \left(\frac{x - \mu_1}{\sigma_1}\right)^2 \right] \right ] \\

= & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} \mathbb{E}_1 [({x - \mu_2})^2] - \frac{1}{2\color{green}\sigma_1^2} \color{green}\mathbb{E}_1 [({x - \mu_1})^2] \\

= & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} \mathbb{E}_1 [({x - \mu_2})^2] -  \frac{1}{2} \\

= & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} \mathbb{E}_1 [({x - \mu_1 + \mu_1 - \mu_2})^2] -  \frac{1}{2} \\

= & \log\frac{\sigma_2}{\sigma_1} + \frac{1}{2\sigma_2^2} {\color{green}\mathbb{E}_1 [(x - \mu_1)^2} + 2(x-\mu_1)(\mu_1 - \mu_2) + (\mu_1 - \mu_2)^2] -  \frac{1}{2} \\

= & \log\frac{\sigma_2}{\sigma_1} + \frac{{\color{green}\sigma_1^2} + (\mu_1 - \mu_2)^2}{2\sigma_2^2} -  \frac{1}{2} 
\end{aligned}
$$


> More generally for multivariate Gaussians with dimension $d$:

$$
\begin{aligned}
& D_{\text{KL}}(\mathcal{N}(\mu_1, \Sigma_1) || \mathcal{N}(\mu_2, \Sigma_2)) \\
= & \frac{1}{2} \left[\log\frac{|\Sigma_2|}{|\Sigma_1|} -d + \mathrm{tr}\{\Sigma_2^{-1}\Sigma_1\}  + (\mu_2 - \mu_1)\Sigma_2^{-1}(\mu_2 - \mu_1) \right]
\end{aligned}
$$

-------------------------------------

### Diffusion Process (DPM-ICML'2015 & DDPM-NeurIPS'2020 & IDDPM-ICML'2021)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/DDPM.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 7. The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source from DDPM'2020)</div>
</center>
<!-- ![generative-overview](./Images/DDPM.png) -->

<center class="half">
    <img src="./images/forward_diffusion_s_curve.gif", width="50%" /><img src="./images/reverse_diffusion_s_curve.gif", width="50%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 8. The S curve distribution from forward(left) to reverse(right) diffusion process.</div>
</center>

<center class="half">
    <img src="./images/forward_diffusion_swiss_roll.gif", width="50%" /><img src="./images/reverse_diffusion_swiss_roll.gif", width="50%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 9. The two dimensions swiss roll distribution from forward(lefet) to reverse(right) diffusion process.</div>
</center>

#### **Forward Diffusion Process**

1. forward diffusion process

<center>
    <img src="./images/forward_process_diffusion.png" />
</center>

original data distribution: $x_{0} \sim q(x)$


$$q(x_{t} \mid x_{t-1}) \sim \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t-1}, \beta_{t}I)$$


where $\beta_{t}$ denotes diffusion rate in DPM paper;  denotes variance(noise) schedule in DDPM and IDDPM papers; and then the add noise $\beta_{t} \in (0, 1)$ , ${t=1, 2, 3, \cdots, T}$ , $\beta_{1} < \beta_{2}<\beta_{3}\cdots<\beta_{T}$ ; and then can proof the reverse process is still gaussian disribution.

> 有数学理论证明，高斯前向扩散过程的噪声 $\beta_{t}$ 足够小，那么逆向扩散过程也可以视为一个高斯分布。

and the joint probability formulation as following:


$$q(x_{1:T} \mid x_{0}) = \prod_{t=1}^{T} q(x_{t} \mid x_{t-1})$$


theoretically, $T \to \infty$ , $x_{T} \to \mathcal{N}(0, I)$ ,the Isotropic Gaussian distribution.

2. forward add noise

> 不是简单的对噪声进行线性叠加，而是使用<span style="color:Crimson">**重参数技巧(Reparameterization Trick)**</span>进行仿射变换的方式添加噪声。

如果要从高斯分布 $z\sim \mathcal{N}\left ( z;\mu_{\theta},\sigma^{2}_{\theta} I\right ) $ 中采样一个 $\bar{z}$，可以写成:


$$ \bar{z} = \mu_{\theta} + \sigma_{\theta} \odot \epsilon , \epsilon \sim \mathcal{N}\left ( 0,I\right ) $$

if $x_{0}$, and $z\sim \mathcal{N}(z; \mu_{\theta}, \sigma_{\theta}^{2} I) $ , and forward diffusion process $q(x_{t} \mid x_{t-1}) \sim \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t-1}, \beta_{t}I)$ ,

and then The step-wise noise addition gradually process of the forward diffusion can be defined as follow:

$$x_{1} = \sqrt{\beta_{1}}z_{1} + \sqrt{1 - \beta_{1}}x_{0}$$

$$x_{2} = \sqrt{\beta_{2}}z_{2} + \sqrt{1 - \beta_{2}}x_{1}$$

$$x_{3} = \sqrt{\beta_{3}}z_{3} + \sqrt{1 - \beta_{3}}x_{2}$$

$$\cdots$$

$$x_{T} = \sqrt{\beta_{T}}z_{T} + \sqrt{1 - \beta_{T}}x_{T-1}$$


这样不断的对原始数据分布进行添加扰动，打乱有规矩的数据概率分布，也就是热力学中的熵增过程，不断变得混乱，最终趋近一个拥有良好性质而且解析上易于处理的分布。 ("**The data distribution is gradually converted into a well behaved (analytically tractable) distribution $π(y)$ by repeated application
of a Markov diffusion kernel.**" quote from DPM paper ICML'2015.)

3. **任意时刻的 $x_{t}$ 可以由 $x_{0}$ 和 $\beta_{t}$ 直接计算得到**

前向扩散过程中有一个良好的性质，就是任意时刻的 $x_{t}$ 可以由 $x_{0}$ 和 $\beta_{t}$ 直接计算得到采样；利用重参数技巧(Reparameterization Trick) 可以得到下面式子：


<!-- $$ x_{t} = \sqrt{\beta_{t}}z + \sqrt{1-\beta_{t}}x_{t-1} ; z \in \mathcal{N}(0, I) \tag{1}$$  -->
$$ x_{t} = \sqrt{\beta_{t}}z + \sqrt{1-\beta_{t}}x_{t-1} ; z \in \mathcal{N}(0, I) $$ 


令 $ \alpha_{t} = 1 - \beta_{t}$ , 则上式子可以化简为：


$$
\begin{aligned}
x_{t}
&= \sqrt{ \alpha_{t} } x_{t-1} + \sqrt{1 - \alpha_{t}} z_{1} \\
&= \sqrt{\alpha_{t}} (\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}z_{2}) + \sqrt{1 - \alpha_{t}} z_{1} & \text{;不断利用重参数技巧进行采样} \\
&= \sqrt{\alpha_{t} \alpha_{t-1}}x_{t-2} + {\color{red} \sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2}} + {\color{blue} \sqrt{1 - \alpha_{t}} z_{1}} \\
\end{aligned}
$$


对上式子中的后面两项进行处理，根据高斯分布的性质：


$$\sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2} \sim \mathcal{N}(0, \alpha_{t}(1- \alpha_{t-1})I) ; z_{2} \in \mathcal{N}(0, I)$$


$$\sqrt{1 - \alpha_{t}} z_{1} \sim \mathcal{N}(0, (1- \alpha_{t})I) ; z_{1} \in \mathcal{N}(0, I)$$


$$ X \sim \mathcal{N}(\mu_{1}, \sigma_{1}^{2}); $$


$$ Y \sim \mathcal{N}(\mu_{2}, \sigma_{2}^{2}); $$


$$ aX + bY \sim \mathcal{N}(a\mu_{1} + b\mu_{2}, a^{2}\sigma_{1}^{2} + b^{2}\sigma_{2}^{2}); $$


$$ \sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2} + \sqrt{1 - \alpha_{t}} z_{1} \sim \mathcal{N}(0, (\alpha_{t}(1- \alpha_{t-1}) + (1- \alpha_{t}))I); $$


$$ \sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2} + \sqrt{1 - \alpha_{t}} z_{1} \sim \mathcal{N}(0, (\alpha_{t}- \alpha_{t}\alpha_{t-1} + 1- \alpha_{t})I); $$


$$ \sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2} + \sqrt{1 - \alpha_{t}} z_{1} \sim \mathcal{N}(0, (1- \alpha_{t}\alpha_{t-1})I); $$



$$
\begin{aligned}
则原式
&= x_{t}\\
&= \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + {\color{red}\sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2}} + {\color{blue}\sqrt{1 - \alpha_{t}} z_{1}} \\
&= \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + \color{green}\sqrt{1- \alpha_{t}\alpha_{t-1}}\bar{z_{2}} \\
&= \cdots & \text{;相同的方式不断迭代}\\
&= \sqrt{\alpha_{t}\alpha_{t-2} \cdots \alpha_{t=T-1}}x_{1} + \color{green}\sqrt{1- \alpha_{t}\alpha_{t-1} \cdots \alpha_{t=T-1}}\bar{z_{T}} \\
&= \sqrt{\alpha_{t}\alpha_{t-1} \cdots \alpha_{t=T}}x_{0} + \color{green}\sqrt{1- \alpha_{t}\alpha_{t-1} \cdots \alpha_{t=T}}\bar{z_{T+1}} \\
&= \sqrt{\prod_{i=1}^{T} \alpha_{i}}x_{0} + \color{green}\sqrt{1- \prod_{i=1}^{T} \alpha_{i}}\bar{z_{T+1}} \\
\end{aligned}
$$


令 $ \bar{\alpha}_{t} = \prod_{i=1}^{T} \alpha_{i}$ , 则上式子可以化简为：


$$x_{t} = \sqrt{\bar{\alpha}_{t}}x_{0} + \color{green}\sqrt{1 - \bar{\alpha}_{t}}\bar{z}_{t} ; \color{red}\bar{z}_{t} \in \mathcal{N}(0, I)$$


$$q(x_{t} \mid x_{0}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$


故此总结一下前向扩散过程的概率分布满足一下式子：


$$
\begin{aligned}
q(x_{t} \mid x_{t-1})
&\sim \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t-1}, \beta_{t}I) \\
&= \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t-1}, \beta_{t}I) \bar{z}_{t} \\
\end{aligned}
$$


$$
\begin{aligned}
q(x_{t} \mid x_{0})
&\sim \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I) \\
&= \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I) \bar{z}_{t} \\
\end{aligned}
$$



这样根据前向扩散过程的要求，最终迭代 $T$ 次后，$x_{T}$ 变成一个标准高斯分布，则可以计算出迭代次数 $T$ 的具体数值(e.g. T=1000)：



$$
sub. to \left\{
\begin{aligned}
&\sqrt{\bar{\alpha_{t}}}x_{0} = 0 \\
&1 - \bar{\alpha_{t}} = 1 \\
\end{aligned}
\right.
$$



因此前向扩散过程中的迭代步数是有一个有限的可解析的数值，$t \in (0, T) $ 进行采样得到具体的数值，the sample-step schedule is different, PDM and DDPM paper is uniform schedule(均匀采样); but IDDPM paper is simple importance sampling technique(基于 loss 进行重要性采样)

> <span style="color:DarkOrange"> Note: Imporved diffusion Code Implementation with OpenAI. That is Awesome repo. </span>

#### **Reverse Diffusion Process**

<center>
    <img src="./images/reverse_process_diffusion.png" />
</center>

如果说前向扩散过程 (forward process)是加噪的过程，那么逆向扩散过程 (reverse process) 就是 diffusion models 的去噪推断过程。如果能够逐步得到逆转后的分布，就可以从完全的标准高斯分布  还采样从而复原出原始分布。 Feller William 在 1949 年的文献中证明了如果 forward process $q(x_{t} \mid x_{t-1})$ 满足高斯分布且 $\beta_{t}$ 足够小，reverse process $q(x_{t-1} \mid x_{t})$ 仍然是一个高斯分布。然而这个逆向分布无法进行简单推断计算出解析式，因此使用深度学习模型 (Neural Networks, NN) 去预测或者拟合这样的一个逆向的分布。

因此假设逆向过程的分布 $q(x_{t-1} \mid x_{t}) \sim \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))$ , 利用 NN 拟合 $\mu_{\theta}$ 和 $\Sigma_{\theta}$ , 均值和方差都是关于 $(x_{t}, t)$ 的仿射变换函数



$$ q(x_{t-1} \mid x_{t}) = p_{\theta}(x_{t-1} \mid x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t)) $$


and the joint probability dist. as follow:


$$ p_{\theta}(X_{0:T}) = p(X_{T}) \prod_{t=1}^{T}(x_{t-1} \mid x_{t}) $$


> <span style="color:green">虽然无法计算出 $q(x_{t-1} \mid x_{t})$ ,但是可以计算出逆向扩散过程的后验概率分布 $q(x_{t-1} \mid x_{t}, x_{0})$ . </span> 联合概率分布可以分解为条件概率分布的乘积形式



$$
\begin{aligned}
q(x_{t-1}, x_{t}, x_{0})
&= q(x_{t-1}, x_{t} \mid x_{0})q(x_{0}) \\
&= q(x_{t-1} \mid x_{t}, x_{0})q(x_{t} \mid x_{0})q(x_{0}) \\
\end{aligned}
$$



基于 diffusion process (forward or reverse) 都是马尔可夫过程 (Markov chain) ，在给定 $x_{0}$ 条件下，$x_{t-1}$ 和 $x_{t}$ 条件独立，则利用对称性，$q(x_{t-1}, x_{t}, x_{0})$ 联合概率分布有如下相同等式



$$
\begin{aligned}
q(x_{t-1}, x_{t}, x_{0})
&= q(x_{t}, x_{t-1} \mid x_{0})q(x_{0}) \\
&= q(x_{t} \mid x_{t-1}, x_{0})q(x_{t-1} \mid x_{0})q(x_{0}) \\
\end{aligned}
$$


那么逆向扩散过程的后验概率分布如下推导：


$$
\begin{aligned}
q(x_{t-1} \mid x_{t}, x_{0})
&=\frac{q(x_{t-1}, x_{t}, x_{0})}{q(x_{t} \mid x_{0}) q(x_{0})} \\
&=\frac{q(x_{t} \mid x_{t-1}, x_{0})q(x_{t-1} \mid x_{0})q(x_{0})}{q(x_{t} \mid x_{0}) q(x_{0})} \\
&=\frac{q(x_{t} \mid x_{t-1}, x_{0})q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} \\
&=q(x_{t} \mid x_{t-1}, x_{0}) \frac{q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} \\
&=\color{red} q(x_{t} \mid x_{t-1}) \frac{q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} & \text{; Markov chain}\\
\end{aligned} \\
$$

$$
\begin{aligned}
q(x_{t} \mid x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}}x_{t-1}, \beta_{t}I)
\end{aligned}
$$

$$
\begin{aligned}
q(x_{t} \mid x_{0}) = \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I)
\end{aligned}
$$

$$
\begin{aligned}
q(x_{t-1} \mid x_{0}) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_{0}, (1 - \bar{\alpha}_{t-1})I)
\end{aligned}
$$


将高斯前向扩散过程带入后验分布式子中，可以化简如下：


$$
\begin{aligned}
q(x_{t-1} \mid x_{t}, x_{0})

&=\color{red} q(x_{t} \mid x_{t-1}) \frac{q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} \\

&\sim \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}}x_{t-1}, \beta_{t}I) \frac{\mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I)}{\mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_{0}, (1 - \bar{\alpha}_{t-1})I)} \\

&\propto exp(-\frac{1}{2}[\color{green} \frac{(x_{t} - \sqrt{1-\beta_{t}}x_{t-1})^{2}}{\beta_{t}} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_{0})^{2}}{1-\bar{\alpha}_{t-1}} - \frac{(x_{t} - \sqrt{\bar{\alpha}_{t}}x_{0})^{2}}{1-\bar{\alpha}_{t}}]) \\ 

&\propto exp(\frac{x_{t}^{2} - 2\sqrt{1-\beta_{t}}x_{t-1}x_{t} + (1-\beta_{t})(x_{t-1})^{2}}{\beta_{t}} + \frac{x_{t-1}^{2} - 2\sqrt{\bar{\alpha}_{t-1}}x_{0}x_{t-1} + \bar{\alpha}_{t-1}(x_{0})^{2}}{1-\bar{\alpha}_{t-1}} - \frac{x_{t}^{2} - 2\sqrt{\bar{\alpha}_{t}}x_{0}x_{t} + \bar{\alpha}_{t}(x_{0})^{2}}{1-\bar{\alpha}_{t}}) \\

&\propto \frac{x_{t}^{2} - 2\sqrt{1-\beta_{t}}x_{t-1}x_{t} + (1-\beta_{t})(x_{t-1})^{2}}{\beta_{t}} + \frac{x_{t-1}^{2} - 2\sqrt{\bar{\alpha}_{t-1}}x_{0}x_{t-1} + \bar{\alpha}_{t-1}(x_{0})^{2}}{1-\bar{\alpha}_{t-1}} - \frac{x_{t}^{2} - 2\sqrt{\bar{\alpha}_{t}}x_{0}x_{t} + \bar{\alpha}_{t}(x_{0})^{2}}{1-\bar{\alpha}_{t}} \\

&\propto (\frac{1-\beta_{t}}{\beta_{t}} + \frac{1}{1-\bar{\alpha}_{t-1}})x_{t-1}^{2} - (\frac{2\sqrt{1-\beta_{t}}x_{t}}{\beta_{t}} + \frac{2\sqrt{\bar{\alpha}_{t-1}}x_{0}}{1-\bar{\alpha}_{t-1}})x_{t-1} + C(x_{t}, x_{0}) \\
\end{aligned} \\
$$

> from line 1 to line 2: 相同底数的幂函数相乘，指数相加即可; 将高斯函数写成指数表示的形式 <br> from line 2 to line 3: 将分子平方展开 <br> from line 3 to line 4: 以 $x_{t-1}$ 为变量进行合并 <br> from line 4 to line 5: 其中 $C$ 与 $x_{t-1}$ 无关的常量 <br>


逆向扩散过程的后验概率分布依然满足高斯分布，假设服从以下分布：



$$
\begin{aligned}
q(x_{t-1} \mid x_{t}, x_{0})
&\sim \mathcal{N}(x_{t-1}; \widetilde{\mu}(x_{t}, x_{0}), \widetilde{\beta}_{t}I) \\
&\sim exp(x_{t-1}; \widetilde{\mu}(x_{t}, x_{0}), \widetilde{\beta}_{t}I) \\
&\propto \color{Aquamarine} (\frac{1}{\widetilde{\beta}_{t}})x_{t-1}^{2} - (\frac{2\widetilde{\mu}}{\widetilde{\beta}_{t}})x_{t-1} + \frac{\widetilde{\mu}^{2}}{\widetilde{\beta}_{t}} \\
\end{aligned} \\
$$



根据以上关于 $q(x_{t-1} \mid x_{t}, x_{0})$ 的两个式子，可以计算出逆向扩散过程中的真实的均值和方差估计 (用于训练 NN 的监督 GT)：



$$
\begin{aligned}
\frac{1}{\widetilde{\beta}_{t}}
&= \frac{1-\beta_{t}}{\beta_{t}} + \frac{1}{1-\bar{\alpha}_{t-1}} \\
&= \frac{(1-\beta_{t})(1-\bar{\alpha}_{t-1})+\beta_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})} \\

\Rightarrow \widetilde{\beta}_{t} &= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{(1-\beta_{t})(1-\bar{\alpha}_{t-1})+\beta_{t}} \\
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}(1-\bar{\alpha}_{t-1})+\beta_{t}} & \text{; $\beta_{t}=1-\alpha_{t}$} \\ 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}-\alpha_{t}\bar{\alpha}_{t-1}+\beta_{t}} \\ 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}-\bar{\alpha}_{t}+\beta_{t}} \\ 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}-\bar{\alpha}_{t}+(1-\alpha_{t})} \\ 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} \\ 
&= \color{red} \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} & \text{; DDPM paper} \\ 
\end{aligned} \\
$$



$$
\begin{aligned}
\frac{2\widetilde{\mu}}{\widetilde{\beta}_{t}}
&= \frac{2\sqrt{1-\beta_{t}}x_{t}}{\beta_{t}} + \frac{2\sqrt{\bar{\alpha}_{t-1}}x_{0}}{1-\bar{\alpha}_{t-1}} \\
&= 2\frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}x_{t} + \beta_{t}\sqrt{\bar{\alpha}_{t-1}}x_{0}}{\beta_{t}(1-\bar{\alpha}_{t-1})} \\

\Rightarrow \widetilde{\mu}_{t}(x_{t},x_{0}) &= \frac{((1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}x_{t} + \beta_{t}\sqrt{\bar{\alpha}_{t-1}}x_{0})\widetilde{\beta}_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})} \\

&= \frac{\beta_{t}\sqrt{\bar{\alpha}_{t-1}}\widetilde{\beta}_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{0} +  \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}\widetilde{\beta}_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{t}\\

&= \frac{\beta_{t}\sqrt{\bar{\alpha}_{t-1}}(\color{red} \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t})}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{0} +  \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}(\color{red} \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t})}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{t}\\

&= \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} x_{0} + \frac{\sqrt{1-\beta_{t}}}{\beta_{t}} \cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} x_{t} & \text{; $\alpha_{t}=1-\beta_{t}$} \\

&= \color{green} \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}x_{0} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} & \text{; DDPM paper} \\
\end{aligned}  \\
$$


前向扩散过程中，任意时刻的 $x_{t}$ 与 $x_{0}$ 关系如下：


$$q(x_{t} \mid x_{0})
= \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I) \bar{z}_{t} $$


$$
\begin{aligned}
x_{t} &= \sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t} \\

\Rightarrow x_{0} &= \frac{1}{\sqrt{\bar{\alpha}_{t}}}(x_{t} - \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t})
\end{aligned}  \\
$$


将该关于 $x_{0}$ 的式子代入上式关于均值 $\widetilde{\mu}_{t}(x_{t},x_{0})$ 中可以推导如下：


$$
\begin{aligned}
\widetilde{\mu}_{t}(x_{t},x_{0}) 
&= \color{green} \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}x_{0} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} \\

&= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}(\frac{1}{\sqrt{\bar{\alpha}_{t}}}(x_{t} - \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t})) + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} \\

&= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}} \cdot \frac{1}{\sqrt{\bar{\alpha}_{t}}}(x_{t} - \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t}) + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} \\

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}x_{t} + \frac{\alpha_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}\bar{z}_{t}\right] & \text{; $\sqrt{\bar{\alpha}_{t}} = \sqrt{\alpha_{t}\bar{\alpha}_{t-1}}$}\\

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}x_{t} + \frac{\alpha_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}\bar{z}_{t}\right] \\

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{\beta_{t} + \alpha_{t} - \alpha_{t}\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}x_{t} - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}\bar{z}_{t}\right] \\

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{(1-\alpha_{t}) + \alpha_{t} - \alpha_{t}\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}x_{t} - \frac{\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})}\bar{z}_{t}\right] \\

&= \color{Cyan} \frac{1}{\sqrt{\alpha_{t}}} (x_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\bar{z}_{t} ) \\
\end{aligned}  \\
$$


**Inference Phase of DDPM**


$$p_{\theta}(x_{t-1} \mid x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t)) $$


根据该式子，可以理解 DDPM paper 的核心思想，训练 NN 网络去预测 $\bar{z}_{t}$ , 用于去噪 (denoising DPM), NN 网络预测的结果为 ${z}_{\theta}(x_{t}, t)$ , 则采样时候的均值可以直接计算得到如下(DDPM paper 中的损失函数为 $\mathcal{L}_{simple}(\theta)$)：


$$\mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{\alpha_{t}}} (x_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}{z}_{\theta}(x_{t}, t) )$$


DDPM paper 中对于方差的策略，直接使用逆向扩散过程推导的解析结果 $\widetilde{\beta}_{t}$ 或者 $\beta_{t}$ , 而且实验结果显示使用前向过程的方差数值和使用逆向过程的后验方差数值，最终的实验结果近视；不需要训练的策略，如下式子：


$$
\begin{aligned}
\Sigma_{\theta}(x_{t}, t) 
&= \widetilde{\beta}_{t} & \text{; reverse process variance} \\
&= \beta_{t} & \text{; forward process variance} \\
&= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} & \text{; reverse process posterior variance} \\
\end{aligned}
$$


IDDPM paper 中对方差策略进行了修正，可以通过 NN 神经网络来学习方差 $\Sigma_{\theta}(x_{t}, t) $ , 根据 VAE 进行推导，可以优化 variational lower bound (VLB) 从而引导方差进行学习；不过方差的可信范围非常小，导致即使在对数域，NN 也很难学习，因此 IDDPM paper 中使用参数化(模型输出的一个向量 $v$) $\beta_{t}$ 和 $\widetilde{\beta}_{t}$ 来进行学习和优化; 故此 Improved DDPM 中的损失函数为：


$$\mathcal{L}_{hybrid} = \mathcal{L}_{simple} + \lambda \mathcal{L}_{vlb}$$


#### **Optimize Diffusion Models**

> 主要推导 DPM、DDPM、Improved DDPM 原始论文中的优化目标

1. 拟合概率模型
2. 交叉熵和 KL 散度
3. optimization objective in DDPMs

**DDPM 和 VAE 的区别**

- DDPM 的后验过程是无参的，而 VAE 是通过网络参数进行拟合的
- DDPM 的推断阶段与 $X$ 无关，而 VAE 中的隐变量 $Z$ 与 $X$ 有关系
- DDPM $dim(X)$ == $dim(Z)$，而 VAE 中的维度一般都是不一致的

**3. optimization objective for DDPMs**

DDPM 的优化目标就是训练 NN 神经网络拟合估计出 $\mu_{\theta}(x_{t},t)$ and $\Sigma_{\theta}(x_{t}, t)$ ; 通过真实数据分布下，最大化模型 NN 预测分布的对数似然即可估计出靠谱的均值和方差；即优化在 $x_{0} \sim q(x_{0})$ 下 $p_{\theta}(x_{0}) $ 的交叉熵：


$$p_{\theta}(x_{t-1} \mid x_{t})  = \mathcal{N}(x_{t-1};\mu_{\theta}(x_{t},t), \Sigma_{\theta}(x_{t}, t)=\beta_{t}I)$$


$$\mathcal{L} = E_{x \sim q(x_{0})}[-\log{p_{\theta}(x_{0})}]$$



> Note: "Training is performed by optimizing the usual variational bound on negative log likelihood," quote from DDPM paper; 

> "The combination of $q$ and $p$ is a variational auto-encoder (Kingma & Welling, 2013), and we can write the variational lower bound (VLB)," quote from Improved DDPM paper.


$$
\begin{aligned}
\mathcal{L}
&= -\log{p_{\theta}(x_{0})} \\
&\le -\log{p_{\theta}(x_{0})} + D_{KL}(q(x_{1:T} \mid x_{0}) || p_{\theta}(x_{1:T} \mid x_{0})) & \text{; No.2} \\
&= -\log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{1:T} \mid x_{0})}}] & \text{; No.3} \\
&= -\log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T}) / p_{\theta}(x_{0})}}] & \text{; No.4} \\
&= -\log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} + \log{p_{\theta}(x_{0})}] & \text{; No.5} \\
&= -\log{p_{\theta}(x_{0})} + \log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ] & \text{; No.6} \\
&= E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ] & \text{; No.7} \\
\end{aligned}
$$


**Note that：**
> 第 2 行式子成立的理由: KL 散度的非负性 <br> 第 3 行式子成立的理由: KL 散度的定义计算公式 <br> 第 4 行式子成立的理由: 条件概率的定义计算公式 <br> 第 5 行式子成立的理由: 对数函数的性质 <br> 第 6 行式子成立的理由: $p_{\theta}(x_{0})$ 与求  $q(x_{1:T} \mid x_{0})$ 的期望无关 <br>

对上式子两边取期望 $E_{q(x_{0})}$, 即为类似计算 VAE 中的变分下限 ([Evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound))：


$$
\begin{aligned}
\mathcal{L}_{VLB} 
&\ge E_{q(x_{0})}[-\log{p_{\theta}(x_{0})}] \\
&= E_{q(x_{0})}[E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ]] \\
&= E_{q(x_{0:T})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ] & \text{; DPM paper} \\
\end{aligned}
$$


which has a lower bound provided by [Jense's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) ; 利用 Jense's inequality 将积分的凸函数的值与凸函数的积分联系起来，提供下限；计算期望对于连续变量而言就是计算积分；这样十分类似 VAE 中的推导形式, 从而可以优化交叉熵对目标分布进行学习：


$$
\begin{aligned}
\mathcal{L} = L_\text{CE}
&= \mathbb{E}_{q(x_{0})}[-\log{p_{\theta}(x_{0})}] \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} [ \log\Big({ p_\theta(\mathbf{x}_0) \int p_\theta(\mathbf{x}_{1:T}) d\mathbf{x}_{1:T}}\Big)] & \text{; No.2} \\

&= - \mathbb{E}_{q(\mathbf{x}_0)} [ \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) ] \\

&= - \mathbb{E}_{q(\mathbf{x}_0)} [ \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) ] \\

&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \\

&\leq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} & \text{; Jense's inequality} \\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] \\

&= \color{red} \mathcal{L}_{VLB} \\
\end{aligned}
$$


**Note that：**
> 第 2 行式子成立的理由: $p_{\theta}(x_{0})$ 与求  $q(x_{1:T} \mid x_{0})$ 的期望无关, 而且积分结果为 1 <br>

进一步对 $\mathcal{L}_{VLB}$ 推导，根据 Improved DDPM paper 中的形式：


$$
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] & \text{; No.2} \\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] & \text{; No.3} \\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.4}\\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \color{red}{ \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} } \Big ) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.5}\\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.6}\\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log \frac{\prod_{i=2}^{T} q(\mathbf{x}_{i} \vert \mathbf{x}_0)}{\prod_{i=2}^{T} q(\mathbf{x}_{i-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.7}\\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.8}\\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] & \text{; No.9} \\

&=Exception || KL ? \\

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ] & \text{; No.10} \\
\end{aligned}
$$


**Note that：**
> 第 2 行式子: convert Joint dist. into conditional dist. prod. <br> 第 3 行式子: 对数函数的性质 <br> 第 5 行式子: reverse process formula for $x_{t}$ and $x_{0}$ <br> 第 8 行式子: 对数函数性质 & 累乘形式下分子分母相同项消除 <br> 第 9 行式子: 根据对数性质进行重新组合排列每一项 <br> how and why line No.9 $\longrightarrow$ line No.10；也知答案，逆推过程，凑一个期望，即对数里面分子的一个积分 <br>

**recall that: where the expectation <span style="color:red">line No.9</span> is over a distribution $\bar{q}(x_{t-1})$ that is independent from the variable (namely $x_{t-1}$).** 


$$D_{\text{KL}}(q(x) || p(x)) = \mathbb{E}_{q(x)} [\log q(x) / p(x)]$$


$$
\begin{aligned}
\mathcal{L}_{t}
&=\mathbb{E}_{q(x_{0:T})} \left[ \log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \\

&=~ \mathbb{E}_{{\color{red}q(x_{t-1}|x_t, x_0)}{\color{green}q(x_t,x_0)q(x_{1:t-2,t+1:T}|x_{t-1},x_t,x_0)}} \left[\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \\

&=~ \mathbb{E}_{{\color{green}\bar{q}(x_{t-1})}} \left[ \mathbb{E}_{{\color{red}q(x_{t-1}|x_t, x_0)}} \left[\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \right]  \\

&=~ \mathbb{E}_{\bar{q}(x_{t-1})} \left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0)|| p_\theta(x_{t-1}|x_t)) \right] \\

&=~ D_{\text{KL}}(q(x_{t-1}|x_t, x_0)|| p_\theta(x_{t-1}|x_t))
\end{aligned}
$$


这样就得到了 Improved DDPM paper 的优化目标

Let’s label each component in the variational lower bound loss separately:


$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$


> Every KL term in $L_\text{VLB}$  (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in [closed form](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions).  $L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $\mathbf{x}_T$ is a Gaussian noise. [Ho et al. 2020](https://arxiv.org/abs/2006.11239) models $L_0$ using a separate discrete decoder derived from $\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \boldsymbol{\Sigma}_\theta(\mathbf{x}_1, 1))$ . 

> DDPM paper 中对逆向扩散过程中最后一步从噪声变为原始数据的处理; 图像生成中常用技巧，计算负对数似然，利用标准分布的累计分布的差分(差分的结果就是一个概率值的逼近 )来逼近或者模拟离散高斯分布。连续随机变量的概率密度函数 pdf(probability density function), 离散随机变量的概率质量函数 pmf(probability mass function), 累计分布函数 cdf(cumulative distribution function);

> 为了便于概率的计算，引入 CDF 的概念; CDF 是概率密度函数的积分，能完整描述一个实随机变量X的概率分布。CDF 是 PDF 的(从负无穷到当前值的)积分，PDF 是 CDF 的导数; CDF 相当于其左侧的面积，也相当于小于该值的概率，负无穷的 CDF 值为０，正无穷的 CDF 值总为１.

> DeepMind 2016 引入的 PixelCNN and PixelRNN, 可以作为一个很好的 decoder; &ensp;[生成模型 PixelCNN and PixelCNN++](https://zhuanlan.zhihu.com/p/461693342); &ensp;[自回归模型 PixelCNN and PixelCNN++](https://zhuanlan.zhihu.com/p/415246165)

<center>
    <img src="./images/decoder_diffusion.png" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 10. Decoder of Reverse Diffusion Process for Image. (Image source from DDPM paper.)</div>
</center>

------------------

**[Lilian Weng blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#parameterization-of-l_t-for-training-loss)**

- Parameterization of $L_t$ for Training Loss
- Connection with noise-conditioned score networks (NCSN)
- Parameterization of $\beta_t$
- Parameterization of reverse process variance $\boldsymbol{\Sigma}_\theta$
- Speed up Diffusion Model Sampling (DDIM paper ICLR'2021)
- Conditioned and Controllable Generation

**[Yang Song(Stanford) ](https://yang-song.net/blog/2021/score/)**

- Langevin dynamics (朗之万动力学)
- stochastic differential equations (SDEs)
- SDEs unified; 统一到随机偏微分方程体系下(理论物理)
- Connection to diffusion models and others

> DDPMs Connection to SDEs(stochastic differential equation, 随机微分方程), ODEs(ordinary differential equation, 常微分方程), PDEs(partial differential equation, 偏微分方程)


$$
\mathbf{x}_{i+1} \gets \mathbf{x}_i + \epsilon \nabla_\mathbf{x} \log p(\mathbf{x}) + \sqrt{2\epsilon}~ \mathbf{z}_i, \quad i=0,1,\cdots, K,
$$


<center>
    <img src="./images/Langevin_dynamics.gif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 11. Annealed Langevin dynamics combine a sequence of Langevin chains with gradually decreasing noise scales. (Image source from Yang Song.)</div>
</center>

<center class="half">
    <img src="./images/celeba_large.gif", width="50%" /><img src="./images/cifar10_large.gif", width="50%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 12. Annealed Langevin dynamics for the Noise Conditional Score Network (NCSN) model trained on CelebA (left) and CIFAR-10 (right). We can start from unstructured noise, modify images according to the scores, and generate nice samples. The method achieved state-of-the-art Inception score on CIFAR-10 at its time. (Image source from Yang Song.)</div>
</center>

## Code Implementation 代码实现

<center class="center">
    <img src="./images/DDPM_algo.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 13. The training and sampling algorithms in DDPM (Image source: DDPM NeurIPS'2020)</div>
</center>

> [The Annotated Diffusion Model on Hugging Face](https://huggingface.co/blog/annotated-diffusion)

<details>
<summary> <span style="color:Teal">the example Source Code for DDPMs with PyTorch</span> </summary>

```python
# Pseudo-Code of forward process and training for DDPMs like PyTorch
class GaussianDiffusionSampler(torch.nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super(GaussianDiffusionSampler, self).__init__()

    # 计算任意时刻的 x_t 采样值，基于 x_0 和重参数化技巧
    def q_x(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        # 在 x[0] 的基础上添加噪声
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    # 6. 训练优化的目标函数 最大化对数似然(最小化负对数似然)
    def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        """ 对任意时刻t进行采样计算loss. """
        batch_size = x_0.shape[0]
        
        #对一个 batchsize 样本生成随机的时刻 t
        t = torch.randint(0, n_steps, size=(batch_size//2,))
        t = torch.cat([t, n_steps-1-t], dim=0)
        t = t.unsqueeze(-1)
        
        # x0 的系数
        a = alphas_bar_sqrt[t]
        # eps 的系数
        aml = one_minus_alphas_bar_sqrt[t]
        #生成随机噪音 eps
        e = torch.randn_like(x_0)

        # 构造模型的输入
        x = x_0 * a + e * aml
        # 送入模型，得到t时刻的随机噪声预测值
        output = NN_Model(x, t.squeeze(-1))

        #与真实噪声一起计算误差，求平均值 ---> MSE loss
        return (e - output).square().mean()
```

```python
# Pseudo-Code of forward process and training for DDPMs like PyTorch

# 逆向扩散过程的采样函数 (inference pahse)
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """ 从 x[T] 恢复 x[T-1]、x[T-2]|...x[0]. """
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i ,betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """ 从 x[T] 采样 t 时刻的重构值. """
    t = torch.tensor([t])
    
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    
    eps_theta = model(x, t)
    
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    
    sample = mean + sigma_t * z
    return (sample)
```

</details>

<center class="center">
    <img src="./images/DPM_fig.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 14. The training and sampling algorithms in DDPM. (Image source: DPM ICML'2015)</div>
</center>

<center class="half">
    <img src="./images/forward_reverse_diffusion_s_curve.gif", width="50%" /><img src="./images/forward_reverse_diffusion_swiss_roll.gif", width="50%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 15. The S curve(left) and a two dimensions swiss roll(right) distribution from forward to reverse diffusion process.</div>
</center>

<center>
    <img src="./images/DDPM_Code.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 16. Code Diagram of DDPM and Improved DDPM for Diffusion Models.</div>
</center>
<!-- ![DDPM Code](./Images/DDPM_Code.png) -->

<center class="half">
    <img src="./images/UNet_architecture.png", width="50%" /><img src="./images/MHSA.png", width="50%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 17. U-Net Architecture; Scaled Dot-Product Attention; Multi-Head Attention consists of several
attention layers running in parallel. (Image source from U-Net paper on MICCAI'2015 and Transformer paper on NeurIPS'2017)</div>
</center>


## Score and Naive Score-Based Models

### 1. Definition of Score

&ensp;&ensp;&ensp;&ensp;Suppose our dataset consists of i.i.d. samples $\{ x_{i} \in R^{D} \}^{N}_{i=1}$ from an unknown data distribution $p_{data}(x)$ . We define the **score** of a probability density $p(x)$ to be $\nabla_{x} \log p(x)$ . The **score network** $s_{\theta} : R^{D} \to R^{D}$ is a neural network parameterized by $\theta$, which will be trained to approximate the score of $p_{data}(x)$ . The goal of generative modeling is to use the dataset to learn a model for generating new samples from $p_{data}(x)$ . The framework of score-based generative modeling has two ingredients: score matching and Langevin dynamics.

### 2. Sampling via Langevin dynamics

&ensp;&ensp;&ensp;&ensp;Langevin dynamics can produce samples from a probability density $p(x)$ using only the score function $\nabla_{x} \log p(x)$. Given a fixed step size $\epsilon > 0$, and an initial value $\tilde{x}_{0} \sim \pi (x)$ with $\pi$ being a prior distribution, the Langevin method recursively computes the follwing,

$$\tilde{x}_{t} = \tilde{x}_{t-1} + \frac{\epsilon}{2} \nabla_{x} \log p(\tilde{x}_{t-1}) + \sqrt{\epsilon} z_{t}$$

where $z_{t} \sim \mathcal{N}(0, I)$. The distribution of $\tilde{x}_{T}$ equals $p(x)$ when $\epsilon \to 0$ and $T \to \infty$, in which case $\tilde{x}_{T}$ becomes an exact sample from $p(x)$ under some regularity conditions. When $\epsilon > 0$ and $T < \infty$, a Metropolis-Hastings update is needed to correct the error of **Equation**, but it can often be ignored in practice. In this work, we assume this error is negligible when $\epsilon$ is small and $T$ is large.

&ensp;&ensp;&ensp;&ensp;Note that sampling from **Equation** only requires the score function $\nabla_{x} \log p(x)$. Therefore, in order to obtain samples from $p_{data}(x)$, we can first train our score network such that $s_{\theta}(x) \approx \nabla_{x} \log p_{data}(x)$ and then approximately obtain samples with Langevin dynamics using $s_{\theta}(x)$. This is the key idea of our framework of score-based generative modeling.

### 3. Score Matching and Denoising

&ensp;&ensp;&ensp;&ensp;Score matching(ICML'2005) is originally designed for learning non-normalized statistical models based on i.i.d. samples from an unknown data distribution. Following, we repurpose it for score estimation. Using score matching, we can directly train a score network $s_{\theta}(x)$ to estimate $\nabla_{x} \log p_{data}(x)$ without training a model to estimate $p_{data}(x)$ first. Different from the typical usage of score matching, we opt not to use the gradient of an energy-based model as the score network to avoid extra computation due to higher-order gradients. The objective minimizes $\frac{1}{2} \mathbb{E}_{p_{data}}[ || s_{\theta}(x) - \nabla_{x} \log p_{data}(x) ||^{2}_{2} ]$, which can be shown equivalent to the following up to a constant,

$$\mathbb{E}_{p_{data}} [ tr(\nabla_{x} s_{\theta}(x)) + \frac{1}{2} || s_{\theta}(x) ||^{2}_{2} ]$$

where $\nabla_{x} s_{\theta}(x)$ denotes the Jacobian of $s_{\theta}(x)$. As shown in, under some regularity conditions
the minimizer of Eq.(3) (denoted as $s_{\theta}∗(x)$ ) satisfies $s_{\theta}∗(x) = \nabla_{x} \log p_{data}(x)$ almost surely. In practice, the expectation over $p_{data}(x)$ in **Equation** can be quickly estimated using data samples. However, score matching is not scalable to deep networks and high dimensional data due to the computation of $ tr(\nabla_{x} s_{\theta}(x))$. Below we discuss two popular methods for large scale score matching.

> 关键在于如何解决如此复杂的雅可比矩阵的运算 (convert intractable into tractable on compute)

**Denoising score matching** &ensp;&ensp;Denoising score matching is a variant of score matching that completely circumvents $ tr(\nabla_{x} s_{\theta}(x))$. It first perturbs the data point $x$ with a pre-specified noise distribution $q_{\sigma}(\tilde{x} \mid x)$ and then employs score matching to estimate the score of the perturbed data distribution $q_{\sigma}(\tilde{x}x) \equiv \int q_{\sigma}(\tilde{x} \mid x) p_{data}(x) \mathrm{d}x $. The objective was proved equivalent to the following,

$$\frac{1}{2} \mathbb{E}_{q_{\sigma}(\tilde{x} \mid x)} [ || s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x) ||^{2}_{2} ]$$

As shown in, the optimal score network (denoted as  $s_{\theta}*(x)$) that minimizes **Equation** satisfies $s_{\theta}(x) = \nabla_{\tilde{x}} \log q_{\sigma}(x)$ almost surely. However, $s_{\theta}*(x) = \nabla_{\tilde{x}} \log q_{\sigma}(x) = \nabla_{\tilde{x}} \log p_{data}(x)$ is true only when the noise is small enought such that $q_{\sigma}(x) \approx p_{data}(x)$.

> Score matching for score estimation: another method is **Sliced score matching**

### 4. Challenges of Score Matching

&ensp;&ensp;&ensp;&ensp;The scarcity of data in low density regions can cause difficulties for both score estimation with score matching and MCMC sampling with Langevin dynamics. 数据密度较低的区域(梯度计算)，分数估计不准确，导致郎之万采样结果就不准确。

<center class="center">
    <img src="./images/pitfalls.jpg" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 18. Estimated scores are only accurate in high density regions. (Image source from Yang Song)</div>
</center>

## Noise Conditional Score Networks (NCSN)

### 1. How to add nosing

> 通过条件分数网络进行解决(这里的条件就是加噪声)，通过添加噪声后可以很好的进行分数估计(准确)。那么如何加噪呢？**进退两难**: how do we choose an appropriate noise scale for the perturbation process? Larger noise can obviously cover more low density regions for better score estimation, but it over-corrupts the data and alters it significantly from the original distribution. Smaller noise, on the other hand, causes less corruption of the original data distribution, but does not cover the low density regions as well as we would like. **折中方案**: we use multiple scales of noise perturbations simultaneously; that mean to say: 1)perturbing the data using various levels of noise; 2) simultaneously estimating scores corresponding to all noise levels by training only a single conditional score network. After training, when using Langevin dynamics to generate samples, we **initially** use scores corresponding to large noise, and **gradually anneal down** the noise level. (similar spirit as Diffusion Process) This helps smoothly transfer the benefits of large noise levels to low noise levels where the perturbed data are almost indistinguishable from the original ones. 

<center class="center">
    <img src="./images/single_noise.jpg" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 19. Estimated scores are accurate everywhere for the noise-perturbed data distribution due to reduced low data density regions. (Image source from Yang Song)</div>
</center>

### 2. Definition of NCSN with Multi-Level noise

> [NCSN paper by Yang Song](https://arxiv.org/pdf/1907.05600.pdf)

<center class="center">
    <img src="./images/multi_scale.jpg" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 20. We apply multiple scales of Gaussian noise to perturb the data distribution (first row), and jointly estimate the score functions for all of them (second row). (Image source from Yang Song)</div>
</center>

### 3. Learning NCSN via multi-level noise score matching

> $\sigma$ 加噪水平设置为等比数列比较好; 参数更新和学习时加上指数移动平均 (EMA) 更加稳定;

### 4. NCSN inference via annealed Langevin dynamics

<center class="center">
    <img src="./images/annealed_Langevin_dynamics.png" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 21. NCSN inference via annealed Langevin dynamics. (Image source from NCSN paper)</div>
</center>


## Stochastic Differential Equations (SDEs)

> [Tutorial on Score-based Generative Modeling through SDEs on Colab](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=XCR6m0HjWGVV)

### 1. Generalizing the number of noise scales to infinity

As we already discussed, adding multiple noise scales is critical to the success of score-based generative models. **By generalizing the number of noise scales to infinity**, we obtain not only **higher quality samples**, but also, among others, **exact log-likelihood computation**, and **controllable generation for inverse problem solving**. 简而言之，将有限次的加噪推广到无限次(无穷次数)，这样更加一般化的扩散过程了，同时将这个一般化的扩散过程用一个 SDE 方程进行表示，同时求解更加方便。

### 2. Perturbing Data via SDEs

<center class="center">
    <img src="./images/schematic.jpg" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 22. Solving a reversetime SDE yields a score-based generative model. (Image source from Score-based SDEs paper)</div>
</center>

In order to generate samples with score-based models, we need to consider a [diffusion process](https://en.wikipedia.org/wiki/Diffusion_process) that corrupts data slowly into random noise. Scores will arise when we reverse this diffusion process for sample generation. (为了生成样本, 需要考虑一个随机扩散过程, 即不断加噪使得原始的数据分布变成一个具有良好性质的噪声分布的过程)

A diffusion process is a [stochastic process](https://en.wikipedia.org/wiki/Stochastic_process#:~:text=A%20stochastic%20or%20random%20process%20can%20be%20defined%20as%20a,an%20element%20in%20the%20set.) similar to [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion). Their paths are like the trajectory of a particle submerged in a flowing fluid, which moves randomly due to unpredictable collisions with other particles. Let $\{\mathbf{x}(t) \in \mathbb{R}^d \}_{t=0}^T$ be a diffusion process, indexed by the continuous time variable $t\in [0,T]$. A diffusion process is governed by a stochastic differential equation (SDE), in the following form ($\mathbf{x}(t)$ 是一个连续随机变量, 扩散过程表达为随机微分方程SDE; 布朗运动具有增量独立性、增量服从高斯分布、轨迹连续; SDE指的是微分方程中含有随机参数或随机过程或随机初始值或随机边界值, 此处 $w$ 随机性使得 SDE 成立, 常微分方程的随机化(布朗运动))

$$d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) d t + g(t) d \mathbf{w},$$

where $\mathbf{f}(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$ is called the *drift coefficient*(漂移系数) of the SDE, $g(t) \in \mathbb{R}$ is called the *diffusion coefficient*, and $\mathbf{w}$ represents the standard Brownian motion. You can understand an SDE as a stochastic generalization to ordinary differential equations (ODEs). Particles moving according to an SDE not only follows the deterministic drift $\mathbf{f}(\mathbf{x}, t)$, but are also affected by the random noise coming from $g(t) d\mathbf{w}$. From now on, we use $p_t(\mathbf{x})$ to denote the distribution of $\mathbf{x}(t)$. 

For score-based generative modeling, we will choose a diffusion process such that $\mathbf{x}(0) \sim p_0$, and $\mathbf{x}(T) \sim p_T$. Here $p_0$ is the data distribution where we have a dataset of i.i.d. samples, and $p_T$ is the prior distribution that has a tractable form and easy to sample from. The noise perturbation by the diffusion process is large enough to ensure $p_T$ does not depend on $p_0$. (此处 $p_{0}$ and $p_{T}$ 可视为 SDE 的两个边界情况)

### 3. Reverseing the SDEs to Sample Generation

By starting from a sample from the prior distribution $p_T$ and reversing the diffusion process, we will be able to obtain a sample from the data distribution $p_0$. Crucially, the reverse process is a diffusion process running backwards in time. It is given by the following reverse-time SDE

$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}},$$

where $\bar{\mathbf{w}}$ is a Brownian motion in the reverse time direction, and $dt$ represents an infinitesimal negative time step. This reverse SDE can be computed once we know the drift and diffusion coefficients of the forward SDE, as well as the score of $p_t(\mathbf{x})$ for each $t\in[0, T]$.

### 4. Estimating the reverse SDEs with denoising score matching

<center class="center">
    <img src="./images/sde_diffusion.png" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 23. Overview of score-based generative modeling through SDEs. (Image source from Score-based SDEs paper ICLR'2021)</div>
</center>

Based on the above intuition, we can use the time-dependent score function $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ to construct the reverse-time SDE, and then solve it numerically to obtain samples from $p_0$ using samples from a prior distribution $p_T$. We can train a time-dependent score-based model $s_\theta(\mathbf{x}, t)$ to approximate $\nabla_\mathbf{x} \log p_t(\mathbf{x})$, using the following weighted sum of [denoising score matching](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) objectives.

$$\min_\theta \mathbb{E}_{t\sim \mathcal{U}(0, T)} [\lambda(t) \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2]],$$

where $\mathcal{U}(0,T)$ is a uniform distribution over $[0, T]$, $p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$ denotes the transition probability from $\mathbf{x}(0)$ to $\mathbf{x}(t)$, and $\lambda(t) \in \mathbb{R}_{>0}$ denotes a positive weighting function. (均匀分布、转移概率、正数权重)

In the objective, the expectation over $\mathbf{x}(0)$ can be estimated with empirical means over data samples from $p_0$. The expectation over $\mathbf{x}(t)$ can be estimated by sampling from $p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$, which is efficient when the drift coefficient $\mathbf{f}(\mathbf{x}, t)$ is affine (每个线性变换一定是仿射变换). The weight function $\lambda(t)$ is typically chosen to be inverse proportional to $\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \|_2^2]$.

### 5. Tricks and Tips for designing the Model Framework

**Time-Dependent Score-Based Model** &ensp;&ensp;There are no restrictions on the network architecture of time-dependent score-based models, except that their output should have the same dimensionality as the input, and they should be conditioned on time.

**Several useful tips on architecture choice**:
- It usually performs well to use the [U-net](https://arxiv.org/abs/1505.04597) architecture as the backbone of the score network $s_\theta(\mathbf{x}, t)$,
- We can incorporate the time information via [Gaussian random features](https://arxiv.org/abs/2006.10739). Specifically, we first sample $\omega \sim \mathcal{N}(\mathbf{0}, s^2\mathbf{I})$ which is subsequently fixed for the model (i.e., not learnable). For a time step $t$, the corresponding Gaussian random feature is defined as  

$$[\sin(2\pi \omega t) ; \cos(2\pi \omega t)],$$

where $[\vec{a} ; \vec{b}]$ denotes the concatenation of vector $\vec{a}$ and $\vec{b}$. This Gaussian random feature can be used as an encoding for time step $t$ so that the score network can condition on $t$ by incorporating this encoding. We will see this further in the code.

- We can rescale the output of the U-net by $1/\sqrt{\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \|_2^2]}$. This is because the optimal $s_\theta(\mathbf{x}(t), t)$ has an $\ell_2$-norm close to $\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))]\|_2$, and the rescaling helps capture the norm of the true score. Recall that the training objective contains sums of the form

$$\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2].$$

Therefore, it is natural to expect that the optimal score model $s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$.

- Use [exponential moving average](https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3) (EMA) of weights when sampling. This can greatly improve sample quality, but requires slightly longer training time, and requires more work in implementation. We do not include this in this tutorial, but highly recommend it when you employ score-based generative modeling to tackle more challenging real problems.


## Coding implementation and MNIST demo via SDEs

**the Core Point for Coding**
- Training with Weighted Sum of Denoising Score Matching Objectives
- Sampling with Numerical SDE Solvers
- Sampling with Predictor-Corrector Methods
- Sampling with Numerical ODE Solvers
- Likelihood Computation (NLL)

### Likelihood Computation

A by-product of the probability flow ODE formulation is likelihood computation. Suppose we have a differentiable one-to-one mapping $\mathbf{h}$ that transforms a data sample $\mathbf{x} \sim p_0$ to a prior distribution $\mathbf{h}(\mathbf{x}) \sim p_T$. We can compute the likelihood of $p_0(\mathbf{x})$ via the following [change-of-variable formula](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function)

$$p_0(\mathbf{x}) = p_T(\mathbf{h}(\mathbf{x})) |\operatorname{det}(J_\mathbf{h}(\mathbf{x}))|,$$

where $J_\mathbf{h}(\mathbf{x})$ represents the Jacobian of the mapping $\mathbf{h}$, and we assume it is efficient to evaluate the likelihood of the prior distribution $p_T$. 

The trajectories of an ODE also define a one-to-one mapping from $\mathbf{x}(0)$ to $\mathbf{x}(T)$. For ODEs of the form

$$d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt,$$

there exists an [instantaneous change-of-variable formula](https://arxiv.org/abs/1806.07366) that connects the probability of $p_0(\mathbf{x})$ and $p_1(\mathbf{x})$, given by

$$p_0 (\mathbf{x}(0)) = e^{\int_0^1 \operatorname{div} \mathbf{f}(\mathbf{x}(t), t) d t} p_1(\mathbf{x}(1)),$$

where $\operatorname{div}$ denotes the divergence function (trace of Jacobian). 

In practice, this divergence function can be hard to evaluate for general vector-valued function $\mathbf{f}$, but we can use an unbiased estimator, named [Skilling-Hutchinson estimator](http://blog.shakirm.com/2015/09/machine-learning-trick-of-the-day-3-hutchinsons-trick/), to approximate the trace. Let $\boldsymbol \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The Skilling-Hutchinson estimator is based on the fact that

$$\operatorname{div} \mathbf{f}(\mathbf{x}) = \mathbb{E}_{\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}[\boldsymbol\epsilon^\intercal  J_\mathbf{f}(\mathbf{x}) \boldsymbol\epsilon]. $$

Therefore, we can simply sample a random vector $\boldsymbol \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and then use $\boldsymbol \epsilon^\intercal J_\mathbf{f}(\mathbf{x}) \boldsymbol \epsilon$ to estimate the divergence of $\mathbf{f}(\mathbf{x})$. This estimator only requires computing the Jacobian-vector product $J_\mathbf{f}(\mathbf{x})\boldsymbol \epsilon$, which is typically efficient.

As a result, for our probability flow ODE, we can compute the (log) data likelihood with the following

$$\log p_0(\mathbf{x}(0)) = \log p_1(\mathbf{x}(1)) -\frac{1}{2}\int_0^1 \frac{d[\sigma^2(t)]}{dt} \operatorname{div} s_\theta(\mathbf{x}(t), t) dt. $$

With the Skilling-Hutchinson estimator, we can compute the divergence via

$$\operatorname{div} s_\theta(\mathbf{x}(t), t) = \mathbb{E}_{\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}[\boldsymbol\epsilon^\intercal  J_{s_\theta}(\mathbf{x}(t), t) \boldsymbol\epsilon]. $$

Afterwards, we can compute the integral with numerical integrators. This gives us an unbiased estimate to the true data likelihood, and we can make it more and more accurate when we run it multiple times and take the average. The numerical integrator requires $\mathbf{x}(t)$ as a function of $t$, which can be obtained by the probability flow ODE sampler.

<details>
<summary> <span style="color:Teal">the Source Code for Scored-based SDEs with PyTorch</span> </summary>

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Yang Song et.al "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR'2021
@Python Version: 3.10.4
@PyTorch Version: 1.12.1+cu113
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-29
"""

import time
import math
import functools
from scipy import integrate
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.DataLoader as DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import numpy as np


"""Step 1. Define U-Net for Time-dependent Score-based Network"""
# ---------------------------------------------------------------
class TimeGaussianEncoding(nn.Module):
    """对时间进行特定的傅里叶编码"""
    def __init__(self, embed_dim, scale=30.):
        super(TimeGaussianEncoding, self).__init__()
        # Gaussian random sample weights during initialization;
        # the weights are fixed during optimization and are not trainable.
        self.weight = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.weight[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """"A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super(Dense, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 针对输出进行扩维度，在最后面增加两个维度
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        super(ScoreNet, self).__init__()
        """Initialize a time-dependent score-based network.

        Args:
        marginal_prob_std: A function that takes time t and gives the standard deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        self.marginal_prob_std = marginal_prob_std

        # Gaussian random feature embedding layer for time
        self.embed_time = nn.Sequential(
            TimeGaussianEncoding(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # the swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

        # Encoder of U-Net, 空间分辨率不断下采样, 通道维度不断增加
        # ---------------------------------------------------------------------------------------
        self.conv_1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=0, bias=False)
        self.dense_1 = Dense(embed_dim, channels[0])
        self.group_norm_1 = nn.GroupNorm(4, num_channels=channels[0])
        
        self.conv_2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=0, bias=False) # to skip conection
        self.dense_2 = Dense(embed_dim, channels[1])
        self.group_norm_2 = nn.GroupNorm(32, num_channels=channels[1])
        
        self.conv_3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=0, bias=False) # to skip conection
        self.dense_3 = Dense(embed_dim, channels[2])
        self.group_norm_3 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.conv_4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=0, bias=False)
        self.dense_4 = Dense(embed_dim, channels[3])
        self.group_norm_4 = nn.GroupNorm(32, num_channels=channels[3])
        # ---------------------------------------------------------------------------------------

        # Decoder of U-Net, 空间分辨率不断上采样, 通道维度不断减少, skip connections from encoder
        # ---------------------------------------------------------------------------------------
        self.trans_conv_4 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=3, stride=2, padding=0, output_padding=0, bias=False)
        self.dense_5 = Dense(embed_dim, channels[2])
        self.trans_group_norm_4 = nn.GroupNorm(32, num_channels=channels[2])
        
        # skip connection from Encoder
        self.trans_conv_3 = nn.ConvTranspose2d(channels[2]+channels[2], channels[1], kernel_size=3, stride=2, padding=0, output_padding=1, bias=False)
        self.dense_6 = Dense(embed_dim, channels[1])
        self.trans_group_norm_3 = nn.GroupNorm(32, num_channels=channels[1])
        
        # skip connection from Encoder
        self.trans_conv_2 = nn.ConvTranspose2d(channels[1]+channels[1], channels[0], kernel_size=3, stride=2, padding=0, output_padding=1, bias=False)
        self.dense_7 = Dense(embed_dim, channels[0])
        self.trans_group_norm_2 = nn.GroupNorm(32, num_channels=channels[0])
        # ---------------------------------------------------------------------------------------
        self.trans_conv_1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, kernel_size=3,stride=1, padding=0, output_padding=0)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def forward(self, x, t):
        start_time = time.time()

        # x.shape: [B, C, H, W]
        embed_time = self.act(self.embed_time(t))

        # the forward process of U-Net Encoder
        head_1 = self.conv_1(x)
        head_1 += self.dense_1(embed_time) # 注入时间
        head_1 = self.group_norm_1(head_1)
        head_1 = self.act(head_1)

        head_2 = self.conv_2(head_1)
        head_2 += self.dense_2(embed_time) # 注入时间
        head_2 = self.group_norm_2(head_2)
        head_2 = self.act(head_2)
        
        head_3 = self.conv_3(head_2)
        head_3 += self.dense_3(embed_time) # 注入时间
        head_3 = self.group_norm_3(head_3)
        head_3 = self.act(head_3)

        head_4 = self.conv_4(head_3)
        head_4 += self.dense_4(embed_time) # 注入时间
        head_4 = self.group_norm_4(head_4)
        head_4 = self.act(head_4)
        # ----------------------------------------------

        # the forward process of U-Net Decoder
        hidden = self.trans_conv_4(head_4)
        hidden += self.dense_5(embed_time) # 注入时间
        hidden = self.trans_group_norm_4(hidden)
        hidden = self.act(hidden)

        hidden = self.trans_conv_3(torch.cat([hidden, head_3], dim=1)) # skip connection
        hidden += self.dense_6(embed_time) # 注入时间
        hidden = self.trans_group_norm_3(hidden)
        hidden = self.act(hidden)

        hidden = self.trans_conv_2(torch.cat([hidden, head_2], dim=1)) # skip connection
        hidden += self.dense_7(embed_time) # 注入时间
        hidden = self.trans_group_norm_2(hidden)
        hidden = self.act(hidden)

        hidden = self.trans_conv_1(torch.cat([hidden, head_1], dim=1)) # skip connection
        # Normalize output 进一步进行范数空间的约束
        # 目的是希望预测的分数的二阶范数也逼近真实分数的二阶范数
        x = hidden / self.marginal_prob_std(t)[:, None, None, None]

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return x # shape:[B, C, H, W]


"""Step 2. Define SDE and Denosing Score Matching Objective"""
# ---------------------------------------------------------------
def marginal_prob_std(t, sigma, device):
    """计算任意时刻的扰动后条件高斯分布的标准差"""
    t = torch.tensor(t, divice=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / math.log(sigma))

def diffusion_coeff(t, sigma, device):
    """计算任意时刻的扩散系数, SDE特定(没有漂移系数)"""
    return torch.tensor(sigma**t, device=device)

def loss_func(score_model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    score_model: A PyTorch model instance that represents a time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    # 1. 从 [0.00001, 0.9999] 中随机生成 batch_size 数量的浮点数作为时刻 t
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    # 2. 利用重参数技巧进行采样 
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]

    # 3. 将当前的加噪样本和时间送入 score model 预测分数
    score = score_model(perturbed_x, random_t)

    # 4. 计算 score matching loss
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss


class ExponentialMovingAverage(nn.Module): # EMA trick
    def __init__(self, model, decay=0.9999, device=None):
        super(ExponentialMovingAverage, self).__init__()
        # make a copy of the model for acculating moving average of weights
        self.modeules = deepcopy(model)
        self.modeules.eval()
        self.deacy = decay
        self.device = device # perform  EMA on different device from model if set
        if self.device is not None:
            self.modeules.to(device=device)
    
    def _update(self, model, update_func):
        with torch.no_grad():
            for ema_v, model_v in zip(self.modeules.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_func(ema_v, model_v))

    def update_func(self, model):
        self._update(model, update_func=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_func=lambda e, m: m)


# Sampling from the ScoreModel
# -------------------------------
# 1. 基于随机微分方程数值解法来生成样本数据
# 2. 融合欧拉数值求解 + 郎之万动力学采样 ---> predictor-corrector
# 3. 基于伴随常微分方程的数值计算来生成样本数据


"""sample 1. the Euler-Maruyama approach"""
def Euler_Maruyama_sampler(score_model, marginal_prob_std, diffusion_coeff, 
                        batch_size=64, num_steps=500, device='cuda', eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.    
    """
    # step 1. 定义初始时间 1 和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]

    # step 2. 定义采样的逆时间网络和每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # step 3. 根据欧拉采样算法求解 reverse time SDE
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      

    # Do not include any noise in the last sampling step.
    # step 4. 取最后一步的期望值作为生成的样本
    return mean_x


"""sample 2. predictor-corrector"""
def pc_sampler(score_model, marginal_prob_std, diffusion_coeff, batch_size=64, 
            num_steps=500, snr=0.16, device='cuda', eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns: 
    Samples.
    """
    # step 1. 定义初始时间 1 和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    
    # step 2. 定义采样的逆时间网络和每一步的时间步长
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    # step 3. 重复交替进行 P & C 采样方式
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            print(f"Langevin step size: {langevin_step_size}")

            for _ in range(10): # hard-code for Langevin 迭代采样次数
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)     
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                print(f"Langevin step size: {langevin_step_size}")

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

    # The last step does not include any noise
    # step 4. 取最后一步的欧拉求解的期望值作为生成的样本
    return x_mean


"""3. 基于伴随常微分方程的数值计算来生成样本数据"""
def ode_sampler(score_model, marginal_prob_std, diffusion_coeff, batch_size=64, 
            atol=1e-5, rtol=1e-5, device='cuda', z=None, eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
        of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t    
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x


# ----------------------------------------
def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x, score_model, marginal_prob_std, diffusion_coeff, batch_size=64, device='cuda', eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
    x: Input data.
    score_model: A PyTorch model representing the score-based model.
    marginal_prob_std: A function that gives the standard deviation of the 
        perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the 
        forward SDE.
    batch_size: The batch size. Equals to the leading dimension of `x`.
    device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
    eps: A `float` number. The smallest time step for numerical stability.

    Returns:
    z: The latent code for `x`.
    bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)
        
    def divergence_eval(sample, time_steps, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    

    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
            return div.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones((shape[0],)) * t    
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    return z, bpd


# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sigma = 25.0
    marginal_prob_std_func = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=sigma, device=device)

    # Traing the Score Model
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_func)
    score_model = score_model.to(device=device)

    num_epochs = 50
    batch_size = 32
    learning_rate = 1e-4

    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    data_loder = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(score_model.parameters(), lr=learning_rate)
    tqdm_epoch = tqdm.tqdm(range(num_epochs))
    ema = ExponentialMovingAverage(score_model)

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loder:
            x = x.to(device)
            loss = loss_func(score_model, x, marginal_prob_std_func)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema._update(score_model)
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
    print(f'Average ScoreMatching Loss: {avg_loss / num_items:6.f}')
    torch.save(score_model.state_dict(), f"score_model_{epoch}_ckpt.pth")
    # ----------------------------------------------------------------------

    # Sampling from the ScoreModel
    # -------------------------------
    # 1. 基于随机微分方程数值解法来生成样本数据
    # 2. 融合欧拉数值求解 + 郎之万动力学采样 ---> predictor-corrector
    # 3. 基于伴随常微分方程的数值计算来生成样本数据
    # 4. 利用 ODE 计算负对数似然
```

</details>


### Connection of Diffusion Models, Score Matching and SDEs
- 可以认为扩散模型和基于分数匹配的方法都是基于随机微分方程生成模型的两种特殊情况
- 或者认为可以用 SDEs 方法对 Diffusion-based and Score-based 进行统一
- Diffusion-based 主要从**变分下界**对模型进行优化求解
- Score-based 主要从**分数匹配**对模型进行优化求解
- DDPM(denoising diffusion probability model) and SMLD(score matching Langevin dynamics)
- 扩散生成模型：贝叶斯概率分布角度建模 or 分数和郎之万采样角度建模
- 对数据增加扰动为手段，通过神经网络对加噪后的数据进行建模并最终学到目标数据分布的过程
- Denoising Score Matching with Langevin Dynamics
    - 为了更高估计分数(低密度数据区域)，需要通过对数据增加不同量级的噪声
    - 噪声量级有大有小，都是在原始数据上加噪，最终的分布趋向 $\mathcal{N}(0, \sigma^{2}I)$
    - 利用分数匹配来训练 NCSN 网络，从而使得 NCSN 能够估计任意加噪后的分布的分数 $\nabla_{x} \log p(x)$
    - 基于任意加噪分布的分数和退火郎之万采样，生成准确的符合原始数据分布的样本
- Denoising Diffusion Probabilistic Model
    - 通过离散马尔科夫链进行加噪过程的
    - 加噪的量级在 $0 \sim 1$ 之间，并且 $\beta_{t}$ 逐渐增大，而且条件分布的均值也是缩放后的，缩放系数 $\sqrt{1 - \beta_{t}}$ 逐渐减少
    - 训练过程是用高斯分布近似反推前向过程，训练的目标函数是对数似然的下界
    - 如果训练的目标函数用分数来表示，那么分数匹配的 $loss$ 的系数其实和 SMLD 中的一致，都是在原始分布上加噪后的分布的方差，但是采样的公式和 Langevin Dynamics 不完全一样 (Yang Song Oral paper ICLR'2021)


-------------
Cited as:
```shell
@article{WeiLi2022DDPM-ScoredMatching-SDE,
  title   = Image Generation via Diffusion Models and Scored-Matching and SDEs,
  author  = Wei Li,
  journal = https://2694048168.github.io/blog/,
  year    = 2022-09,
  url     = https://2694048168.github.io/blog/#/PaperMD/diffusion_models
}
```


## Reference

----------------------------

[1] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics," ICML'2015

[DPM Paper on ICML'2015](https://proceedings.mlr.press/v37/sohl-dickstein15.html)
&emsp;&emsp;[DPM Paper on arXiv'2015](https://arxiv.org/abs/1503.03585)
&emsp;&emsp;[DPM Original Code on GitHub](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)

<details>
<summary> <span style="color:Teal">Abstract 摘要</span> </summary>

> A **central problem** in machine learning involves modeling complex data-sets using highly **flexible** families of <span style="color:DarkOrchid">**probability distributions**</span> in which learning, sampling, inference, and evaluation are still analytically or computationally **tractable**. Here, we develop an approach that simultaneously achieves both flexibility and tractability. The essential idea, inspired by <span style="color:DarkOrchid">**non-equilibrium statistical physics**</span>, is to systematically and slowly destroy structure in a data distribution through an <span style="color:red">**iterative forward diffusion process**</span>. We then learn a <span style="color:red">**reverse diffusion process**</span> that restores structure in data, yielding a  highly flexible and tractable <span style="color:Aqua">**generative model of the data**</span>. This approach allows us to rapidly learn, sample from, and  evaluate probabilities in deep generative models with thousands of layers or time steps, as well as to <span style="color:Aqua">**compute conditional and posterior probabilities**</span> under the learned model. We additionally release an open source reference implementation of the  algorithm.

</details>

<details>
<summary> <span style="color:PeachPuff">Conclusion 结论</span> </summary>

> We have introduced a <span style="color:Aqua">**novel algorithm for modeling probability distributions**</span> that enables exact sampling and evaluation of probabilities and demonstrated its effectiveness on a variety of toy and real datasets, including challenging natural image datasets. For each of these tests we used a similar basic algorithm, showing that our method can **accurately model a wide variety of distributions**. Most existing density estimation techniques must sacrifice modeling power in order to stay tractable and efficient, and sampling or evaluation are often extremely expensive. <span style="color:DarkOrchid">**The core of our algorithm consists of estimating the reversal of a Markov diffusion chain which maps data to a noise distribution; as the number of steps is made large, the reversal distribution of each diffusion step becomes simple and easy to estimate**</span>. The result is an algorithm that can learn a fit to any data distribution, but which remains tractable to train, exactly sample from, and evaluate, and under <span style="color:Khaki">**which it is straightforward to manipulate conditional and posterior distributions**</span>.

</details>

----------------------------

[2] Jonathan Ho, Ajay Jain, Pieter Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS'2020

[DDPM Paper on NeurIPS'2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
&emsp;&emsp;[DDPM Paper on arXiv'2020](https://arxiv.org/abs/2006.11239)
&emsp;&emsp;[DDPM Original Code on GitHub](https://github.com/hojonathanho/diffusion)

<details>
<summary> <span style="color:Teal">Abstract 摘要</span> </summary>

> We present <span style="color:Aqua">**high quality image synthesis results**</span> using <span style="color:Aqua">diffusion probabilistic models </span>, a class of <span style="color:Aqua">latent variable models</span> inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound  designed according to a <span style="color:red">novel connection</span> between diffusion  probabilistic models and <span style="color:DarkOrchid">**denoising score matching with Langevin dynamics**</span>, and our models naturally admit a <span style="color:DarkOrchid">progressive lossy  decompression scheme that can be interpreted as a generalization of **autoregressive decoding**</span>. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of  3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. 

</details>

<details>
<summary> <span style="color:PeachPuff">Conclusion 结论</span> </summary>

> We have presented **high quality image samples using diffusion models**, and we have <span style="color:DarkOrchid">**found connections among diffusion models and variational inference**</span> for training Markov chains, denoising score matching and annealed Langevin dynamics (and energy-based models by extension), autoregressive models, and progressive lossy compression. Since <span style="color:Aqua">**diffusion models seem to have excellent inductive biases for image data**</span>, we look forward to investigating their utility in other **data modalities** and as components in other types of generative models and machine learning systems.

</details>

---------------------------

[3] Alex Nichol, Prafulla Dhariwal, "Improved  Denoising Diffusion Probabilistic Models," ICML'2021

[Improved DDPM Paper on ICML'2021](http://proceedings.mlr.press/v139/nichol21a.html)
&emsp;&emsp;[Improved DDPM Paper on arXiv'2021](https://arxiv.org/abs/2102.09672)
&emsp;&emsp;[Improved DDPM Original Code on GitHub](https://github.com/openai/improved-diffusion)

<details>
<summary> <span style="color:Teal">Abstract 摘要</span> </summary>

> Denoising diffusion probabilistic models (DDPM) are a class of **generative models** which have recently been shown to produce  excellent samples. We show that with a few simple modifications, DDPMs can also <span style="color:DarkOrchid">**achieve competitive loglikelihoods while maintaining  high sample quality**</span>. Additionally, we <span style="color:Aqua">find that learning variances of the reverse diffusion process allows sampling with an order of magnitude fewer forward passes with a negligible difference in sample quality, which is important for the practical deployment of  these models</span>. We additionally use precision and recall to compare <span style="color:Khaki">**how well DDPMs and GANs cover the target distribution**</span>. Finally, we  show that the sample quality and likelihood of these models scale smoothly with model capacity and training compute, making them **easily scalable**.

</details>

<details>
<summary> <span style="color:PeachPuff">Conclusion 结论</span> </summary>

> We have shown that, with a few modifications, <span style="color:Khaki">**DDPMs can sample much faster and achieve better log-likelihoods with little impact on sample quality**</span>. The likelihood is improved by learning Σθ using our parameterization and Lhybrid objective. This brings the likelihood of these models much closer to other likelihood-based models. We surprisingly discover that this change also allows <span style="color:Khaki">**sampling from these models with many fewer steps**</span>.

> We have also found that DDPMs can match the sample quality of GANs while <span style="color:DarkOrchid">**achieving much better mode coverage</span>** as measured by  recall. Furthermore, we have investigated how DDPMs scale with the amount of available training compute, and found that <span style="color:DarkOrchid">**more training compute trivially leads to better sample quality and log-likelihood**</span>.

> The combination of these results makes <span style="color:Aqua">**DDPMs an attractive choice for generative modeling**</span>, since they combine **good  log-likelihoods**, **high-quality samples**, and **reasonably fast sampling** with a well-grounded, **stationary training objective that scales easily with training compute**. These results indicate that DDPMs are a promising direction for future research.

</details>

----------------------------

[4] Prafulla Dhariwal, Alex Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS'2021

[Paper on NeurIPS'2021](https://papers.nips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2105.05233)
&emsp;&emsp;[Original Code on GitHub](https://github.com/openai/guided-diffusion)

<details>
<summary> <span style="color:Teal">Abstract 摘要</span> </summary>

> We show that <span style="color:Aqua">**diffusion models**</span> can achieve **image sample quality superior** to the current state-of-the-art <span style="color:Aqua">**generative models**</span>. We achieve this on **unconditional image synthesis** by finding a better architecture through a series of ablations. For **conditional image synthesis**, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier. We achieve an FID of 2.97 on ImageNet 128×128, 4.59 on ImageNet 256×256, and 7.72 on ImageNet 512×512, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution. Finally, we find that classifier guidance combines well with upsampling diffusion models, further improving FID to 3.94 on ImageNet 256×256 and 3.85 on ImageNet 512×512.

</details>

<details>
<summary> <span style="color:PeachPuff">Conclusion 结论</span> </summary>

> We have shown that diffusion models, a class of likelihood-based  models with a stationary training objective, can obtain <span style="color:DarkOrchid">**better sample quality than state-of-the-art GANs**</span>. Our improved architecture is sufficient to achieve this on **unconditional** image generation tasks, and our classifier guidance technique allows us to do so on **class-conditional** tasks. In the latter case, we find that the scale of the **classifier gradients can be adjusted to trade off diversity for fidelity**. These guided diffusion models can **reduce the sampling time gap** between GANs and diffusion models, although diffusion models still require multiple forward passes during sampling. Finally, by combining guidance with upsampling, we can further improve sample  quality on high-resolution conditional image synthesis. 

</details>

----------------------------

[5] Chitwan Saharia, Jonathan Ho, et al. "Image Super-Resolution via Iterative Refinement," ICCV'2021

**Denoising diffusion models for image super-resolution and cascaded image generation.**

[SR3 Project on Google](https://iterative-refinement.github.io/)
&emsp;&emsp;[SR3 Paper on arXiv'2021](https://arxiv.org/abs/2104.07636)
&emsp;&emsp;[Code on GitHub](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

<details>
<summary> <span style="color:Teal">Abstract 摘要</span> </summary>

> We present SR3, an approach to **image Super-Resolution** via Repeated Refinement. SR3 adapts **denoising diffusion probabilistic** models [DDPM, DPM] to **conditional image generation** and performs super-resolution through a stochastic iterative denoising process. Output generation starts with pure Gaussian noise and iteratively refines the noisy output using a U-Net model trained on denoising at various noise levels. SR3 exhibits strong performance on super-resolution tasks at different magnification factors, on **faces and natural images**. We conduct human evaluation on a standard 8× face super-resolution task on CelebA-HQ, comparing with SOTA GAN methods. SR3 achieves a fool rate close to 50%, suggesting **photo-realistic** outputs, while GANs do not exceed a fool rate of 34%. We further show the effectiveness of SR3 in **cascaded image generation**, where generative models are chained with uper-resolution models, yielding a competitive FID score of 11.3 on ImageNet.

</details>

<details>
<summary> <span style="color:PeachPuff">Conclusion 结论</span> </summary>

> Bias is an important problem in all generative models. SR3 is no  different, and suffers from bias issues. While in theory, our log- likelihood based objective is mode covering (e.g., unlike some GAN- based objectives), we believe it is likely our diffusion-based models  drop modes. We observed some evidence of mode dropping, the model consistently generates nearly the same image output during  sampling (when conditioned on the same input). We also observed the model to generate very continuous skin texture in face super- resolution, dropping moles, pimples and piercings found in the reference. SR3 should not be used for any real world super-resolution tasks, until these biases are thoroughly understood and mitigated.

> In conclusion, SR3 is an approach to image superresolution via terative refinement. SR3 can be used in a cascaded fashion to  generate high resolution super-resolution images, as well as unconditional samples when cascaded with a unconditional model. We demonstrate SR3 on face and natural image super-resolution at high resolution and high magnification ratios (e.g., 64×64!256×256 and 256×256!1024×1024). SR3 achieves a human fool rate close to  50%, suggesting photo-realistic outputs.

</details>

----------------------------

[6] Jiaming Song, Chenlin Meng, Stefano Ermon, "Denoising Diffusion Implicit Models," ICLR'2021

[DDIM Paper on ICLR'2021](https://openreview.net/forum?id=St1giarCHLP)
&emsp;&emsp;[DDIM Paper on arXiv'2021](https://arxiv.org/abs/2010.02502)
&emsp;&emsp;[DDIM Original Code on GitHub](https://github.com/ermongroup/ddim)

----------------------------

[7] Alec Radford, Jong Wook Kim, Chris Hallacy, "Learning Transferable Visual Models From Natural Language Supervision," ICML'2021

[CLIP on OpenAI](https://openai.com/blog/clip/)
&emsp;&emsp;[CLIP Paper on ICML'2021 Oral](https://icml.cc/virtual/2021/oral/9194)
&emsp;&emsp;[CLIP Paper on ICML'2021](http://proceedings.mlr.press/v139/radford21a.html)
&emsp;&emsp;[CLIP Paper on arXiv'2021](https://arxiv.org/abs/2103.00020)
&emsp;&emsp;[CLIP Original Code on GitHub](https://github.com/OpenAI/CLIP)

----------------------------

[8] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, et al. "Zero-Shot Text-to-Image Generation," ICML'2021

[DALL·E on OpenAI](https://openai.com/blog/dall-e/)
&emsp;&emsp;[DALL·E Paper on ICML'2021 Spotlight](https://icml.cc/virtual/2021/spotlight/9430)
&emsp;&emsp;[DALL·E Paper on ICML'2021](http://proceedings.mlr.press/v139/ramesh21a.html)
&emsp;&emsp;[DALL·E Paper on arXiv'2021](https://arxiv.org/abs/2102.12092)
&emsp;&emsp;[DALL·E Original Code on GitHub](https://github.com/openai/dall-e)

----------------------------

[9] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents," OpenAI arXiv'2022

[DALL·E 2 on OpenAI](https://openai.com/dall-e-2/)
&emsp;&emsp;[DALL·E 2 Paper on arXiv'2022](https://arxiv.org/abs/2204.06125)
&emsp;&emsp;[DALL·E 2 Code on GitHub](https://github.com/lucidrains/DALLE2-pytorch)

----------------------------

[10] Chitwan Saharia, William Chan, Saurabh Saxena, et al. "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding," Google arXiv'2022

[Imagen on Google](https://imagen.research.google/)
&emsp;&emsp;[Imagen Paper on arXiv'2022](https://arxiv.org/abs/2205.11487)
&emsp;&emsp;[Imagen Code on GitHub](https://github.com/lucidrains/imagen-pytorch)

----------------------------

[11] Ming Ding, Zhuoyi Yang, Wenyi Hong, et al. "CogView: Mastering Text-to-Image Generation via Transformers," NeurIPS'2021

[CogView Paper on NeurIPS'2021](https://proceedings.neurips.cc/paper/2021/hash/a4d92e2cd541fca87e4620aba658316d-Abstract.html)
&emsp;&emsp;[CogView Paper on arXiv'2021](https://arxiv.org/abs/2105.13290)
&emsp;&emsp;[CogView Original Code on GitHub](https://github.com/THUDM/CogView)

----------------------------

[12] Ming Ding, Wendi Zheng, Wenyi Hong, Jie Tang, "CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformer," arXiv'2022

[CogView2 Paper on arXiv'2022](https://arxiv.org/abs/2204.14217)
&emsp;&emsp;[CogView2 Original Code on GitHub](https://github.com/thudm/cogview2)

----------------------------

[13] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, Jie Tang, "CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers," arXiv'2022

[CogVideo Paper on arXiv'2022](https://arxiv.org/abs/2205.15868)
&emsp;&emsp;[CogVideo Original Code on GitHub](https://github.com/thudm/cogvideo)

----------------------------

[14] Chenfei Wu, Jian Liang, Lei Ji, et al. "NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion," ECCV'2022

[NÜWA Paper on arXiv'2021](https://arxiv.org/abs/2111.12417)
&emsp;&emsp;[NÜWA Code on GitHub](https://github.com/lucidrains/nuwa-pytorch)

----------------------------

[15] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, et al. "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models," OpenAI arXiv'2021

[GLIDE Paper on arXiv'2021](https://arxiv.org/abs/2112.10741)
&emsp;&emsp;[GLIDE Original Code on GitHub](https://github.com/openai/glide-text2im)

----------------------------

[16] Han Zhang, Weichong Yin, Yewei Fang, et al. "ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation," arXiv'2021

[ERNIE-ViLG Paper on arXiv'2021](https://arxiv.org/abs/2112.15283)
&emsp;&emsp;[ERNIE-ViLG Code on GitHub](https://github.com/PaddlePaddle/FleetX)

----------------------------

[17] Diederik P. Kingma, Max Welling, "Auto-Encoding Variational Bayes," ICLR'2014

[Reparameterization Paper on ICLR'2014](https://openreview.net/forum?id=33X9fd2-9FyZd)
&emsp;&emsp;[Reparameterization Paper on arXiv'2013](https://arxiv.org/abs/1312.6114)
&emsp;&emsp;[Reparameterization Code on GitHub](https://github.com/AntixK/PyTorch-VAE)

----------------------------

[18] Robin Rombach, Andreas Blattmann, et al. "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR'2022

[Oral Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
&emsp;&emsp;[Paper on arXiv'2022](https://arxiv.org/abs/2112.10752)
&emsp;&emsp;[Paper Project](https://ommer-lab.com/research/latent-diffusion-models/)
&emsp;&emsp;[Paper Original Code on GitHub](https://github.com/compvis/latent-diffusion)
&emsp;&emsp;[Stable Diffusion Project](https://ommer-lab.com/research/latent-diffusion-models/)
&emsp;&emsp;[Stable Diffusion Release](https://stability.ai/blog/stable-diffusion-public-release)
&emsp;&emsp;[Stable Diffusion Release for researchers](https://stability.ai/blog/stable-diffusion-announcement)
&emsp;&emsp;[Stable Diffusion on Hugging Face](https://huggingface.co/CompVis)

----------------------------

[19] Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf, "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting," ICML'2021

[Paper on ICML'2021](https://icml.cc/virtual/2021/poster/8591)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2101.12072)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/zalandoresearch/pytorch-ts)

----------------------------

[20] Feller William, "On the Theory of Stochastic Processes, with Particular Reference to Applications," Berkeley Symposium on Mathematical Statistics and Probability, 1949: 403-432 (1949)

[Paper for Open Access, OA](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-the-Theory-of-Stochastic-Processes-with-Particular-Reference-to/chapter/On-the-Theory-of-Stochastic-Processes-with-Particular-Reference-to/bsmsp/1166219215)

----------------------------

[21] Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI'2015

[U-Net Paper on arXiv'2015](https://arxiv.org/abs/1505.04597)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/milesial/Pytorch-UNet)

----------------------------

[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need," NeurIPS'2017

[Transformer Paper on NeurIPS'2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
&emsp;&emsp;[Transformer Paper on arXiv'2017](https://arxiv.org/abs/1706.03762)
&emsp;&emsp;[Paper Original Code on GitHub](https://github.com/tensorflow/tensor2tensor)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/huggingface/transformers)

----------------------------

[23] Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song, "Denoising Diffusion Restoration Models," ICLR'2022

[DDRM Project on Website](https://ddrm-ml.github.io/)
&emsp;&emsp;[DDRM Paper on ICLR'2022 Oral](https://openreview.net/forum?id=BExXihVOvWq)
&emsp;&emsp;[DDRM Paper on arXiv'2022](https://arxiv.org/abs/2201.11793)
&emsp;&emsp;[DDRM Original Code on GitHub](https://github.com/bahjat-kawar/ddrm)

----------------------------

[24] Diederik P. Kingma, Prafulla Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions," NeurIPS'2018

[Glow Paper on NeuriPS'2018](https://papers.nips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html)
&emsp;&emsp;[Glow Paper on arXiv'2018](https://arxiv.org/abs/1807.03039)
&emsp;&emsp;[Glow Original Code on GitHub](https://github.com/openai/glow)

----------------------------

[25] Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, Koray Kavukcuoglu, "Conditional Image Generation with PixelCNN Decoders," NeurIPS'2016

[PixelCNN Paper on NeuriPS'2016](https://proceedings.neurips.cc/paper/2016/hash/b1301141feffabac455e1f90a7de2054-Abstract.html)
&emsp;&emsp;[PixelCNN Paper on arXiv'2016](https://arxiv.org/abs/1606.05328)
&emsp;&emsp;[Implementation Code](https://paperswithcode.com/paper/conditional-image-generation-with-pixelcnn)

----------------------------

[26] Tim Salimans, Andrej Karpathy, Xi Chen, Diederik P. Kingma, "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications," 2017 from OpenAI

[PixelCNN++ Paper on arXiv'2017](https://arxiv.org/abs/1701.05517)
&emsp;&emsp;[PixelCNN++ Original Code](https://github.com/openai/pixel-cnn)
&emsp;&emsp;[Implementation Code](https://paperswithcode.com/paper/pixelcnn-improving-the-pixelcnn-with)

----------------------------

[27] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole, "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR'2021

[Oral and Outstanding Paper Award on ICLR'2021](https://openreview.net/forum?id=PxTIG12RRHS)
&emsp;&emsp;[paper at arXiv'2021](https://arxiv.org/abs/2011.13456)
&emsp;&emsp;[Original Code](https://github.com/yang-song/score_sde)

----------------------------

[28] Yang Song, and Stefano Ermon, "Improved Techniques for Training Score-Based Generative Models," NeurIPS'2020

[Paper on NeurIPS'2020](https://proceedings.neurips.cc/paper/2020/hash/92c3b916311a5517d9290576e3ea37ad-Abstract.html)
&emsp;&emsp;[paper at arXiv'2020](https://arxiv.org/abs/2006.09011)
&emsp;&emsp;[Original Code](https://github.com/ermongroup/ncsnv2)

----------------------------

[29] Yang Song, and Stefano Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," NeurIPS'2019

[Oral top-5% Paper on NeurIPS'2019](https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html)
&emsp;&emsp;[Oral top-5% Paper on Openreview](https://openreview.net/forum?id=B1lcYrBgLH)
&emsp;&emsp;[paper at arXiv'2019](https://arxiv.org/abs/1907.05600)
&emsp;&emsp;[Original Code](https://github.com/ermongroup/ncsn)

----------------------------

[30] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng, "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains," NeurIPS'2020

[spotlight Paper at NeurIPS'2020](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)
&emsp;&emsp;[paper project](https://bmild.github.io/fourfeat/)
&emsp;&emsp;[paper at arXiv'2020](https://arxiv.org/abs/2006.10739)
&emsp;&emsp;[Original Code](https://github.com/tancik/fourier-feature-networks)

----------------------------
[31] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet, "Video Diffusion Models," arXiv'2022 from Google Inc.

[paper project](https://video-diffusion.github.io/)
&emsp;&emsp;[paper at arXiv'2020](https://arxiv.org/abs/2204.03458)
&emsp;&emsp;[Implemental Code](https://paperswithcode.com/paper/video-diffusion-models)

----------------------------
[32] Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein, "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise," arXiv'2022

[Cold Diffusion paper at arXiv'2022](https://arxiv.org/abs/2208.09392)
&emsp;&emsp;[Paper Code](https://github.com/arpitbansal297/cold-diffusion-models)
&emsp;&emsp;[Implementation Code](https://paperswithcode.com/paper/cold-diffusion-inverting-arbitrary-image)
