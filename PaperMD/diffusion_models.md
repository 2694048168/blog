# Image Generation via Diffusion Models 基于扩散模型的图像生成

- &ensp;<span style="color:MediumPurple">Title</span>: Image Generation via Diffusion Models
- &ensp;<span style="color:Moccasin">Tags</span>: Generative Models; Diffusion Models; Probability Models;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-08

> Deep Generative Learning: Learning to generate data

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

[DALLE 2: a new AI system that can create realistic images and art from a description in natural language](https://openai.com/dall-e-2/)

<center>
    <img src="./images/DALLE2_0.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 1. An astronaut riding a horse in a photorealistic style. (Image source from DALLE-2, OpenAI)</div>
</center>

[Imagen: unprecedented photorealism x deep level of language understanding](https://imagen.research.google/)

<center>
    <img src="./images/Imagen_1.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 2. The Toronto skyline with Google brain logo written in fireworks. (Image source from Imagen, Google)</div>
</center>

## Image Generation Paradigm 图像生成研究范式

**Reference Blogs**

- [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
- [CVPR 2022 Tutorial: Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://cvpr2022-tutorial-diffusion-models.github.io/)
- [Diffusion Models for Deep Generative Learning](https://zaixiang.notion.site/Diffusion-Models-for-Deep-Generative-Learning-24ccc2e2a11e40699723b277a7ebdd64)
- [Computer Vison: Models, Learning, and Inference 中英版本图书](https://item.jd.com/12218342.html)
- [Awesome Diffusion Models](https://github.com/heejkoo/Awesome-Diffusion-Models)

**各类生成模型对比**

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/generative-overview.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 3. Overview of different types of generative models. (Image source from Lilian Weng blog, that the Applied AI Research Manager, OpenAI)</div>
</center>
<!-- ![generative-overview](./Images/generative-overview.png) -->

Diffusion Models 和其他生成模型最大的区别是它的 latent code(z) 和原图是同尺寸大小的，当然也有基于压缩的 Latent Diffusion Model &ensp;[CVPR'2022][<sup>[18]</sup>](#refer-18)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/2022_CVPR_DDPMs_tutorial_fig.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 4. The Landscape of Deep Generative Learning. (Image source from 2022-CVPR-Tutorial DDPMs slides)</div>
</center>

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
&= p_{r}(w,x,y|z)p_{r}(z) \newline
&= p_{r}(w,x|y,z)p_{r}(y|z)p_{r}(z) \newline
&= p_{r}(w|x,y,z)p_{r}(x|y,z)p_{r}(y|z)p_{r}(z) \newline
\end{aligned}
$$

**同时利用上马尔科夫链的条件独立性质 $x -> y -> z$**

$$
\begin{aligned}
p_{r}(x,y,z)
&= p_{r}(x,y|z)p_{r}(z) \newline
&= p_{r}(x|y,z)p_{r}(y|z)p_{r}(z) \newline
&= p_{r}(x|y)p_{r}(y|z)p_{r}(z) \newline
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
& \text{假设学校学生的总人数为M} \newline
& \text{穿长裤的男生为：} M\ast P(Boy)\ast P(Pants|Boy) \newline
& \text{穿长裤的女生为：} M\ast P(Girl)\ast P(Pants|Girl) \newline
& \text{穿长裤的学生总人数为：}  \newline
& M\ast P(Boy)\ast P(Pants|Boy) + M\ast P(Girl)\ast P(Pants|Girl) \newline
& \text{求解穿长裤的学生里面有多少女生：}\newline
P(Girl|Pants)
&= \frac{M\ast P(Girl)\ast P(Pants|Girl)}{M\ast P(Boy)\ast P(Pants|Boy) + M\ast P(Girl)\ast P(Pants|Girl)} \newline
&= \frac{P(Girl)\ast P(Pants|Girl)}{P(Boy)\ast P(Pants|Boy) + P(Girl)\ast P(Pants|Girl)}  & \text{;分母就是P(Pants)} \newline
&= \frac{P(Pants|Girl) \ast P(Girl)}{P(Pants)} \newline
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
&= \frac{{\color{cyan}p_{r}(x|y)} {\color{red}p_{r}(y)}} {\color{blue}p_{r}(x)} & \text{ ;origin Bayes's rule} \newline
&= \frac{p_{r}(x|y)p_{r}(y)}{\int p_{r}(x,y) \mathrm{d}y} & \text{ ;marginal dist.}\newline
&= \frac{p_{r}(x|y)p_{r}(y)}{\int p_{r}(x|y)p_{r}(t) \mathrm{d}y} & \text{ ;conditional dist.} \newline
\end{aligned}
$$

其中 $p_{r}(y|x)$ 称之为后验概率 (posterior); $p_{r}(y)$ 称之为先验概率 (prior); $p_{r}(x)$ 称之为证据 (evidence); $p_{r}(x|y)$ 称之为似然性 (likelihood).

**多元变量的贝叶斯公式 Bayes's rule**

$$
\begin{aligned}
p_{r}(x_{t-1},x_{t},x_{0})  
&= p_{r}(x_{t-1}, x_{t}|x_{0})p_{r}(x_{0}) \newline
&= p_{r}(x_{t-1}|x_{t},x_{0})p_{r}(x_{t}|x_{0})p_{r}(x_{0}) \newline
\end{aligned}
$$

$$
\begin{aligned}
p_{r}(x_{t},x_{t-1},x_{0})  
&= p_{r}(x_{t}, x_{t-1}|x_{0})p_{r}(x_{0}) \newline
&= p_{r}(x_{t}|x_{t-1},x_{0})p_{r}(x_{t-1}|x_{0})p_{r}(x_{0}) \newline
\end{aligned}
$$

两种形式的联合概率分布表示的是同一个联合概率分布，因此是完全相等的，故此可以推导出：

$$
\begin{aligned}
p_{r}(x_{t-1}|x_{t},x_{0})
&= \frac{p_{r}(x_{t},x_{t-1},x_{0})}{p_{r}(x_{t}|x_{0})p_{r}(x_{0})} \newline
&= \frac{p_{r}(x_{t}|x_{t-1},x_{0})p_{r}(x_{t-1}|x_{0})p_{r}(x_{0})}{p_{r}(x_{t}|x_{0})p_{r}(x_{0})} \newline
&= \frac{p_{r}(x_{t}|x_{t-1},x_{0})p_{r}(x_{t-1}|x_{0})}{p_{r}(x_{t}|x_{0})} \newline
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
&= p_{r}(x_{1}, x_{2}|x_{3})p_{r}(x_{3}) \newline
&= p_{r}(x_{1}|x_{2}, x_{3})p_{r}(x_{2}|x_{3})p_{r}(x_{3}) \newline
&= p_{r}(x_{1}|x_{2})p_{r}(x_{2}|x_{3})p_{r}(x_{3}) & \text{ ;Conditional Independence} \newline
\end{aligned}
$$

<span style="color:GoldenRod">**Note that conditional independence relations are always symmetric**</span>

$$
\begin{aligned}
p_{r}(x_{3},x_{2},x_{1})  
&= p_{r}(x_{3}, x_{2}|x_{1})p_{r}(x_{1}) \newline
&= p_{r}(x_{3}|x_{2}, x_{1})p_{r}(x_{2}|x_{1})p_{r}(x_{1}) \newline
&= p_{r}(x_{3}|x_{2})p_{r}(x_{2}|x_{1})p_{r}(x_{1}) & \text{ ;Conditional Independence}\newline
\end{aligned}
$$

条件独立关系意味着对条件分布以一定的方式进行因子分解(并因此视为冗余)，这种冗余意味着可用更少量的参数来描述数据的概率分布，同时对含有大规模参数的模型更加易于处理。计算机视觉中常引入图模型来表示这种条件独立关系，如有向图模型(即贝叶斯网络)，链式模型(即马尔科夫链)和树模型。参考书籍 "Computer Vision: Models, Learning, and Inference".

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
    padding: 2px;">Fig. 6.  Special cases of expectation. (Image source from the book, "Computer Vision: Models, Learning, and Inference")</div>
</center>

### **8. 重参数技巧 Reparameterization Trick**

如果从高斯分布中随机采样一个样本，这个过程不可微分的，即无法反传梯度的。通过**重参数 (reparameterization) 技巧**[<sup>[17]</sup>](#refer-17)来使其可微。最通常的做法是把这种随机性通过一个独立的随机变量 $\epsilon$ 进行转移。举个例子，如果要从高斯分布 $z\sim \mathcal{N}\left ( z;\mu_{\theta},\sigma^{2}_{\theta} I\right ) $ 中采样一个 z，可以写成:

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
&= \sqrt{\frac{\tau}{2\pi}} e^{\frac{-\tau (x - \mu)^{2}}{2}} \newline
&= \sqrt{\frac{1}{2\pi \sigma^{2}}} e^{- \frac{(x - \mu)^{2}}{2 \sigma^{2}}} \newline
&= \sqrt{\frac{1}{2\pi \sigma^{2}}} exp(- \frac{(x - \mu)^{2}}{2 \sigma^{2}}) \newline
&\propto exp(- \frac{(x - \mu)^{2}}{2 \sigma^{2}}) \newline
&\propto exp(\frac{(x - \mu)^{2}}{\sigma^{2}}) \newline
&\propto \frac{x^{2} - 2 \mu x + \mu^{2}}{\sigma^{2}} \newline
&= \frac{1}{\sigma^{2}}x^{2} -  \frac{2 \mu}{\sigma^{2}}x + \frac{\mu^{2}}{\sigma^{2}}\newline
\end{aligned}
$$

### **10. 信息论和概率模型**

**1. 拟合概率模型**

Fitting probability models

- **最大似然法 Maximum likelihood, ML**
- **最大后验法 Maximum a posteriori, MAP**
- <span style="color:red">**贝叶斯方法 the Bayesian approach**</span>

**maximum likelihood, ML**

最大似然 ML 用来求数据 $x_{i}$ , $[i=1, 2, 3, \cdots, I]$ 最有可能的参数集合 $\mathbf{\hat{\theta}}$ 。为了计算在单个数据点 $x_{i}$ 处的似然函数 $P_{r}(x_{i} \mid \mathbf{\theta})$ , 只需要简单估计在 $x_{i}$ 处的概率密度函数 (probability density function, pdf) 。假设每一个数据点都是从分布中独立采样，点的集合的似然函数 $P_{r}(x_{1\cdots}I \mid \mathbf{\theta})$ 就是独立似然的乘积。因此，参数的最大似然估计如下：

$$
\begin{aligned}
\mathbf{\hat{\theta}}
&=\underset{\theta}{argmax}[P_{r}(x_{1\cdots}I \mid \mathbf{\theta})] \newline
&=\underset{\theta}{argmax}[\prod_{i=1}^{I} P_{r}(x_{i} \mid \mathbf{\theta})] \newline
\newline
& \text{其中，$\underset{\theta}{argmax} f[\theta]$ 返回使得 $f[\theta]$ 最大化的 $\theta$ 数值} \newline
\end{aligned}
$$

为了估计新的数据点 $x^{\ast}$ 的概率分布，其中计算 $x^{\ast}$ 属于拟合模型的概率，用最大似然拟合参数 $\mathbf{\hat{\theta}}$ 简单估计概率密度函数 $P_{r}(x^{\ast} \mid \mathbf{\hat{\theta}})$ 即可。

**Maximum a posteriori, MAP**

最大后验拟合 MAP 中，引入参数 $\theta$ 的先验 (prior) 信息。 From previous experience we may know something about the possible parameter values. For example, in a time-sequence  the values of the parameters at time $t$ tell us a lot about the possible values at time $t + 1$. 而且这个先验信息可以被先验分布所编码。

最大后验估计就是最大化参数的后验概率 $P_{r}(\mathbf{\theta} \mid x_{1 \cdots I})$

$$
\begin{aligned}
\mathbf{\hat{\theta}}
&=\underset{\theta}{argmax}[P_{r}(\mathbf{\theta} \mid x_{1 \cdots I})] \newline
&=\underset{\theta}{argmax}[\frac{P_{r}(x_{1 \cdots I} \mid \mathbf{\theta}) P_{r}(\mathbf{\theta})}{P_{r}(x_{1 \cdots I})}] \newline
&=\underset{\theta}{argmax}[\frac{\prod_{i=1}^{I} P_{r}(x_{i} \mid \mathbf{\theta}) P_{r}(\mathbf{\theta})}{P_{r}(x_{i \cdots I})}] \newline
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
&= \int P_{r}(x^{\ast} \mid \mathbf{\theta}) \delta[\mathbf{\theta} - \mathbf{\hat{\theta}}] \mathrm{d}\theta \newline
&= P_{r}(x^{\ast} \mid \mathbf{\theta}) \newline
& \text{where $\int \delta[\mathbf{\theta} - \mathbf{\hat{\theta}}] \mathrm{d}\theta = 1$}
\end{aligned}
$$

which is exactly the calculation we originally prescribed:  we simply evaluate the probability of the data under the  model with the estimated parameters. 可以估计数据在参数模型下的概率。

**2. 信息熵，交叉熵和KL散度**

**信息论与编码**

> Information entropy; Entropy; Shannon entropy; Cross entropy; Relative entropy; Kullback–Leibler divergence ([KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence))

- 信息量：指用一个信息所需要的编码长度来定义，而一个信息的便阿门长度与其出现的概率呈负相关。

> 其实也就可以说一件事情发生概率很小的话，那么当其发生的时候所代表的信息量也更大

$$I = log_{2}(\frac{1}{p(x)}) = -log_{2}(p(x))$$

- 信息熵 (香农熵)：信息熵则代表一个分布的信息量,或者编码的平均长度。

> 信息熵度量的是随机变量 (<span style="color:green">**离散和连续**</span>) 或整个系统的不确定性，熵越大，随机变量或系统的不确定性就越大，也就是信息量的一个总期望值也叫均值；根据真实分布，能够找到一个最优策略，以最小的代价消除系统的不确定性，而这个代价大小就是信息熵；信息熵衡量了系统的不确定性，而要消除这个不确定性，所要付出的最小努力(猜题次数、编码长度等)的大小就是信息熵。

$$
\begin{aligned}
H(p) 
&= \sum_{x}p(x) log_{2}(\frac{1}{p(x)}) \newline
&= -\sum_{x}p(x) log_{2}(p(x)) \newline
&= E_{x \sim p(x)}[-log_{2}p(x)] \newline
&= -\sum_{i=1}^{n} p(x) log_{2}(p(x)) \newline
\end{aligned}
$$

$$
\begin{aligned}
H(p) 
&= \int p(x) log_{2}(\frac{1}{p(x)}) \mathrm{d}x \newline
&= -\int p(x) log_{2}(p(x)) \mathrm{d}x \newline
&= E_{x \sim p(x)}[-log_{2}p(x)] \newline
&= -\int_{-\infty}^{\infty} p(x) log_{2}(p(x)) \newline
\end{aligned}
$$

在信息论和编码中 log 的底数是 2，但一般在神经网络中，默认以 e (<span style="color:DeepPink">e = 2.73 magic number, such as 42 </span>) 为底，这样算出来的香农信息量虽然不是最小的可用于完整表示实践的比特数，但对于信息熵的含义来说是区别不大的，只要这个底数大于 1，就可以表达信息熵的大小。

- 交叉熵: 本质上可以视为用一个猜测(预估)的分布的编码方式去编码其真实的分布, 得到的平均编码长度或者信息量。

> 交叉熵，用来度量在给定的真实分布 $p$ 下，使用非真实分布 $q$ 所指定的策略消除系统的不确定性所需要付出的努力的大小; 交叉熵越低，这个策略就越好，最低的交叉熵也就是使用了真实分布所计算出来的信息熵，故此 “交叉熵 = 信息熵” ；这也是为什么在机器学习中的分类算法中，总是最小化交叉熵，因为交叉熵越低，就证明由算法所产生的策略最接近最优策略，也间接证明算法所算出的非真实分布越接近真实分布。

$$
\begin{aligned}
H_p(q) 
&= \sum_{x}p(x)log(\frac{1}{q(x)}) \newline
&= - \sum_{x}p(x)log(q(x)) \newline
&= - \sum_{i=1}^{n} p(x_{i})log(q(x_{i})) \newline
&= - E_{x \sim p(x)} log(q(x)) & \text{; 离散和连续}\newline
&= - \int_{-\infty}^{\infty} p(x)log(q(x)) \mathrm{d}x \newline
\end{aligned}
$$

> 相对熵 (KL 散度)：KL散度或距离是度量两个分布的差异，KL 距离一般用 $D(p||q)$ 或 $D_{p}(q)$ 称之为 $p$ 对 $q$ 的相对熵。

$$
\begin{aligned}
D(p || q) = D_p(q) 
&= H_{p}(q) - H(P) & \text{; cross entropy minus information entropy} \newline
&= \sum_{x}p(x)log(\frac{1}{q(x)}) - [- \sum_{i=1}^{n} p(x) log(p(x))] \newline
&= \sum_{x}p(x)log(\frac{1}{q(x)}) + \sum_{i=1}^{n} p(x) log(p(x)) \newline
&= \sum_{x}p(x) \log{\frac{p(x)}{q(x)}} \newline
&= E_{x \sim p(x)} [\log{\frac{p(x)}{q(x)}}] \newline
&= D_{KL}(p || q) \newline
\end{aligned}
$$

在 $p$ and $q$ 满足可交换的条件下，交叉熵和 KL 散度相等。还有联合信息熵；条件信息熵；自信息；互信息等针对不同用途的度量形式。

> [Difference of KL divergence and cross entropy](https://stats.stackexchange.com/questions/357963/what-is-the-difference-cross-entropy-and-kl-divergence)

利用 詹森不等式 ([Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)) 可以推导出 KL 散度的非负性：

> 对数的期望大于等于期望的对数 $\Phi(E[X]) \le E[\Phi(X)]$

$$
\begin{aligned}
D_{KL}(p || q)
&= \sum_{x}p(x)log(\frac{p(x)}{q(x)}) \newline
&= - \sum_{x}p(x)log(\frac{q(x)}{p(x)}) \newline
&= - \int_{-\infty}^{\infty} p(x)\mathrm{d}x log(\frac{q(x)}{p(x)}) \newline
&= - E_{x \sim p(x)}[log(\frac{q(x)}{p(x)})] \newline
&\le - log E_{x \sim p(x)}[\frac{q(x)}{p(x)}] \newline
&\ge log E_{x \sim p(x)}[\frac{q(x)}{p(x)}] \newline
&= log \sum_{x} p(x) \frac{q(x)}{p(x)} \newline
&= log \sum_{x} q(x) \newline
&= log(1) \newline
&= 0 \newline
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
    padding: 2px;">Fig. 7. The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source from DDPM'2020)</div>
</center>
<!-- ![generative-overview](./Images/DDPM.png) -->

<center class="half">
    <img src="./images/forward_diffusion_s_curve.gif"><img src="./images/reverse_diffusion_s_curve.gif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 8. The S curve distribution from forward(left) to reverse(right) diffusion process.</div>
</center>

<center class="half">
    <img src="./images/forward_diffusion_swiss_roll.gif"><img src="./images/reverse_diffusion_swiss_roll.gif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 9. The two dimensions swiss roll distribution from forward(lefet) to reverse(right) diffusion process.</div>
</center>

#### **Forward Diffusion Process**

1. forward diffusion process

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
&= \sqrt{ \alpha_{t} } x_{t-1} + \sqrt{1 - \alpha_{t}} z_{1} \newline
&= \sqrt{\alpha_{t}} (\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}z_{2}) + \sqrt{1 - \alpha_{t}} z_{1} & \text{;不断利用重参数技巧进行采样} \newline
&= \sqrt{\alpha_{t} \alpha_{t-1}}x_{t-2} + {\color{red} \sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2}} + {\color{blue} \sqrt{1 - \alpha_{t}} z_{1}} \newline
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
&= x_{t}\newline
&= \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + {\color{red}\sqrt{\alpha_{t}(1 - \alpha_{t-1})}z_{2}} + {\color{blue}\sqrt{1 - \alpha_{t}} z_{1}} \newline
&= \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + \color{green}\sqrt{1- \alpha_{t}\alpha_{t-1}}\bar{z_{2}} \newline
&= \cdots & \text{;相同的方式不断迭代}\newline
&= \sqrt{\alpha_{t}\alpha_{t-2} \cdots \alpha_{t=T-1}}x_{1} + \color{green}\sqrt{1- \alpha_{t}\alpha_{t-1} \cdots \alpha_{t=T-1}}\bar{z_{T}} \newline
&= \sqrt{\alpha_{t}\alpha_{t-1} \cdots \alpha_{t=T}}x_{0} + \color{green}\sqrt{1- \alpha_{t}\alpha_{t-1} \cdots \alpha_{t=T}}\bar{z_{T+1}} \newline
&= \sqrt{\prod_{i=1}^{T} \alpha_{i}}x_{0} + \color{green}\sqrt{1- \prod_{i=1}^{T} \alpha_{i}}\bar{z_{T+1}} \newline
\end{aligned}
$$

令 $ \bar{\alpha}_{t} = \prod_{i=1}^{T} \alpha_{i}$ , 则上式子可以化简为：

$$x_{t} = \sqrt{\bar{\alpha}_{t}}x_{0} + \color{green}\sqrt{1 - \bar{\alpha}_{t}}\bar{z}_{t} ; \color{red}\bar{z}_{t} \in \mathcal{N}(0, I)$$

$$q(x_{t} \mid x_{0}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

故此总结一下前向扩散过程的概率分布满足一下式子：

$$
\begin{aligned}
q(x_{t} \mid x_{t-1})
&\sim \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t-1}, \beta_{t}I) \newline
&= \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t-1}, \beta_{t}I) \bar{z}_{t} \newline
\end{aligned}
$$

$$
\begin{aligned}
q(x_{t} \mid x_{0})
&\sim \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I) \newline
&= \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I) \bar{z}_{t} \newline
\end{aligned}
$$

这样根据前向扩散过程的要求，最终迭代 $T$ 次后，$x_{T}$ 变成一个标准高斯分布，则可以计算出迭代次数 $T$ 的具体数值(e.g. T=1000)：

$$
sub. to \left\{
\begin{aligned}
&\sqrt{\bar{\alpha_{t}}}x_{0} = 0 \newline
&1 - \bar{\alpha_{t}} = 1 \newline
\end{aligned}
\right.
$$

因此前向扩散过程中的迭代步数是有一个有限的可解析的数值，$t \in (0, T) $ 进行采样得到具体的数值，the sample-step schedule is different, PDM and DDPM paper is uniform schedule(均匀采样); but IDDPM paper is simple importance sampling technique(基于 loss 进行重要性采样)

> <span style="color:DarkOrange"> Note: Imporved diffusion Code Implementation with OpenAI. </span>

#### **Reverse Diffusion Process**

如果说前向扩散过程 (forward process)是加噪的过程，那么逆向扩散过程 (reverse process) 就是 diffusion models 的去噪推断过程。如果能够逐步得到逆转后的分布，就可以从完全的标准高斯分布  还采样从而复原出原始分布。 Feller William 在 1949 年的文献中证明了如果 forward process $q(x_{t} \mid x_{t-1})$ 满足高斯分布且 $\beta_{t}$ 足够小，reverse process $q(x_{t-1} \mid x_{t})$ 仍然是一个高斯分布。然而这个逆向分布无法进行简单推断计算出解析式，因此使用深度学习模型 (Neural Networks, NN) 去预测或者拟合这样的一个逆向的分布。

因此假设逆向过程的分布 $q(x_{t-1} \mid x_{t}) \sim \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))$ , 利用 NN 拟合 $\mu_{\theta}$ 和 $\Sigma_{\theta}$ , 均值和方差都是关于 $(x_{t}, t)$ 的仿射变换函数

$$ q(x_{t-1} \mid x_{t}) = p_{\theta}(x_{t-1} \mid x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t)) $$

and the joint probability dist. as follow:

$$ p_{\theta}(X_{0:T}) = p(X_{T}) \prod_{t=1}^{T}(x_{t-1} \mid x_{t}) $$

> <span style="color:Gold">虽然无法计算出 $q(x_{t-1} \mid x_{t})$ ,但是可以计算出逆向扩散过程的后验概率分布 $q(x_{t-1} \mid x_{t}, x_{0})$ . </span> 联合概率分布可以分解为条件概率分布的乘积形式

$$
\begin{aligned}
q(x_{t-1}, x_{t}, x_{0})
&= q(x_{t-1}, x_{t} \mid x_{0})q(x_{0}) \newline
&= q(x_{t-1} \mid x_{t}, x_{0})q(x_{t} \mid x_{0})q(x_{0}) \newline
\end{aligned}
$$

基于 diffusion process (forward or reverse) 都是马尔可夫过程 (Markov chain) ，在给定 $x_{0}$ 条件下，$x_{t-1}$ 和 $x_{t}$ 条件独立，则利用对称性，$q(x_{t-1}, x_{t}, x_{0})$ 联合概率分布有如下相同等式

$$
\begin{aligned}
q(x_{t-1}, x_{t}, x_{0})
&= q(x_{t}, x_{t-1} \mid x_{0})q(x_{0}) \newline
&= q(x_{t} \mid x_{t-1}, x_{0})q(x_{t-1} \mid x_{0})q(x_{0}) \newline
\end{aligned}
$$

那么逆向扩散过程的后验概率分布如下推导：

$$
\begin{aligned}
q(x_{t-1} \mid x_{t}, x_{0})
&=\frac{q(x_{t-1}, x_{t}, x_{0})}{q(x_{t} \mid x_{0}) q(x_{0})} \newline
&=\frac{q(x_{t} \mid x_{t-1}, x_{0})q(x_{t-1} \mid x_{0})q(x_{0})}{q(x_{t} \mid x_{0}) q(x_{0})} \newline
&=\frac{q(x_{t} \mid x_{t-1}, x_{0})q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} \newline
&=q(x_{t} \mid x_{t-1}, x_{0}) \frac{q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} \newline
&=\color{red} q(x_{t} \mid x_{t-1}) \frac{q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} & \text{; Markov chain}\newline
\end{aligned} \newline
$$

$$ q(x_{t} \mid x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}}x_{t-1}, \beta_{t}I) $$

$$ q(x_{t} \mid x_{0}) = \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I) $$

$$ q(x_{t-1} \mid x_{0}) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_{0}, (1 - \bar{\alpha}_{t-1})I) $$

将高斯前向扩散过程带入后验分布式子中，可以化简如下：

$$
\begin{aligned}
q(x_{t-1} \mid x_{t}, x_{0})

&=\color{red} q(x_{t} \mid x_{t-1}) \frac{q(x_{t-1} \mid x_{0})}{q(x_{t} \mid x_{0})} \newline

&\sim \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}}x_{t-1}, \beta_{t}I) \frac{\mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I)}{\mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_{0}, (1 - \bar{\alpha}_{t-1})I)} \newline

&\propto exp(-\frac{1}{2}[\color{green} \frac{(x_{t} - \sqrt{1-\beta_{t}}x_{t-1})^{2}}{\beta_{t}} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_{0})^{2}}{1-\bar{\alpha}_{t-1}} - \frac{(x_{t} - \sqrt{\bar{\alpha}_{t}}x_{0})^{2}}{1-\bar{\alpha}_{t}}]) & \text{; 相同底数的幂函数相乘，指数相加即可; 将高斯函数写成指数表示的形式} \newline

&\propto exp(\frac{x_{t}^{2} - 2\sqrt{1-\beta_{t}}x_{t-1}x_{t} + (1-\beta_{t})(x_{t-1})^{2}}{\beta_{t}} + \frac{x_{t-1}^{2} - 2\sqrt{\bar{\alpha}_{t-1}}x_{0}x_{t-1} + \bar{\alpha}_{t-1}(x_{0})^{2}}{1-\bar{\alpha}_{t-1}} - \frac{x_{t}^{2} - 2\sqrt{\bar{\alpha}_{t}}x_{0}x_{t} + \bar{\alpha}_{t}(x_{0})^{2}}{1-\bar{\alpha}_{t}}) & \text{; 将分子平方展开} \newline

&\propto \frac{x_{t}^{2} - 2\sqrt{1-\beta_{t}}x_{t-1}x_{t} + (1-\beta_{t})(x_{t-1})^{2}}{\beta_{t}} + \frac{x_{t-1}^{2} - 2\sqrt{\bar{\alpha}_{t-1}}x_{0}x_{t-1} + \bar{\alpha}_{t-1}(x_{0})^{2}}{1-\bar{\alpha}_{t-1}} - \frac{x_{t}^{2} - 2\sqrt{\bar{\alpha}_{t}}x_{0}x_{t} + \bar{\alpha}_{t}(x_{0})^{2}}{1-\bar{\alpha}_{t}} & \text{; 以 $x_{t-1}$ 为变量进行合并} \newline

&\propto (\frac{1-\beta_{t}}{\beta_{t}} + \frac{1}{1-\bar{\alpha}_{t-1}})x_{t-1}^{2} - (\frac{2\sqrt{1-\beta_{t}}x_{t}}{\beta_{t}} + \frac{2\sqrt{\bar{\alpha}_{t-1}}x_{0}}{1-\bar{\alpha}_{t-1}})x_{t-1} + C(x_{t}, x_{0}) & \text{; 其中 $C$ 与 $x_{t-1}$ 无关的常量} \newline
\end{aligned} \newline
$$

逆向扩散过程的后验概率分布依然满足高斯分布，假设服从以下分布：

$$
\begin{aligned}
q(x_{t-1} \mid x_{t}, x_{0})
&\sim \mathcal{N}(x_{t-1}; \widetilde{\mu}(x_{t}, x_{0}), \widetilde{\beta}_{t}I) \newline
&\sim exp(x_{t-1}; \widetilde{\mu}(x_{t}, x_{0}), \widetilde{\beta}_{t}I) \newline
&\propto \color{Aquamarine} (\frac{1}{\widetilde{\beta}_{t}})x_{t-1}^{2} - (\frac{2\widetilde{\mu}}{\widetilde{\beta}_{t}})x_{t-1} + \frac{\widetilde{\mu}^{2}}{\widetilde{\beta}_{t}} \newline
\end{aligned} \newline
$$

根据以上关于 $q(x_{t-1} \mid x_{t}, x_{0})$ 的两个式子，可以计算出逆向扩散过程中的真实的均值和方差估计 (用于训练 NN 的监督 GT)：

$$
\begin{aligned}
\frac{1}{\widetilde{\beta}_{t}}
&= \frac{1-\beta_{t}}{\beta_{t}} + \frac{1}{1-\bar{\alpha}_{t-1}} \newline
&= \frac{(1-\beta_{t})(1-\bar{\alpha}_{t-1})+\beta_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})} \newline

\Rightarrow \widetilde{\beta}_{t} &= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{(1-\beta_{t})(1-\bar{\alpha}_{t-1})+\beta_{t}} \newline
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}(1-\bar{\alpha}_{t-1})+\beta_{t}} & \text{; $\beta_{t}=1-\alpha_{t}$} \newline 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}-\alpha_{t}\bar{\alpha}_{t-1}+\beta_{t}} \newline 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}-\bar{\alpha}_{t}+\beta_{t}} \newline 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}-\bar{\alpha}_{t}+(1-\alpha_{t})} \newline 
&= \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} \newline 
&= \color{red} \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} & \text{; DDPM paper} \newline 
\end{aligned} \newline
$$

$$
\begin{aligned}
\frac{2\widetilde{\mu}}{\widetilde{\beta}_{t}}
&= \frac{2\sqrt{1-\beta_{t}}x_{t}}{\beta_{t}} + \frac{2\sqrt{\bar{\alpha}_{t-1}}x_{0}}{1-\bar{\alpha}_{t-1}} \newline
&= 2\frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}x_{t} + \beta_{t}\sqrt{\bar{\alpha}_{t-1}}x_{0}}{\beta_{t}(1-\bar{\alpha}_{t-1})} \newline

\Rightarrow \widetilde{\mu}_{t}(x_{t},x_{0}) &= \frac{((1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}x_{t} + \beta_{t}\sqrt{\bar{\alpha}_{t-1}}x_{0})\widetilde{\beta}_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})} \newline

&= \frac{\beta_{t}\sqrt{\bar{\alpha}_{t-1}}\widetilde{\beta}_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{0} +  \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}\widetilde{\beta}_{t}}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{t}\newline

&= \frac{\beta_{t}\sqrt{\bar{\alpha}_{t-1}}(\color{red} \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t})}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{0} +  \frac{(1-\bar{\alpha}_{t-1})\sqrt{1-\beta_{t}}(\color{red} \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t})}{\beta_{t}(1-\bar{\alpha}_{t-1})}x_{t}\newline

&= \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} x_{0} + \frac{\sqrt{1-\beta_{t}}}{\beta_{t}} \cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} x_{t} & \text{; $\alpha_{t}=1-\beta_{t}$} \newline

&= \color{green} \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}x_{0} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} & \text{; DDPM paper} \newline
\end{aligned}  \newline
$$

前向扩散过程中，任意时刻的 $x_{t}$ 与 $x_{0}$ 关系如下：

$$q(x_{t} \mid x_{0})
= \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})I) \bar{z}_{t} $$

$$
\begin{aligned}
x_{t} &= \sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t} \newline

\Rightarrow x_{0} &= \frac{1}{\sqrt{\bar{\alpha}_{t}}}(x_{t} - \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t})
\end{aligned}  \newline
$$

将该关于 $x_{0}$ 的式子代入上式关于均值 $\widetilde{\mu}_{t}(x_{t},x_{0})$ 中可以推导如下：

$$
\begin{aligned}
\widetilde{\mu}_{t}(x_{t},x_{0}) 
&= \color{green} \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}x_{0} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} \newline

&= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}(\frac{1}{\sqrt{\bar{\alpha}_{t}}}(x_{t} - \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t})) + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} \newline

&= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}} \cdot \frac{1}{\sqrt{\bar{\alpha}_{t}}}(x_{t} - \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{t}) + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} \newline

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}x_{t} + \frac{\alpha_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}\bar{z}_{t}\right] & \text{; $\sqrt{\bar{\alpha}_{t}} = \sqrt{\alpha_{t}\bar{\alpha}_{t-1}}$}\newline

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}x_{t} + \frac{\alpha_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t} - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}\bar{z}_{t}\right] \newline

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{\beta_{t} + \alpha_{t} - \alpha_{t}\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}x_{t} - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})\sqrt{\bar{\alpha}_{t-1}}}\bar{z}_{t}\right] \newline

&= \frac{1}{\sqrt{\alpha_{t}}} \left[ \frac{(1-\alpha_{t}) + \alpha_{t} - \alpha_{t}\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}x_{t} - \frac{\beta_{t}(\sqrt{1 - \bar{\alpha}_{t}})}{(1-\bar{\alpha}_{t})}\bar{z}_{t}\right] \newline

&= \color{Cyan} \frac{1}{\sqrt{\alpha_{t}}} (x_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\bar{z}_{t} ) \newline
\end{aligned}  \newline
$$

**Inference Phase of DDPM**

$$p_{\theta}(x_{t-1} \mid x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t)) $$

根据该式子，可以理解 DDPM paper 的核心思想，训练 NN 网络去预测 $\bar{z}_{t}$ , 用于去噪 (denoising DPM), NN 网络预测的结果为 ${z}_{\theta}(x_{t}, t)$ , 则采样时候的均值可以直接计算得到如下(DDPM paper 中的损失函数为 $\mathcal{L}_{simple}(\theta)$)：

$$\mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{\alpha_{t}}} (x_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}{z}_{\theta}(x_{t}, t) )$$

DDPM paper 中对于方差的策略，直接使用逆向扩散过程推导的解析结果 $\widetilde{\beta}_{t}$ , 而且实验结果显示使用前向过程的方差数值和使用逆向过程的后验方差数值，最终的实验结果近视；不需要训练的策略，如下式子：

$$
\begin{aligned}
\Sigma_{\theta}(x_{t}, t) 
&= \widetilde{\beta}_{t} & \text{; reverse process variance} \newline
&= \beta_{t} & \text{; forward process variance} \newline
&= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t} & \text{; reverse process posterior variance} \newline
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
&= -\log{p_{\theta}(x_{0})} \newline
&\le -\log{p_{\theta}(x_{0})} + D_{KL}(q(x_{1:T} \mid x_{0}) || p_{\theta}(x_{1:T} \mid x_{0})) & \text{; No.2} \newline
&= -\log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{1:T} \mid x_{0})}}] & \text{; No.3} \newline
&= -\log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T}) / p_{\theta}(x_{0})}}] & \text{; No.4} \newline
&= -\log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} + \log{p_{\theta}(x_{0})}] & \text{; No.5} \newline
&= -\log{p_{\theta}(x_{0})} + \log{p_{\theta}(x_{0})} + E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ] & \text{; No.6} \newline
&= E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ] & \text{; No.7} \newline
\end{aligned}
$$

**Note that：**

- 第 2 行式子成立的理由: KL 散度的非负性
- 第 3 行式子成立的理由: KL 散度的定义计算公式
- 第 4 行式子成立的理由: 条件概率的定义计算公式
- 第 5 行式子成立的理由: 对数函数的性质
- 第 6 行式子成立的理由: $p_{\theta}(x_{0})$ 与求  $q(x_{1:T} \mid x_{0})$ 的期望无关

对上式子两边取期望 $E_{q(x_{0})}$, 即为类似计算 VAE 中的变分下限 ([Evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound))：

$$
\begin{aligned}
\mathcal{L}_{VLB} 
&\ge E_{q(x_{0})}[-\log{p_{\theta}(x_{0})}] \newline
&= E_{q(x_{0})}[E_{q(x_{1:T} \mid x_{0})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ]] \newline
&= E_{q(x_{0:T})}[\log{\frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}} ] & \text{; DPM paper} \newline
\end{aligned}
$$

which has a lower bound provided by [Jense's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) ; 利用 Jense's inequality 将积分的凸函数的值与凸函数的积分联系起来，提供下限；计算期望对于连续变量而言就是计算积分；这样十分类似 VAE 中的推导形式, 从而可以优化交叉熵对目标分布进行学习：

$$
\begin{aligned}
\mathcal{L} = L_\text{CE}
&= \mathbb{E}_{q(x_{0})}[-\log{p_{\theta}(x_{0})}] \newline
&= - \mathbb{E}_{q(\mathbf{x}_0)} [ \log\Big({ p_\theta(\mathbf{x}_0) \int p_\theta(\mathbf{x}_{1:T}) d\mathbf{x}_{1:T}}\Big)] & \text{; No.2} \newline

&= - \mathbb{E}_{q(\mathbf{x}_0)} [ \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) ] \newline

&= - \mathbb{E}_{q(\mathbf{x}_0)} [ \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) ] \newline

&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \newline

&\leq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} & \text{; Jense's inequality} \newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] \newline

&= \color{red} \mathcal{L}_{VLB} \newline
\end{aligned}
$$

**Note that：**

- 第 2 行式子成立的理由: $p_{\theta}(x_{0})$ 与求  $q(x_{1:T} \mid x_{0})$ 的期望无关, 而且积分结果为 1

进一步对 $\mathcal{L}_{VLB}$ 推导，根据 Improved DDPM paper 中的形式：

$$
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] & \text{; No.2} \newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] & \text{; No.3} \newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.4}\newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \color{red}{ \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} } \Big ) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.5}\newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.6}\newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log \frac{\prod_{i=2}^{T} q(\mathbf{x}_{i} \vert \mathbf{x}_0)}{\prod_{i=2}^{T} q(\mathbf{x}_{i-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.7}\newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] & \text{; No.8}\newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] & \text{; No.9} \newline

&=Exception || KL ? \newline

&= \mathbb{E}_{q(\mathbf{x}_{0:T})} [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ] & \text{; No.10} \newline
\end{aligned}
$$

**Note that：**

- 第 2 行式子: convert Joint dist. into conditional dist. prod.
- 第 3 行式子: 对数函数的性质
- 第 5 行式子: reverse process formula for $x_{t}$ and $x_{0}$
- 第 8 行式子: 对数函数性质 & 累乘形式下分子分母相同项消除
- 第 9 行式子: 根据对数性质进行重新组合排列每一项
- how and why line No.9 $\longrightarrow$ line No.10；也知答案，逆推过程，凑一个期望，即对数里面分子的一个积分

**recall that: where the expectation **line No.9** is over a distribution $\bar{q}(x_{t-1})$ that is independent from the variable (namely $x_{t-1}$).** 

$$D_{\text{KL}}(q(x) || p(x)) = \mathbb{E}_{q(x)} [\log q(x) / p(x)]$$

$$
\begin{aligned}
gereral_{line-9}
&=\mathbb{E}_{q(x_{0:T})} \left[ \log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \newline

&=~ \mathbb{E}_{{\color{red}q(x_{t-1}|x_t, x_0)}{\color{green}q(x_t,x_0)q(x_{1:t-2,t+1:T}|x_{t-1},x_t,x_0)}} \left[\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \newline

&=~ \mathbb{E}_{{\color{green}\bar{q}(x_{t-1})}} \left[ \mathbb{E}_{{\color{red}q(x_{t-1}|x_t, x_0)}} \left[\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right] \right]  \newline

&=~ \mathbb{E}_{\bar{q}(x_{t-1})} \left[D_{\text{KL}}(q(x_{t-1}|x_t, x_0)|| p_\theta(x_{t-1}|x_t)) \right] \newline

&=~ D_{\text{KL}}(q(x_{t-1}|x_t, x_0)|| p_\theta(x_{t-1}|x_t))
\end{aligned}
$$

这样就得到了 Improved DDPM paper 的优化目标

Let’s label each component in the variational lower bound loss separately:

$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \newline
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \newline
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \newline
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$

> Every KL term in $L_\text{VLB}$  (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in [closed form](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions).  $L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $\mathbf{x}_T$ is a Gaussian noise. [Ho et al. 2020](https://arxiv.org/abs/2006.11239) models $L_0$ using a separate discrete decoder derived from $\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \boldsymbol{\Sigma}_\theta(\mathbf{x}_1, 1))$ . (DDPM paper 中对逆向扩散过程中最后一步从噪声变为原始数据的处理)

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
\begin{align}
\mathbf{x}_{i+1} \gets \mathbf{x}_i + \epsilon \nabla_\mathbf{x} \log p(\mathbf{x}) + \sqrt{2\epsilon}~ \mathbf{z}_i, \quad i=0,1,\cdots, K,
\end{align}
$$

<center class="half">
    <img src="./images/Langevin_dynamics.gif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 10. Annealed Langevin dynamics combine a sequence of Langevin chains with gradually decreasing noise scales. (Image source from Yang Song.)</div>
</center>

<center class="half">
    <img src="./images/celeba_large.gif"><img src="./images/cifar10_large.gif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 11. Annealed Langevin dynamics for the Noise Conditional Score Network (NCSN) model trained on CelebA (left) and CIFAR-10 (right). We can start from unstructured noise, modify images according to the scores, and generate nice samples. The method achieved state-of-the-art Inception score on CIFAR-10 at its time. (Image source from Yang Song.)</div>
</center>

## Code Implementation 代码实现

<center class="center">
    <img src="./images/DDPM_algo.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 12. The training and sampling algorithms in DDPM (Image source: DDPM NeurIPS'2020)</div>
</center>

```python
# Training for Denoising Diffusion Probabilistic Models
# source code from DDPM original github: 
# github: https://github.com/hojonathanho/diffusion
def train_fn(self, x, y):
    B, H, W, C = x.shape
    if self.randflip:
        x = tf.image.random_flip_left_right(x)
        assert x.shape == [B, H, W, C]

    t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)
    losses = self.diffusion.p_losses(denoise_fn=
        functools.partial(self._denoise, y=y, dropout=self.dropout), x_start=x, t=t)

    assert losses.shape == t.shape == [B]
    return {'loss': tf.reduce_mean(losses)}

def samples_fn(self, dummy_noise, y):
    return {'samples': 
        self.diffusion.p_sample_loop(
            denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
            shape=dummy_noise.shape.as_list(),
            noise_fn=tf.random_normal)}
```

```python
# Sampleing for Denoising Diffusion Probabilistic Models
# source code from DDPM original github: 
# github: https://github.com/hojonathanho/diffusion
def samples_fn_denoising_trajectory(self, dummy_noise, y, repeat_noise_steps=0):
    times, imgs = self.diffusion.p_sample_loop_trajectory(
        denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
        shape=dummy_noise.shape.as_list(),
        noise_fn=tf.random_normal,
        repeat_noise_steps=repeat_noise_steps)

return {'samples': imgs[-1],
    'denoising_trajectory_times': times,
    'denoising_trajectory_images': imgs}
```

<center class="center">
    <img src="./images/DPM_fig.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 13. The training and sampling algorithms in DDPM. (Image source: DPM ICML'2015)</div>
</center>

<center class="half">
    <img src="./images/forward_reverse_diffusion_s_curve.gif"><img src="./images/forward_reverse_diffusion_swiss_roll.gif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 14. The S curve(left) and a two dimensions swiss roll(right) distribution from forward to reverse diffusion process.</div>
</center>

<center class="half">
    <img src="./images/DDPM_Code.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 15. Code Diagram of DDPM and Improved DDPM for Diffusion Models.</div>
</center>
<!-- ![DDPM Code](./Images/DDPM_Code.png) -->

<center class="half">
    <img src="./images/UNet_architecture.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 16. U-Net Architecture. (Image source from U-Net paper on MICCAI'2015)</div>
</center>

<center class="half">
    <img src="./images/MHSA.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 17. (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several
attention layers running in parallel. (Image source from Transformer paper on NeurIPS'2017)</div>
</center>

## Reference 参考文献

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

<div id="refer-17"></div>

[17] Diederik P. Kingma, Max Welling, "Auto-Encoding Variational Bayes," ICLR'2014

[Reparameterization Paper on ICLR'2014](https://openreview.net/forum?id=33X9fd2-9FyZd)
&emsp;&emsp;[Reparameterization Paper on arXiv'2013](https://arxiv.org/abs/1312.6114)
&emsp;&emsp;[Reparameterization Code on GitHub](https://github.com/AntixK/PyTorch-VAE)

----------------------------

<div id="refer-18"></div>

[18] Robin Rombach, Andreas Blattmann, et al. "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR'2022

[Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
&emsp;&emsp;[Paper on arXiv'2022](https://arxiv.org/abs/2112.10752)
&emsp;&emsp;[Paper Original Code on GitHub](https://github.com/compvis/latent-diffusion)

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

[23] Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song, "Denoising Diffusion Restoration Models," NeurIPS'2017

[DDRM Project on Website](https://ddrm-ml.github.io/)
&emsp;&emsp;[DDRM Paper on arXiv'2022](https://arxiv.org/abs/2201.11793)
&emsp;&emsp;[DDRM Original Code on GitHub](https://github.com/bahjat-kawar/ddrm)
