# Awesome Understanding of Convolution and Transpose Convolution

- &ensp;<span style="color:MediumPurple">Title</span>: Awesome Understanding of Convolution and Transpose Convolution
- &ensp;<span style="color:Moccasin">Tags</span>: Convolution2D; Transpose Convolution; Dilation; Depthwise and Pointwise;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-08

> Quote learning from 'deep_thoughts' uploader on bilibili.

---------------------

> [Visualtion of Convolution Animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

## **Convolution**

- convolution
    - 控制参数
        - input size; kernel size; stride; padding; group; dilation
    - 计算方法
        - torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        - torch.nn.functional.conv2d(input, weight)
        - 对 input 展开处理，然后利用矩阵相乘 (torch.unfold or 手动分块)
        - 对 kernel 展开处理，然后利用矩阵乘法 (转置卷积)
    - 特点
        - 对局部关系建模；下采样处理
        - 卷积和互相关 (信号系统和处理)
- transpose convolution
    - 计算方式
        - torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        - torch.nn.functional.conv_transpose2d(input, weight)
        - 对 kernel 展开的 matrix 转置，然后进行矩阵乘法 (转置卷积)
    - 特点：上采样处理
- residual convolution operator fusion for inference
    - RepVGG paper on CVPR'2021
    - 卷积算子进行融合，提高推理效率
    - Automatic Differentiation in Machine Learning

---------------------

### **PyTorch API**

$$
\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)
$$

where $\star$ is the valid 2D [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) operator.

$$
H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
$$

$$
W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
$$


```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Awesome Understanding of Convolution and Transpose Convolution
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-11
"""

import torch
import torch.nn.functional as F

# hyper-parameter for Convolution
in_channels = 1
out_channels = 1
kernel_size = 3
batch_size = 1
img_H, img_W = 8, 8
bias = False
input_size = [batch_size, in_channels, img_H, img_W]

# PyTorch API
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
input_feature_map = torch.randn(input_size)
output_feature_map = conv_layer(input_feature_map)
print(input_feature_map, f"\n the shape is {input_feature_map.shape}")
print(conv_layer.weight, f"\n the shape is {conv_layer.weight.shape}") # shape: [out_channels, in_inchannels, H, W]
print(output_feature_map, f"\n the shape is {output_feature_map.shape}")

output_feature_map_functional = F.conv2d(input_feature_map, conv_layer.weight)
print(output_feature_map_functional, f"\n the shape is {output_feature_map_functional.shape}")

flag_equal = torch.allclose(output_feature_map, output_feature_map_functional)
print(flag_equal)
```

> PyTorch 对应的有 class 实现的和 functional 实现的 API，其中 class API 内部调用的是 functional API，而最终底层调用的是 C/C++ 实现的 API，对应着 CPU and GPU 版本；两种之间的区别，在于 functional API 需要传入 kernel or weight 权重矩阵，而 class API 内部封装并初始化了 kernel or weight matrix；这对应 pythonic 思想，有需要学习参数的模块，封装抽象为 class，没有需要学习的模块，仅仅只是计算操作处理，封装抽象为 function.

---------------------

### **滑动示意进行卷积**

<center>
    <img src="./images/conv_stride.jpg" />
</center>

```python
import math

import torch
import torch.nn.functional as F

# ---------------------------------------
# 利用 向量内积 形式计算卷积
# step 1. 根据原始矩阵运算实现二维卷积
def matrix_multiplication_conv2d(input, kernel, bias=0, stride=1, padding=0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape
    
    output_h = (math.floor((input_h - kernel_h) / stride) + 1) # 卷积输出的 H
    output_w = (math.floor((input_w - kernel_w) / stride) + 1) # 卷积输出的 W
    output = torch.zeros(output_h, output_w) # 初始化卷积结果

    # 模拟滑窗方式的卷积操作
    for i in range(0, input_h - kernel_h + 1, stride): # 对 H 维度进行遍历，即从上往下进行滑窗
        for j in range(0, input_w - kernel_w + 1, stride): # 对 W 维度进行遍历，即从左往右进行滑窗
            region_window = input[i:i + kernel_h, j:j + kernel_w] # 选取被 kernel 滑动的区域 (torch.unfold)
            output[int(i/stride), int(j/stride)] = torch.sum(region_window * kernel) + bias # 点乘，对应位置相乘再相加，赋值给输出位置

    return output

# step 2. 根据原始矩阵运算实现二维卷积, 考虑 batch-size 和 channels dimensions, 不考虑效率
def matrix_multiplication_conv2d_full(input, kernel, bias=None, stride=1, padding=0):
    # input and kernel is 4D tensor
    if padding > 0:
        # check the PyTorch API for torch.nn.functional.pad()
        # W-dim(up,down); H-dim(up,dowm); channel-dim(up,down); batch-dim(up,down); 
        input = F.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))

    batch_size, in_channels, input_h, input_w = input.shape
    out_channels, in_channels, kernel_h, kernel_w = kernel.shape
    if bias is None:
        bias = torch.zeros(out_channels)
    
    output_h = (math.floor((input_h - kernel_h) / stride) + 1) # 卷积输出的 H
    output_w = (math.floor((input_w - kernel_w) / stride) + 1) # 卷积输出的 W
    output = torch.zeros(batch_size, out_channels, output_h, output_w) # 初始化卷积结果

    for idx in range(batch_size): # the all samplers for batch-size dimension
        for oc in range(out_channels): # the output channels traverse for channels dimension
            for ic in range(in_channels): # the output channels traverse for channels dimension
                # 模拟滑窗方式的卷积操作
                for i in range(0, input_h - kernel_h + 1, stride): # 对 H 维度进行遍历，即从上往下进行滑窗
                    for j in range(0, input_w - kernel_w + 1, stride): # 对 W 维度进行遍历，即从左往右进行滑窗
                        region_window = input[idx, ic, i:i + kernel_h, j:j + kernel_w] # 选取被 kernel 滑动的区域 (torch.unfold)
                        output[idx, oc, int(i/stride), int(j/stride)] += torch.sum(region_window * kernel[oc, ic]) # 点乘，对应位置相乘再相加，赋值给输出位置
            output[idx, oc] += bias[oc]
    return output

# ---- test ----
input = torch.randn(5, 5)
kernel = torch.randn(3, 3)
bias = torch.randn(1) # 默认输出通道数为 1

mat_mul_conv_output = matrix_multiplication_conv2d(input, kernel, bias=bias, stride=2, padding=1)
print(mat_mul_conv_output)

pytorch_api_conv_output = F.conv2d(input.reshape(1, 1, input.shape[0], input.shape[1]), kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1]), stride=2, padding=1, bias=bias)
pytorch_api_conv_output = pytorch_api_conv_output.squeeze(0).squeeze(0)
print(pytorch_api_conv_output)

flag_equal_2 = torch.allclose(mat_mul_conv_output, pytorch_api_conv_output)
print(flag_equal_2)

# ---- test ----
input_3 = torch.randn(2, 2, 5, 5) # [B, C, H, W]
kernel_3 = torch.randn(3, 2, 3, 3) # [out_C, in_C, kernle_H, kernel_W]
bias_3 = torch.randn(3) # output channels

pytorch_conv2d_output_3 = F.conv2d(input_3, kernel_3, bias=bias_3, padding=1, stride=2)
mm_conv2d_output_3 = matrix_multiplication_conv2d_full(input_3, kernel_3, bias=bias_3, padding=1, stride=2)
print(pytorch_conv2d_output_3)
print(mm_conv2d_output_3)
flag_equal_3 = torch.allclose(mm_conv2d_output_3, pytorch_conv2d_output_3)
print(flag_equal_3)
```

---------------------

### **向量内积进行卷积**

**对 input 展开，然后利用矩阵相乘的优化算法进行加速计算**

> 直观上理解卷积通常都是用滑窗的形式，但是这样去实现显然很不高效; PyTorch 或者 [Caffe](https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp) 中卷积的实现都是基于一个 im2col 算法，具体来说，将特征图中每个待卷积的窗口展开成一列并拼接到一起，这样卷积的操作就可以用一个矩阵乘法来代替; 首先要说明的是，卷积是指 CNN 中默认的卷积，而不是数学意义上的卷积; 其实，ConvNet 中的卷积对于与数学中的 cross correlation; 计算卷积的方法有很多种，常见的有以下几种方法：<br> 滑窗：这种方法是最直观最简单的方法, 但是，该方法不容易实现大规模加速，因此，通常情况下不采用这种方法 (但是也不是绝对不会用，在一些特定的条件下该方法反而是最高效的); <br> im2col：目前几乎所有的主流计算框架包括 Caffe, MXNet 等都实现了该方法, 该方法把整个卷积过程转化成了 GEMM 过程，而 GEMM 在各种 BLAS 库中都是被极致优化的，一般来说，速度较快; <br> FFT：傅里叶变换和快速傅里叶变化是在经典图像处理里面经常使用的计算方法，但是，在 ConvNet 中通常不采用，主要是因为在 ConvNet 中的卷积模板通常都比较小，例如 3×3 等，这种情况下，FFT 的时间开销反而更大，所以很少在 CNN 中利用 FFT 实现卷积; <br> Winograd： Winograd 是存在已久最近被重新发现的方法，在大部分场景中, Winograd 方法都显示和较大的优势，目前 cudnn 中计算卷积就使用了该方法。

<center class="half">
    <img src="./images/img2col.png", width="50%" /><img src="./images/img2col_2.png", width="50%" />
</center>

> BLAS(Basic Linear Algebra Subprograms) 是一组线性代数计算中通用的基本运算操作函数集合; <br> General matrix multiply gemm, one of the Basic Linear Algebra Subprograms; <br> GEMM 在深度学习中是十分重要的，全连接层以及卷积层基本上都是通过 GEMM 来实现的，而网络中大约 90% 的运算都是在这两层中, 一个良好的 GEMM 的实现可以充分利用系统的多级存储结构和程序执行的局部性来充分加速运算。

> [Winograd Algorithms for CNN paper on arXiv](https://arxiv.org/abs/1509.09308)

> [img2col reference](https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf)


```python
import math

import torch
import torch.nn.functional as F

# ---------------------------------------
# step 3. 对 input region 展开
def matrix_multiplication_conv2d_flatten(input, kernel, bias=0, stride=1, padding=0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape
    
    output_h = (math.floor((input_h - kernel_h) / stride) + 1) # 卷积输出的 H
    output_w = (math.floor((input_w - kernel_w) / stride) + 1) # 卷积输出的 W
    output = torch.zeros(output_h, output_w) # 初始化卷积结果

    # input region unfold
    region_matrix = torch.zeros(output.numel(), kernel.numel()) # 存储所有拉平后特征区域向量
    kernel_matrix = kernel.reshape(kernel.numel(), 1) # 列向量 column vector
    row_idx = 0
    for i in range(0, input_h - kernel_h + 1, stride): # 对 H 维度进行遍历，即从上往下进行滑窗
        for j in range(0, input_w - kernel_w + 1, stride): # 对 W 维度进行遍历，即从左往右进行滑窗
            region_window = input[i:i + kernel_h, j:j + kernel_w] # 选取被 kernel 滑动的区域 (torch.unfold)
            region_vector = torch.flatten(region_window)
            region_matrix[row_idx] = region_vector
            row_idx += 1
    
    output_matrix = region_matrix @ kernel_matrix # shape: [output_h*output_w, 1]
    output = output_matrix.reshape((output_h, output_w)) + bias

    return output

# ---- test ----
input_4 = torch.randn(5, 5)
kernel_4 = torch.randn(3, 3)
bias_4 = torch.randn(1) # 默认输出通道数为 1

mat_mul_conv_output_4 = matrix_multiplication_conv2d_flatten(input, kernel, bias=bias, stride=2, padding=1)
print(mat_mul_conv_output_4)

pytorch_api_conv_output_4 = F.conv2d(input.reshape(1, 1, input.shape[0], input.shape[1]), kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1]), stride=2, padding=1, bias=bias)
pytorch_api_conv_output_4 = pytorch_api_conv_output_4.squeeze(0).squeeze(0)
print(pytorch_api_conv_output_4)

flag_equal_4 = torch.allclose(mat_mul_conv_output_4, pytorch_api_conv_output_4)
print(flag_equal_4)
```

---------------------

### **转置卷积**

```python
import torch
import torch.nn.functional as F

# -----------------------------------------------------------
# step 4. 对 kernel 展开, 然后拼成矩阵，与 input 进行矩阵乘法
# 这样来理解卷积操作，很容易推导出转置卷积
# 不考虑 batch size || channels || padding
# stride == 1
def get_kernel_matrix(kernel, input_size):
    """基于 kernel 和 输入特征图的大小, 计算并得到拉平和填充(zero padding)后的 kernel matrix."""
    kernel_h, kernel_w = kernel.shape
    input_h, input_w = input_size
    num_out_feat_map = (input_h - kernel_h + 1) * (input_w - kernel_w + 1)
    
    # 初始化结果矩阵 [输出特征图元素个数，输入特征图元素个数]
    result = torch.zeros((num_out_feat_map, input_h*input_w))

    # 计算每次 region 的 High-dimension and Width-dimension 的 zero-padding
    # for i in range(0, input_h - kernel_h + 1, stride)
    count = 0
    for i in range(0, input_h - kernel_h + 1, 1):
        for j in range(0, input_w - kernel_w + 1, 1):
            # 针对 kernel 进行 zero-padding, 跟输入特征图大小一致
            padded_kernel = F.pad(kernel, (i, input_h - kernel_h - i, j, input_w - kernel_w - j))
            result[count] = padded_kernel.flatten()
            count += 1

    return result


# ---- test convolution ----
input_5 = torch.randn(4, 4) # shape: [H, W]
kernel_5 = torch.randn(3, 3) # shape: [kernel-H, kernel-W]

kernel_matrix = get_kernel_matrix(kernel_5, input_5.shape) # shape: [(H - kernel_H + 1)*(H - kernel_H + 1), H*W]
mm_conv2d_output_5 = kernel_matrix @ input_5.reshape((-1, 1)) # shape: [(H - kernel_H + 1)*(H - kernel_H + 1), 1]
print(mm_conv2d_output_5)
# mm_conv2d_output_5 需要进行 reshape

pytorch_api_conv_output_5 = F.conv2d(input_5.unsqueeze(0).unsqueeze(0), kernel_5.unsqueeze(0).unsqueeze(0))
print(pytorch_api_conv_output_5)

# ---- test transpose convolution ----
# 通过矩阵乘法实现转置卷积
mm_trans_conv_output_5 = kernel_matrix.transpose(-1, -2) @ mm_conv2d_output_5
print(mm_trans_conv_output_5.reshape((4, 4)))

pytorch_api_trans_conv_output_5 = F.conv_transpose2d(pytorch_api_conv_output_5, kernel_5.unsqueeze(0).unsqueeze(0))
print(pytorch_api_trans_conv_output_5)

print(torch.allclose(mm_trans_conv_output_5.reshape((4, 4)), pytorch_api_trans_conv_output_5))
```

---------------------

### **空洞卷积 & 群组卷积 & 深度可分离卷积**

<center class="half">
    <img src="./images/group_conv.png", width="50%" /><img src="./images/depthwise_conv.png", width="50%" />
</center>

```python
import torch
import torch.nn.functional as F

"""how to understanding of dilation concept via code."""
feat_map = torch.randn(7, 7)
print(feat_map)

# case 1. kernel=3, dialtion=1 (default)
print(feat_map[0:3, 0:3])

# case 2. kernel=3, dialtion=2
print(feat_map[0:5:2, 0:5:2])

# case 3. kernel=3, dialtion=3
print(feat_map[0:7:3, 0:7:3])

# ----------------------------------------------------
"""how to understanding of group concept via code."""
# case 1. kernel=3, group=1 (default)
in_channels, out_channels = 2, 4

# case 2. kernel=3, group=2
in_channels, out_channels = 2, 4
sub_in_channels, sub_out_channels = 1, 2
# 前提假设 or inductive bias：group>1, 通道间融合不需要完全充分, 只需要再一个个群组内进行融合即可, 最后拼接

# 1x1 convolution ---> pointwise convolution ---> 考虑通道间建模关系

# case 3. kernel=3, group=out_channels (depthwise convolution)
in_channels, out_channels = 64, 64
sub_in_channels, sub_out_channels = 64, 64 # single kernel as a one group
# inductive bias：group=out_channels, 完全不考虑通道间关系，只考虑空间上局部建模
```

```python
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------------------------------------
def matrix_multiplication_conv2d_final(input, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # input and kernel is 4D tensor
    if padding > 0:
        # W-dim(up,down); H-dim(up,dowm); channel-dim(up,down); batch-dim(up,down); 
        input = F.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))

    batch_size, in_channels, input_h, input_w = input.shape
    # out_channels, in_channels//groups, kernel_h, kernel_w = kernel.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape

    assert out_channels % groups == 0 and in_channels % groups == 0, "groups 必须要同时被输入和输出通道数整除！"
    input = input.reshape((batch_size, groups, in_channels//groups, input_h, input_w))
    kernel = kernel.reshape((groups, out_channels//groups, in_channels//groups, kernel_h, kernel_w))

    kernel_h = (kernel_h - 1)*(dilation-1) + kernel_h
    kernel_w = (kernel_w - 1)*(dilation-1) + kernel_w

    output_h = math.floor((input_h - kernel_h) / stride) + 1
    output_w = math.floor((input_w - kernel_w) / stride) + 1
    output_shape = (batch_size, groups, out_channels//groups, output_h, output_w)
    output = torch.zeros(output_shape)

    if bias is None:
        bias = torch.zeros(out_channels)

    for idx in range(batch_size):
        for g in range(groups):
            for oc in range(out_channels//groups):
                for ic in range(in_channels//groups):
                    for i in range(0, input_h - kernel_h + 1, stride):
                        for j in range(0, input_w - kernel_w + 1, stride):
                            region_window = input[idx, g, ic, i:i+kernel_h:dilation, j:j+kernel_w:dilation]
                            output[idx, g, oc, int(i/stride), int(j/stride)] += torch.sum(region_window * kernel[g, oc, ic])
                output[idx, g, oc] += bias[g*(out_channels//groups) + oc] # 考虑偏置项

    output = output.reshape((batch_size, out_channels, output_h, output_w)) # convert 5-dim into 4-dim

    return output
# -----------------------------------------------------------------------------------------------------------
batch_size, in_channels, input_h, input_w = 2, 2, 5, 5
kernel_size, out_channels = 3, 4
groups, dilation, stride, padding = 2, 2, 2, 1

input = torch.randn(batch_size, in_channels, input_h, input_w)
kernel = torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size)
bias = torch.randn(out_channels)

pytorch_api_conv2d_output = F.conv2d(input, kernel, bias=bias, padding=padding,
                                    stride=stride, dilation=dilation, groups=groups)

mm_conv2d_output = matrix_multiplication_conv2d_final(input, kernel, bias=bias, padding=padding,
                                    stride=stride, dilation=dilation, groups=groups)

flag = torch.allclose(pytorch_api_conv2d_output, mm_conv2d_output)
print(f"all close: {flag}")
```

-------------------------------

**RepVGG paper "RepVGG: Making VGG-style ConvNets Great Again" on [CVPR'2021](https://arxiv.org/pdf/2101.03697.pdf)**

<center class="half">
    <img src="./images/RepVGG_0.png", width="50%" /><img src="./images/RepVGG_1.png", width="50%" />
</center>

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: RepVGG: Making VGG-style ConvNets Great Again
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-11
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
if __name__ == "__main__":
    batch_size, in_channels, out_channels = 1, 2, 2
    kernel_size, h, w = 3, 64, 64
    feature_map = torch.ones(batch_size, in_channels, h, w)
    # res_block = 3*3 conv + 1*1 conv + input

    # ----------------------------------------------------------------------------
    # case 1: original residual convolution
    start_time_1 = time.time()
    conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    conv_2d_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    result_1 = conv_2d(feature_map) + conv_2d_pointwise(feature_map) + feature_map
    print(f"the time of origin ResNet: {time.time() - start_time_1}")
    print(f"the shape of output: {result_1.shape}")
    # ----------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------
    # case 2: convert pointwise and identity into 3*3 convolution
    # [2, 2, 1, 1] ---> [2, 2, 3, 3]
    pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight, [1, 1, 1, 1, 0, 0, 0, 0])
    conv_2d_for_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    conv_2d_for_pointwise.weight = nn.Parameter(pointwise_to_conv_weight)
    conv_2d_for_pointwise.bias = conv_2d_pointwise.bias

    # [2, 2, 3, 3]
    zeros = torch.unsqueeze(torch.zeros(kernel_size, kernel_size), 0)
    stars = torch.unsqueeze(F.pad(torch.ones(1, 1), [1, 1, 1, 1]), 0)
    stars_zeros = torch.unsqueeze(torch.cat([stars, zeros], 0), 0)
    zeros_stars = torch.unsqueeze(torch.cat([zeros, stars], 0), 0)
    identity_to_conv_weight = torch.cat([stars_zeros, zeros_stars], 0)
    identity_to_conv_bias = torch.zeros(out_channels)
    conv_2d_for_identity = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    conv_2d_for_identity.weight = nn.Parameter(identity_to_conv_weight)
    conv_2d_for_identity.bias = nn.Parameter(identity_to_conv_bias)

    result_2 = conv_2d(feature_map) + conv_2d_for_pointwise(feature_map) + conv_2d_for_identity(feature_map)
    print(f"the shape of output: {result_2.shape}")

    flag_1 = torch.all(torch.isclose(result_1, result_2))
    print(f"all close equal of the output: {flag_1}")
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # case 3: fusion the three 3*3 convolution into single 3*3 convolution
    start_time_2 = time.time()
    conv_2d_for_fusion = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    conv_2d_for_fusion.weight = nn.Parameter(conv_2d.weight.data + conv_2d_for_pointwise.weight.data + conv_2d_for_identity.weight.data)
    conv_2d_for_fusion.bias = nn.Parameter(conv_2d.bias.data + conv_2d_for_pointwise.bias.data + conv_2d_for_identity.bias.data)

    result_3 = conv_2d_for_fusion(feature_map)
    print(f"the time of fusion operators: {time.time() - start_time_2}")
    print(f"the shape of output: {result_3.shape}")

    flag_2 = torch.all(torch.isclose(result_3, result_2))
    print(f"all close equal of the output: {flag_2}")
    # ----------------------------------------------------------------------------

"""
$ python rep_vgg.py 
the time of origin ResNet: 0.001994609832763672
the shape of output: torch.Size([1, 2, 64, 64])
the shape of output: torch.Size([1, 2, 64, 64])
all close equal of the output: True
the time of fusion operators: 0.0
the shape of output: torch.Size([1, 2, 64, 64])
all close equal of the output: True
(SR_PyTorch) 
WeiLi@LAPTOP-UG2EDDHM MINGW64 /d/GitRepository/blog (main)
$ python rep_vgg.py 
the time of origin ResNet: 0.002012014389038086
the shape of output: torch.Size([1, 2, 64, 64])    
the shape of output: torch.Size([1, 2, 64, 64])    
all close equal of the output: True
the time of fusion operators: 0.0009970664978027344
the shape of output: torch.Size([1, 2, 64, 64])    
all close equal of the output: True
(SR_PyTorch) 
WeiLi@LAPTOP-UG2EDDHM MINGW64 /d/GitRepository/blog (main)
$ python rep_vgg.py 
the time of origin ResNet: 0.0019936561584472656
the shape of output: torch.Size([1, 2, 64, 64])   
the shape of output: torch.Size([1, 2, 64, 64])   
all close equal of the output: True
the time of fusion operators: 0.000997304916381836
the shape of output: torch.Size([1, 2, 64, 64])   
all close equal of the output: True
(SR_PyTorch)
"""
```


### Reference
----------------------------

[1] Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind, "Automatic Differentiation in Machine Learning: a Survey," JMLR'2018

[Paper on JMLR'2018](https://jmlr.org/papers/v18/17-468.html)

[2] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun, "RepVGG: Making VGG-style ConvNets Great Again," CVPR'2021

[RepVGG Paper on CVPR'2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)
&emsp;&emsp;[Paper on arXiv'2021](https://arxiv.org/abs/2101.03697)
&emsp;&emsp;[Original Code on GitHub](https://github.com/DingXiaoH/RepVGG)
&emsp;&emsp;[Implementation Code on GitHub](https://paperswithcode.com/paper/repvgg-making-vgg-style-convnets-great-again)
