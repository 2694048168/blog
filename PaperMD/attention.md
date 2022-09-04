# Attention Mechanism in Deep Learning

- &ensp;<span style="color:MediumPurple">Title</span>: Attention Mechanism in Deep Learning
- &ensp;<span style="color:Moccasin">Tags</span>: Attention Mechanism; Spatial Attention; Channel Attention; Multi-Head Self-Attention;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Mini-Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-09

> **Understanding of Attention Mechanism for Computer Vision with Diagram, Pseudo-code and PyTorch code** <br> [视觉注意力机制的生理解释](https://zhuanlan.zhihu.com/p/55928013)

<center class="half">
  <img src="./images/attention.jpg" />
</center>

## Overview of Attention
- External Attention arXiv'2021
- Self-Attention NeurIPS'2017
- Squeeze Excitation Attention CVPR'2018
- Selective Kernel Attention CVPR'2019
- CBAM attention ECCV'2018
- BAM attention BMCV'2018
- Efficient Channel Attention CVPR'2020

----------------------------
### [External Attention'2021](<https://arxiv.org/pdf/2105.02358.pdf> (Meng-Hao Guo, Zheng-Ning Liu, et al. "Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks," arXiv'2021)) &ensp;&ensp; [Self-Attention'2017](<https://arxiv.org/pdf/1706.03762.pdf> (Ashish Vaswani, Noam Shazeer, et al. "Attention Is All You Need," NeurIPS'2017))

> self-attention and multi-head self-attention, 通过计算同一个样本所有位置之间的相关性，来捕获长距离依赖; 其计算复杂度是平方级的, 且忽略了不同样本间的联系; 本文提出了一个新颖的注意力方法 External Attention, 仅由两个线性层和两个归一化层构造, 且具备线性的计算复杂度; 进一步提出了 multi-head external attention, 构造一个纯 MLP 的架构网络 EAMLP. External Attention 使用一个外部矩阵 M 来建模第 i 个像素和第 j 行之间的相似性, 且 M 是可学习的、大小可变的,同时，M 还可以随着训练过程的进行建模整个数据集不同样本间的联系; External Attention 就解决了 Self-Attention 的两个缺点. Self-Attention 中使用的 Normalization 是 SoftMax, 其对输入 feature 的尺度非常敏感; 在 External Attention 中使用的 double-normalization, 其对列和行分别计算. Transformer 中 self-attention 在不同输入尺寸下计算了多次，形成 Multi-head self-attention, Multi-head 机制可以捕获 token(patch) 之间不同的关系，对提升性能至关重要

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: External Attention with PyTorch Implementation
@Python Version: 3.10.4
@PyTorch Version: 1.12.1+cu113
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-26
"""

import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalAttention(nn.Module):
    def __init__(self, model_dim, linear_dim):
        super(ExternalAttention, self).__init__()
        self.linear_mk = nn.Linear(model_dim, linear_dim, bias=False)
        self.linear_mv = nn.Linear(linear_dim, model_dim, bias=False)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [batch_size, num_patch/seq_len, model_dim]
        attn = self.linear_mk(x)
        # attn.shape: [batch_size, num_patch/seq_len, linear_dim]
        attn = F.softmax(attn, dim=1) # 针对 num_patch 维度计算概率值作为系数
        attn = attn / torch.sum(attn, dim=2, keepdim=True) # 针对 model_dim 维度做 Normalization 操作
        x = self.linear_mv(attn)

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return x


class MultiHeadExternalAttention(nn.Module):
    def __init__(self, model_dim, linear_dim, num_head):
        super(MultiHeadExternalAttention, self).__init__()
        assert model_dim % num_head == 0, "the number of head must be the aliquot by model dimension."
        self.num_head = num_head
        self.head_dim = model_dim // num_head
        self.multi_head_external_attention = [ExternalAttention(self.head_dim, linear_dim) for _ in range(num_head)]

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [batch_size, num_patch/seq_len, model_dim] ---> [B, L, num_head, model_dim//num_head]
        batch_size, num_patch, model_dim = x.size()
        attn_multi_head = x.view(batch_size, num_patch, self.num_head, self.head_dim)

        multi_head_result = []
        for idx, external_attention in enumerate(self.multi_head_external_attention):
            result_idx = external_attention(attn_multi_head[:, :, idx, :].squeeze())
            multi_head_result.append(result_idx)
        x = torch.cat(multi_head_result, dim=-1)

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return x


# ------------------------------------------------------
class ImagePatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, model_dim):
        super(ImagePatchEmbedding, self).__init__()
        patch_depth = patch_size * patch_size * in_channels
        self.patch_size = patch_size
        self.embedding_patch = nn.Linear(patch_depth, model_dim)

    def forward(self, image):
        # image shape: [B, C, H, W]
        patch_img = F.unfold(image, kernel_size=self.patch_size, stride=self.patch_size)
        # patch_img.shape: [batch_size, path_depth, num_patch]
        patchs = patch_img.transpose(-1, -2)

        patch_embedding = self.embedding_patch(patchs)

        return patch_embedding # shape: [batch_size, num_patch, model_dim]


# --------------------------
if __name__ == "__main__":
    batch_size, in_channels, img_height, img_width = 16, 3, 256, 256
    input = torch.randn(batch_size, in_channels, img_height, img_width)
    print(f"the shape of input: {input.shape}")

    patch_size = 16
    model_dim = 64
    linear_dim = 256
    num_head = 8

    image_embedding = ImagePatchEmbedding(in_channels, patch_size, model_dim)
    input_embedding = image_embedding(input)

    external_attention = ExternalAttention(model_dim, linear_dim)
    external_attention.module_params()
    external_attention.module_flops()
    output = external_attention(input_embedding)
    print(f"the shape of output: {output.shape}")

    print("------------------------------------")

    external_attention_multi_head = MultiHeadExternalAttention(model_dim, linear_dim, num_head)
    external_attention.module_params()
    external_attention.module_flops()
    output_multi_head = external_attention_multi_head(input_embedding)
    print(f"the shape of multi head output: {output_multi_head.shape}")

    print(f"the difference of multi head: {torch.allclose(output, output_multi_head)}")
```

### [Squeeze Excitation Attention CVPR'2018](<https://arxiv.org/pdf/1709.01507.pdf> (Jie Hu, Li Shen, et al. "Squeeze-and-Excitation Networks," CVPR'2018))

> SENet 是 Squeeze-and-Excitation Networks 的简称, 其提出的 Squeeze-Excitation Attention 思想简单, 易于实现, 容易可以加载到现有的网络模型框架中(可扩展性); 核心思想是学习 channel 间的相关性, 筛选出针对通道的注意力, 稍微增加了一点计算量，但是效果比较好. 简单理解就是通过对卷积得到的 feature map 进行处理, 得到一个和通道数一样的一维向量作为每个通道的评价分数, 然后将改分数分别施加到对应的通道上, 得到其结果, 就在原有的基础上只添加了一个模块, 对于其深入理解和详细解释见原始论文.

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Squeeze Excitation Attention with PyTorch Implementation
@Python Version: 3.10.4
@PyTorch Version: 1.12.1+cu113
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-26
"""

import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitationAttention(nn.Module):
    def __init__(self, num_feature, reduction=16):
        super(SqueezeExcitationAttention, self).__init__()
        # 空间信息直接删除，探索通道的相关性
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.full_connect = nn.Sequential(
            nn.Linear(num_feature, num_feature // reduction, bias=False),
            nn.PReLU(num_feature//reduction),
            nn.Linear(num_feature//reduction, num_feature, bias=False),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [B, C, H, W]
        b, c, h, w = x.size()
        attention_prob = self.avg_pool(x).view(b, c)
        print(f"the result shape through the average pooling: {attention_prob.shape}")
        channel_score = self.full_connect(attention_prob).view(b, c, 1, 1)
        print(f"the result shape through the channel correlation: {channel_score.shape}")
        # 直接利用广播机制，对空间分辨率进行复制，将通道分数乘以对应的通道
        x = x * channel_score.expand_as(x)

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return x


# --------------------------
if __name__ == "__main__":
    batch_size, num_feature, img_height, img_width = 16, 128, 256, 256
    input_feature_map = torch.randn(batch_size, num_feature, img_height, img_width)
    print(f"the shape of input: {input_feature_map.shape}")

    reduction = 8

    squeeze_excitation_attention = SqueezeExcitationAttention(num_feature, reduction)
    squeeze_excitation_attention.module_params()
    squeeze_excitation_attention.module_flops()
    output_feature_map = squeeze_excitation_attention(input_feature_map)

    print(f"the difference of feature map: {torch.allclose(output_feature_map, input_feature_map)}")
```

### [Selective Kernel Attention CVPR'2019](<https://arxiv.org/pdf/1903.06586.pdf> (Xiang Li, Wenhai Wang, et al. "Selective Kernel Networks," CVPR'2019))

> 主要提出一种动态选择机制, 允许每个神经元根据输入信息的多个尺度自适应地调整其感受野大小; 本质上设计一种称为选择核 Selective Kernel block, 利用 softmax attention 对不同核大小的多个分支进行融合, 对这些分支的不同 attention 产生融合层神经元有效感受野的大小是不同的. 由于下一个卷积层线性地聚集了来自不同分支的多尺度信息, 因此在同一层具有多尺度信息的模型, 具有根据输入内容调整下一个卷积层神经元感受野大小的内在机制, 本文提出了一种非线性的多核信息聚合方法来实现神经元的自适应感受野大小: SK 由三个算子组成: Split-Fuse-Select, 其中 Split 算子产生多条不同核大小的路径, 对应于不同的神经元感受野大小; Fuse 算子将来自多条路径的信息进行组合和聚合, 以获得选择权重的全局综合表示; Select 操作符根据选择权重聚合不同大小内核的特征映射, SK 卷积在计算上是轻量级的, 会稍微增加参数和计算成本.

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Selective Kernel Attention with PyTorch Implementation
@Python Version: 3.10.4
@PyTorch Version: 1.12.1+cu113
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-26
"""

import time
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveKernelAttention(nn.Module):
    def __init__(self, num_feature, kernels=[1, 3, 5, 7, 11], reduction=16, group=1, channels_minimal=32):
        # if the 'group' == 'num_feature', and then depth-wise convolution.
        super(SelectiveKernelAttention, self).__init__()

        # d and L in the original paper
        self.channels_minimal = max(channels_minimal, num_feature//reduction)
        self.conv_layers = nn.ModuleList([])
        for k_size in kernels:
            self.conv_layers.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(num_feature, num_feature, kernel_size=k_size, padding=k_size//2, groups=group)),
                    ('bn', nn.BatchNorm2d(num_feature)),
                    ('relu', nn.PReLU(num_feature)),
                ]))
            )

        self.fc = nn.Linear(num_feature, self.channels_minimal)
        self.fc_layers = [nn.Linear(self.channels_minimal, num_feature) for _ in range(len(kernels))]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [B, C, H, W]
        b, c, h, w = x.size()
        # ======== Split ========
        conv_outs = []
        for conv in self.conv_layers:
            conv_outs.append(conv(x))
        feature_maps = torch.stack(conv_outs, dim=0) # shape: [kernel, B, C, H, W]

        # ======== Selective ========
        feature_fuse = sum(conv_outs) # shape: [B, C, H, W]
        feature_reduction_channel = feature_fuse.mean(-1).mean(-1) # shape: [B, C]
        feature_attention = self.fc(feature_reduction_channel) # shape: [B, channels_minimal]
        # compute the attention coefficient weight matrix
        weights = []
        for fc_layer in self.fc_layers:
            weight = fc_layer(feature_attention)
            weights.append(weight.view(b, c, 1, 1)) # shape: [B, C, 1, 1]
        attention_weights = torch.stack(weights, dim=0) # shape: [kernel, B, C, 1, 1]
        attention_coefficients = F.softmax(attention_weights, dim=0) # 针对 kernel 维度计算权重系数

        # ======== Fuse ========
        x = (attention_coefficients * feature_maps).sum(dim=0) # shape: [B, C, H, W]

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return x


# --------------------------
if __name__ == "__main__":
    batch_size, num_feature, img_height, img_width = 16, 128, 256, 256
    input_feature_map = torch.randn(batch_size, num_feature, img_height, img_width)
    print(f"the shape of input: {input_feature_map.shape}")

    reduction = 8
    kernel_size_sective = [1, 3, 5, 7]
    # kernel_size_sective = [1, 3, 5, 7, 11]
    # group_num = 1
    group_num = num_feature

    selective_kernel_attention = SelectiveKernelAttention(num_feature, kernel_size_sective, reduction, group_num, channels_minimal=32)
    selective_kernel_attention.module_params()
    selective_kernel_attention.module_flops()
    output_feature_map = selective_kernel_attention(input_feature_map)

    print(f"the difference of feature map: {torch.allclose(output_feature_map, input_feature_map)}")
```

### [CBAM Attention ECCV'2018](<https://arxiv.org/pdf/1807.06521.pdf> (Sanghyun Woo, Jongchan Park, et.al "CBAM: Convolutional Block Attention Module," ECCV'2018))

> 提出一个简单但有效的注意力模块 CBAM, 对于卷积后的特征图, 沿着空间和通道两个维度依次计算出注意力权重, 然后与原特征图相乘来对特征进行自适应调整. CBAM 是一个轻量级的通用模块, 可以无缝地集成到任何 CNN 架构中, 额外开销忽略不计, 并且可以与基本 CNN 一起进行端到端的训练. 提升 CNN 模型的表现, 主要集中在三个重要的方面：深度、宽度和基数 (cardinality), ResNet 让构建非常深的网络成为可能; GoogLeNet 则表明宽度也是提升模型性能的另一个重要的因素; Xception 和 ResNeXt 提出增加网络的基数, 经验表明基数不仅可以节省参数总量, 还可以产生比深度和宽度更强的表示能力. 网络架构设计方向即使注意力机制, 注意力不仅要告诉重点关注哪里, 还要提高关注点的表示, 通过使用注意机制来增加表现力, 关注重要特征并抑制不必要的特征. CBAM 强调空间和通道这两个维度上的有意义特征, 依次应用通道和空间注意模块, 来分别在通道和空间维度上学习关注什么、在哪里关注, 同时通过了解要强调或抑制的信息也有助于网络内的信息流动. 详细内容和消融实验细节和解释见原始论文.

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: CBAM Attention with PyTorch Implementation
@Python Version: 3.10.4
@PyTorch Version: 1.12.1+cu113
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-26
"""

import time
import warnings

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.share_mlp = nn.Sequential(
            nn.Conv2d(num_channels, num_channels//reduction, kernel_size=1, bias=False),
            nn.PReLU(num_channels//reduction),
            nn.Conv2d(num_channels//reduction, num_channels, kernel_size=1, bias=False),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [B, C, H, W]
        x_max_pool = self.max_pool(x) # shape[B, C, 1, 1]
        x_avg_pool = self.avg_pool(x)
        # print(x_max_pool.shape)
        # print(x_avg_pool.shape)

        max_feature = self.share_mlp(x_max_pool)
        avg_feature = self.share_mlp(x_avg_pool)

        channel_attention_coefficient = torch.sigmoid(max_feature + avg_feature)

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return channel_attention_coefficient # shape:[B, C, 1, 1]

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # [max_pool, avg_pool] ---> [spatial_attention]
        self.conv_layer = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [B, C, H, W]
        max_result, _ = torch.max(x, dim=1, keepdim=True) # shape:[B, H, W]
        avg_result = torch.mean(x, dim=1, keepdim=True)

        spatial_feature = torch.cat([max_result,avg_result], 1) # shape:[B, 2, H, W]
        feature_map = self.conv_layer(spatial_feature) # shape:[B, 1, H, W]
        spatial_attention_coefficient = torch.sigmoid(feature_map)

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return spatial_attention_coefficient # shape:[B, 1, H, W]


class CBAMAttention(nn.Module):
    def __init__(self, num_channels, reduction=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.channel_attention_block = ChannelAttention(num_channels, reduction)
        self.spatial_attention_block = SpatialAttention(kernel_size)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [B, C, H, W]
        feature_channel = x * self.channel_attention_block(x)
        feature_spatial = feature_channel * self.spatial_attention_block(feature_channel)
        x = x + feature_spatial

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return x


# --------------------------
if __name__ == "__main__":
    batch_size, num_feature, img_height, img_width = 16, 128, 256, 256
    input_feature_map = torch.randn(batch_size, num_feature, img_height, img_width)
    print(f"the shape of input: {input_feature_map.shape}")

    reduction = 8
    kernel_size = 5

    cbam_attention = CBAMAttention(num_feature, reduction, kernel_size)
    cbam_attention.module_params()
    cbam_attention.module_flops()
    output_feature_map = cbam_attention(input_feature_map)

    print(f"the difference of feature map: {torch.allclose(output_feature_map, input_feature_map)}")
```


### [BAM Attention BMVC'2018](<https://arxiv.org/pdf/1807.06514.pdf> (Jongchan Park, Sanghyun Woo, et.al "BAM: Bottleneck Attention Module," BMVC'2018))

> 人眼视觉系统不可能把注意力放在所有的图像上, 会把焦点目光聚集在图像的重要物体上(显著性). 提出 BAM 注意力机制, 仿照人的眼睛聚焦在图像几个重要的点上. 本文将重心放在 Attention 对于一般深度神经网络的影响上, 可以结合到任何前向传播卷积神经网络中; 通过两个分离的路径 channel and spatial, 得到一个 Attention Map; 主要结合 Bottleneck 的核心思想, 而 CBAM (the same authors) 将注意力直接用在 convolution block 之间.

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: BAM Attention with PyTorch Implementation
@Python Version: 3.10.4
@PyTorch Version: 1.12.1+cu113
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-26
"""

import time
import warnings

import torch
import torch.nn as nn


# 这里写成 class 是为了以 torch.nn.Module 模块形式进行使用
class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0], -1)

class ChannelAttention(nn.Module):
    def __init__(self, num_channel, reduction=16, num_layers=3):
        super(ChannelAttention, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        gate_channels = [num_channel]
        gate_channels += [num_channel // reduction] * num_layers
        gate_channels += [num_channel]
        # print(gate_channels) # list 存储通道数量 bottleneck 形状

        self.channel_attention = nn.Sequential()
        self.channel_attention.add_module('flatten', Flatten())
        for idx in range(len(gate_channels) - 2): # 最后一层 full connectting(MLP)
            self.channel_attention.add_module('fc%d'%idx, nn.Linear(gate_channels[idx], gate_channels[idx+1]))
            self.channel_attention.add_module('bn%d'%idx, nn.BatchNorm1d(gate_channels[idx+1]))
            self.channel_attention.add_module('prelu%d'%idx, nn.PReLU())
        self.channel_attention.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))
        

    def forward(self, x) :
        res = self.avgpool(x) # shape:[B, C, 1, 1]
        res = self.channel_attention(res) # shape:[B, C]
        res = res.unsqueeze(-1).unsqueeze(-1) # shape:[B, C, 1, 1]
        return res

class SpatialAttention(nn.Module):
    def __init__(self, num_channel, reduction=16, num_layers=3, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        self.spatial_attention=nn.Sequential()
        self.spatial_attention.add_module('conv_reduce1', nn.Conv2d(kernel_size=1, in_channels=num_channel,out_channels=num_channel//reduction)) # shape[B, C//reduction, H, W]
        self.spatial_attention.add_module('bn_reduce1', nn.BatchNorm2d(num_channel//reduction))
        self.spatial_attention.add_module('prelu_reduce1', nn.PReLU())

        for idx in range(num_layers):
            self.spatial_attention.add_module('conv_%d'%idx, nn.Conv2d(kernel_size=3, in_channels=num_channel//reduction,out_channels=num_channel//reduction, padding=dilation_rate, dilation=dilation_rate))
            self.spatial_attention.add_module('bn_%d'%idx, nn.BatchNorm2d(num_channel//reduction))
            self.spatial_attention.add_module('prelu_%d'%idx, nn.PReLU())

        self.spatial_attention.add_module('last_conv', nn.Conv2d(num_channel//reduction, 1, kernel_size=1))

    def forward(self, x):
        # x.shape: [B, C, H, W]
        res = self.spatial_attention(x) # shape:[B, 1, H, W]
        res = res.expand_as(x)
        return res


class BAMBlock(nn.Module):
    def __init__(self, num_channel, reduction=16, dilation_rate=2):
        super(BAMBlock, self).__init__()
        self.channel_attention_block = ChannelAttention(num_channel, reduction=reduction)
        self.spaial_attention_block = SpatialAttention(num_channel, reduction=reduction, dilation_rate=dilation_rate)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        channel_attention_output = self.channel_attention_block(x)
        spaial_attention_output = self.spaial_attention_block(x)
        weight_coefficient = torch.sigmoid(spaial_attention_output + channel_attention_output)
        # print(weight_coefficient.shape) # [B, C, H, W]
        out = (1 + weight_coefficient) * x

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return out


# --------------------------
if __name__ == "__main__":
    batch_size, num_channel, img_height, img_width = 16, 128, 256, 256
    input_feature_map = torch.randn(batch_size, num_channel, img_height, img_width)
    print(f"the shape of input: {input_feature_map.shape}")

    reduction = 8
    dilation_rate = 4

    bam_attention = BAMBlock(num_channel, reduction, dilation_rate)
    bam_attention.module_params()
    bam_attention.module_flops()
    output_feature_map = bam_attention(input_feature_map)

    print(f"the difference of feature map: {torch.allclose(output_feature_map, input_feature_map)}")
```


### [ECA Attention CVPR'2020](<https://arxiv.org/pdf/1910.03151.pdf> (Qilong Wang, Banggu Wu, et.al "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," CVPR'2020))

> 对通道注意力进行改进, ECANet 主要对 SENet 模块进行一些改进, 提出一种不降维的局部跨信道交互策略和自适应选择一维卷积核大小的方法, 从而实现性能上的提优. 通过对 SENet 中通道注意模块的分析, 经验表明避免降维对于学习通道注意力非常重要, 适当的跨信道交互可以在显著降低模型复杂度的同时保持性能, 提出不降维的局部跨信道交互策略, 该策略可以通过一维卷积有效地实现; 进一步提出一种自适应选择一维卷积核大小的方法, 以确定局部跨信道交互的覆盖率.

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Efficient Channel Attention with PyTorch Implementation
@Python Version: 3.10.4
@PyTorch Version: 1.12.1+cu113
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-26
"""

import time
import warnings

import torch
import torch.nn as nn


class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(EfficientChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_layer = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1)//2)

    def module_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"the number of learnable parameter for this module: {total_params}")
        print(f"the number of learnable parameter for this module: {total_params/1000.0:.4f}K")
        print(f"the number of learnable parameter for this module: {total_params/1000000.0:.4f}M")
        # 100,000,000 = 100,000K = 100M

        return total_params

    def module_flops(self):
        print(f"\033[1;33;40m the FLOPs of this module is not implemented! \033[0m")
        warnings.warn('the FLOPs of this module is not implemented!')

    def forward(self, x):
        start_time = time.time()

        # x.shape: [B, C, H, W]
        global_avg_pool_result = self.global_avg_pool(x) # shape:[B, C, 1, 1]
        # print(global_avg_pool_result.shape)
        global_avg_pool_result = global_avg_pool_result.squeeze(-1).permute(0, 2, 1) # shape:[B, 1, C]
        # global_avg_pool_result = global_avg_pool_result.squeeze(-1).transpose(-1, -2)

        # 这样处理维度是为了进行 1-dimension convolution
        conv_result = self.conv_layer(global_avg_pool_result)
        score_prob = torch.sigmoid(conv_result) # shape:[B, 1, C]
        # print(score_prob.shape)
        channel_score_prob = score_prob.permute(0, 2, 1).unsqueeze(-1) # shape:[B, C, 1, 1]
        x = x * channel_score_prob.expand_as(x)

        inference_time = time.time() - start_time
        print(f"the forward inference time: {inference_time:.6f} seconds")

        return x # shape:[B, C, H, W]


# --------------------------
if __name__ == "__main__":
    batch_size, num_channel, img_height, img_width = 16, 128, 256, 256
    input_feature_map = torch.randn(batch_size, num_channel, img_height, img_width)
    print(f"the shape of input: {input_feature_map.shape}")

    kernel_size = 3

    eca_attention = EfficientChannelAttention(kernel_size)
    eca_attention.module_params()
    eca_attention.module_flops()
    output_feature_map = eca_attention(input_feature_map)

    print(f"the difference of feature map: {torch.allclose(output_feature_map, input_feature_map)}")
```


### Reference
----------------------------

[1] Meng-Hao Guo, Zheng-Ning Liu, Tai-Jiang Mu, Shi-Min Hu, "Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks," arXiv'2021

[Paper on arXiv'2021](https://arxiv.org/abs/2105.02358)
&emsp;&emsp;[PyTorch code for paper](https://github.com/MenghaoGuo/-EANet)

[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need," NeurIPS'2017

[Transformer Paper on NeurIPS'2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
&emsp;&emsp;[Transformer Paper on arXiv'2017](https://arxiv.org/abs/1706.03762)
&emsp;&emsp;[Paper Original Code on GitHub](https://github.com/tensorflow/tensor2tensor)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/huggingface/transformers)

[3] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, "Squeeze-and-Excitation Networks," CVPR'2018 and TPAMI'2020

[Paper on CVPR'2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)
&emsp;&emsp;[Paper on TPAMI'2020](https://ieeexplore.ieee.org/document/8701503)
&emsp;&emsp;[Paper on arXiv'2019](https://arxiv.org/abs/1709.01507)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/hujie-frank/SENet)

[4] Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang, "Selective Kernel Networks," CVPR'2019

[Paper on CVPR'2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html)
&emsp;&emsp;[Paper on arXiv'2019](https://arxiv.org/abs/1903.06586)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/implus/SKNet)

[5] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon, "CBAM: Convolutional Block Attention Module," ECCV'2018

[Paper on ECCV'2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)
&emsp;&emsp;[Paper on arXiv'2018](https://arxiv.org/abs/1807.06521)
&emsp;&emsp;[Implementation Code on GitHub](https://paperswithcode.com/paper/cbam-convolutional-block-attention-module)

[6] Jongchan Park, Sanghyun Woo, Joon-Young Lee, In So Kweon, "BAM: Bottleneck Attention Module," BMVC'2018

[Paper on BMVC'2018](http://bmvc2018.org/contents/papers/0092.pdf)
&emsp;&emsp;[Paper on arXiv'2018](https://arxiv.org/abs/1807.06514)
&emsp;&emsp;[Implementation Code on GitHub](https://paperswithcode.com/paper/bam-bottleneck-attention-module)

[7] Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu, "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," CVPR'2020

[Paper on CVPR'2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html)
&emsp;&emsp;[Paper on arXiv'2020](https://arxiv.org/abs/1910.03151)
&emsp;&emsp;[Implementation Code on GitHub](https://github.com/BangguWu/ECANet)
