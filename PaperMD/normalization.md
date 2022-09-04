# the Principle of Normalization in Deep Learning

- &ensp;<span style="color:MediumPurple">Title</span>: the Principle of Normalization in Deep Learning
- &ensp;<span style="color:Moccasin">Tags</span>: BatchNorm; LayerNorm; InstanceNorm; GroupNorm; WeightNorm;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-08

> Quote learning from 'deep_thoughts' uploader on bilibili.

---------------------

**Normalization in Deep Neural Networks**
- Batch Normalization
    - per channel across mini-batch
    - NLP:input_shape=[N, L, C] --> statistic term output_shape=[C] dimension
    - CV:input_shape=[N, C, H, W] --> statistic term output_shape=[C] dimension
- Layer Normalization
    - per sample, per layer
    - NLP:input_shape=[N, L, C] --> statistic term output_shape=[N, L] dimension
    - CV:input_shape=[N, C, H, W] --> statistic term output_shape=[N, H, W] dimension
- Instance Normalization
    - per sample, per channel
    - NLP:input_shape=[N, L, C] --> statistic term output_shape=[N, C] dimension
    - CV:input_shape=[N, C, H, W] --> statistic term output_shape=[N, C] dimension
- Group Normalization
    - per sample, per group
    - NLP:input_shape=[N, G L, C//G] --> statistic term output_shape=[N, G] dimension
    - CV:input_shape=[N, G, C//G, H, W] --> statistic term output_shape=[N, G] dimension
- Weight Normalization
    - decompose weight into magnitude and direction
- Unified Normalization


<center class="center">
    <img src="./images/normalization_1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 1. **Normalization methods**. Each subplot shows a feature map tensor, with $N$ as the batch axis, $C$ as the channel axis, and $(H, W)$ as the spatial axes. The pixels in blue are normalized by the same mean and variance, computed by aggregating the values of these pixels. (Image source from GroupNorm paper)</div>
</center>

<center class="center">
    <img src="./images/normalization_2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    font-size:12px;
    padding: 2px;">Fig. 2. **Normalization methods**. Each subfigure shows a feature map tensor, where $B$ is the batch axis, $N$ is the number of tokens (or the sequence length) axis, and $C$ is the channel (also known as the embedding size) axis. (Image source from Unified Normalization paper)</div>
</center>


```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Normlization of Neural Network for Deep Learning and Machine Learning
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-21
"""

import torch
import torch.nn as nn


# --------------------------
if __name__ == "__main__":
    batch_size = 2
    time_steps = 3
    embedding_dim = 4
    num_group = 2

    input_x = torch.randn(batch_size, time_steps, embedding_dim) # shape=[N, L, C]

    # Step 1. batch_norm implementation and PyTorch API
    batch_norm_op = torch.nn.BatchNorm1d(embedding_dim, affine=False)
    bn_result = batch_norm_op(input_x.transpose(-1, -2)).transpose(-1, -2)

    bn_mean = input_x.mean(dim=(0, 1), keepdim=True)
    # unbias=False 有偏估计
    bn_var = input_x.var(dim=(0, 1), unbiased=False, keepdim=True) + 1e-5
    verify_bn_y = (input_x - bn_mean) / torch.sqrt(bn_var)

    assert bn_result.shape == verify_bn_y.shape, "the shape is different."
    print(f"the computation is correct: {torch.all(torch.isclose(bn_result, verify_bn_y))}")
    # ------------------------------------------------------------------------

    # Step 2. layer_norm implementation and PyTorch API
    layer_norm_op = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False)
    ln_result = layer_norm_op(input_x)

    ln_mean = input_x.mean(dim=-1, keepdim=True)
    ln_var = input_x.var(dim=-1, unbiased=False, keepdim=True) + 1e-5
    verify_ln_y = (input_x - ln_mean) / torch.sqrt(ln_var)

    assert ln_result.shape == verify_ln_y.shape, "the shape is different."
    print(f"the computation is correct: {torch.all(torch.isclose(ln_result, verify_ln_y))}")
    # ------------------------------------------------------------------------

    # Step 3. instance_norm implementation and PyTorch API
    instance_norm_op = torch.nn.InstanceNorm1d(embedding_dim, affine=False)
    in_result = instance_norm_op(input_x.transpose(-1, -2)).transpose(-1, -2)

    in_mean = input_x.mean(dim=1, keepdim=True)
    in_var = input_x.var(dim=1, unbiased=False, keepdim=True) + 1e-5
    verify_in_y = (input_x - in_mean) / torch.sqrt(in_var)

    assert in_result.shape == verify_in_y.shape, "the shape is different."
    print(f"the computation is correct: {torch.all(torch.isclose(in_result, verify_in_y))}")
    # ------------------------------------------------------------------------

    # Step 4. group_norm implementation and PyTorch API
    group_norm_op = torch.nn.GroupNorm(num_groups=num_group, num_channels=embedding_dim, affine=False)
    gn_result = group_norm_op(input_x.transpose(-1, -2)).transpose(-1, -2)

    group_inputs = torch.split(input_x, split_size_or_sections=embedding_dim//num_group, dim=-1)
    # print(group_inputs[0].shape)
    group_result  = []
    for g_input in group_inputs:
        gn_mean = g_input.mean(dim=(1, 2), keepdim=True)
        gn_var = g_input.var(dim=(1, 2), unbiased=False, keepdim=True) + 1e-5
        each_group = (g_input - gn_mean) / torch.sqrt(gn_var)
        group_result.append(each_group)

    verify_gn_y = torch.cat(group_result, dim=-1)

    assert gn_result.shape == verify_gn_y.shape, "the shape is different."
    print(f"the computation is correct: {torch.all(torch.isclose(gn_result, verify_gn_y))}")
    # ------------------------------------------------------------------------

    # Step 5. weight_norm implementation and PyTorch API
    linear = nn.Linear(embedding_dim, 3, bias=False)
    wn_linear = torch.nn.utils.weight_norm(linear)
    wn_linear_result = wn_linear(input_x)
    print(wn_linear_result.shape)

    weight_direction = linear.weight / (linear.weight.norm(dim=1, keepdim=True))
    weight_magnitude = wn_linear.weight_g # learnable vector 可以随机初始化
    print(weight_direction.shape)
    print(weight_magnitude.shape)
    verify_wn_linear_y = input_x @ (weight_direction.transpose(-1, -2)) * weight_magnitude.transpose(-1, -2)

    assert wn_linear_result.shape == verify_wn_linear_y.shape, "the shape is different."
    print(f"the computation is correct: {torch.all(torch.isclose(wn_linear_result, verify_wn_linear_y))}")
    # ------------------------------------------------------------------------------------------------------
```


### Reference
----------------------------

[1] Sergey Ioffe, Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ICML'2015

[BatchNorm Paper on ICML'2015](https://proceedings.mlr.press/v37/ioffe15.html)
&emsp;&emsp;[BatchNorm Paper on arXiv'2015](https://arxiv.org/abs/1502.03167)
&emsp;&emsp;[PyTorch API for BatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

[2] Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, "Layer Normalization," arXiv'2016

[LayerNorm Paper on arXiv'2016](https://arxiv.org/abs/1607.06450)
&emsp;&emsp;[PyTorch API for LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

[3] Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, Junyang Lin, "Understanding and Improving Layer Normalization," NeurIPS'2019

[Improved LayerNorm Paper on NeurIPS'2019](https://papers.nips.cc/paper/2019/hash/2f4fe03d77724a7217006e5d16728874-Abstract.html)
&emsp;&emsp;[Improved LayerNorm Paper on arXiv'2019](https://arxiv.org/abs/1911.07013)

[4] Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky, "Instance Normalization: The Missing Ingredient for Fast Stylization," arXiv'2017

[InstanceNorm Paper on arXiv'2017](https://arxiv.org/abs/1607.08022)
&emsp;&emsp;[PyTorch API for InstanceNorm](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html)

[5] Yuxin Wu, Kaiming He, "Group Normalization," ECCV'2018

[GroupNorm Paper on ECCV'2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html)
&emsp;&emsp;[GroupNorm Paper on arXiv'2018](https://arxiv.org/abs/1803.08494)
&emsp;&emsp;[PyTorch API for GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)

[6] Tim Salimans, Diederik P. Kingma, "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks," NeurIPS'2016

[WeightNorm Paper on NeurIPS'2016](https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html)
&emsp;&emsp;[WeightNorm Paper on arXiv'2016](https://arxiv.org/abs/1602.07868)
&emsp;&emsp;[PyTorch API for WeightNorm](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html)

[7] Qiming Yang, Kai Zhang, Chaoxiang Lan, Zhi Yang, Zheyang Li, Wenming Tan, Jun Xiao, Shiliang Pu, "Unified Normalization for Accelerating and Stabilizing Transformers," ACM MM'2022

[Unified Normalization Paper on arXiv'2022](https://arxiv.org/abs/2208.01313)
&emsp;&emsp;[Unified Normalization code on GitHub](https://github.com/hikvision-research/unified-normalization)
