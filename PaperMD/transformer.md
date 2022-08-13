# Vision Transformer and Swin Transformer and Masked AutoEncoder

- &ensp;<span style="color:MediumPurple">Title</span>: Vision Transformer and Swin Transformer and Masked AutoEncoder
- &ensp;<span style="color:Moccasin">Tags</span>: Transformer; ViT; Swin Transformer; MAE;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2022-08

> Quote learning from 'deep_thoughts' uploader on bilibili.

---------------------

## **Model Architecture**
- CNN
    - 权重共享：平移不变性；可并行计算
    - 滑动窗口：局部关联性；依靠多层堆叠进行长距离关系建模
    - 对相对位置敏感，对绝对位置不敏感
- RNN
    - 以此有序递归建模：对顺序敏感
    - 串行计算耗时：计算复杂度与序列长度呈线性关系；单步计算复杂度不变
    - 长距离建模能力弱 (solution of Deep Neural Network)
    - 对相对位置敏感，对绝对位置也敏感
- Transformer
    - 无局部假设(归纳偏置)：可并行计算；对相对位置不敏感
    - 无有序假设：需要位置编码来反映位置变化对特征的影响；对绝对位置不敏感
    - 无先验假设或归纳偏置：数据量的要求与先验假设的程度成负相关关系(反比); Transformer 模型性能的上限很高, 减少对人类的归纳偏置知识的需求
    - 任意两个 Tokens 都可以建模关系：擅长远距离建模；自注意力机制复杂度与 Token 长度呈平方关系
    - 使用类型：Encoder only(BERT, 分类, 非流式任务); Decoder only(GPT系列, 自回归生成任务, 流式任务); Encoder-Decoder(机器翻译, 语音识别)


> [HuggingFace](https://huggingface.co/) <br> [HuggingFace Transformer](https://huggingface.co/docs/transformers/index) <br> [The Annotated Transformer from Harvard-NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html) <br> [Inductive bias](https://en.wikipedia.org/wiki/Inductive_reasoning) <br>  [Deductive bias](https://en.wikipedia.org/wiki/Deductive_reasoning) <br> 

----------------------------
### Transformer

Transformer
- Encoder
    - input word embedding: 将稀疏的 one-hot 通过无 bias 的 Feed Forward Network 得到稠密的连续向量编码表征
    - position embedding or encoding:
        - 通过 sin-cos 来固定表征：每个位置是确定性的(不需要学习); 对于不同序列，相同位置的距离一致；可以推广到更长的测试序列
        - PE(pos + k) 可以通过 PE(k) 的线性组合来表示，有数学理论支持
        - 通过残差连接 shortcut 使得位置信息表征信息流不断流入深层，保证位置表征信息不会消息
    - multi-head self-attention (MHSA):
        - 多头机制使得模型能力更强，表征空间更丰富
        - 多组 Q、K、V 构成，每组相互独立计算 attention，然后 concatenate 组合在一起
        - 由于多头机制存在：dim_model_head = dim_model // num_head; 保持计算量不会剧增
        - 每一组 attention 向量拼接，并通过 FFN 得到最终的向量
    - feed-forward network (FFN):
        - 只考虑每一个单独位置建模 (NLP for token each dimension-fusion)
        - 不同位置参数共享
        - FFN：dim_model ---> 4*dim_model ---> dim_model
        - 类似 1*1 pointwise convolution (image for channel-fusion)
- Decoder
    - output word embedding
    - casual or masked multi-head self-attention: 考虑因果序列
    - memory-based or cross multi-head self-attention: Query from Decoder; Key-Value from Encoder 
    - feed-forward network
    - softmax to classification
- PyTorch API Implementation
    - torch.nn.Transformer
    - TransformerEncoderLayer class
    - TransformerEncoder class
    - TransformerDecoderLayer class
    - TransformerDecoder class

<center class="half">
  <img src="./images/transformer.png" width="50%" /><img src="./images/MHSA.png" width="50%" />
</center>


----------------------------
### Vision Transformer

Vision Transformer
- Deep Neural Network(DNN) perspective
    - image2patch
    - patch2embedding
- CNN perspective
    - 2-dimension convolution over image (kernel size == stride == patch size)
    - flatten the output feature map
- class token embedding
- position embedding: interpolation when inference
- Transformer Encoder
- classification head

<center class="half">
  <img src="./images/vit.png" />
</center>

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Vision Transformer
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------
def image2embedding_naive(image, patch_size, weight):
    # image shape: [B, C, H, W]
    patch_img = F.unfold(image, kernel_size=patch_size, stride=patch_size)
    print(patch_img.shape)
    patch = patch_img.transpose(-1, -2)
    print(patch.shape)

    patch_embedding = patch @ weight
    print(patch_embedding.shape)

    return patch_embedding # shape: [batch_size, token_sequence_len, model_dim]

# ------------------------------------------------------
def image2embedding_conv(image, kernel, stride):
    conv_output = F.conv2d(image, weight=kernel, stride=stride) # [B, out_channels, out_h, out_w]
    batch_size, out_channels, out_h, out_w = conv_output.shape # out_channels == model_dimension

    # kernel shape: [patch_depth, model_dim] ---> [out_channels, in_channels, kernel_h, kernel_w]
    # model_dim == out_channels; patch_depth = in_channels, patch_size, patch_size; kernel_size == patch_size
    patch_embedding = conv_output.reshape((batch_size, out_channels, out_h * out_w)).transpose(-1, -2)
    print(patch_embedding.shape)

    return patch_embedding # shape: [batch_size, token_sequence_len, model_dim]


# --------------------------
if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Step 1. convert image to patch into embedding vector sequence.
    batch_size, in_channels, img_h, img_w = 1, 3, 64, 64
    patch_size = 16
    patch_depth = patch_size * patch_size * in_channels # kernel_size * kernel_size * in_channels
    model_dim = 512
    # max_num_token = 16 # token sequence length
    num_classes = 10
    lables = torch.randint(10, (batch_size, ))

    image = torch.randn(batch_size, in_channels, img_h, img_w)
    weight = torch.randn(patch_depth, model_dim) # patch to embedding

    patch_embedding_naive = image2embedding_naive(image, patch_size, weight)
    print(patch_embedding_naive.shape)

    # kernel shape: [out_channels, in_channels, kernel_h, kernel_w]
    kernel = weight.transpose(0, 1).reshape(-1, in_channels, patch_size, patch_size)

    patch_embedding_conv = image2embedding_naive(image, patch_size, weight)
    print(patch_embedding_conv.shape)

    print(torch.all(torch.isclose(patch_embedding_naive, patch_embedding_conv)))

    # -------------------------------------------------------------------
    # Step 2. prepend CLS token embedding
    cls_token_embedding = torch.randn(batch_size, 1, model_dim, requires_grad=True)
    token_embedding = torch.cat([cls_token_embedding, patch_embedding_conv], dim=1)

    # -------------------------------------------------------------------
    # Step 3. add position embedding into the token embedding
    seq_len = token_embedding.shape[1]
    position_embedding_table = torch.randn(seq_len, model_dim, requires_grad=True)
    # 从二维张量复制为三维张量，增加 batch size 维度
    position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])

    token_embedding += position_embedding

    # -------------------------------------------------------------------
    # Step 4. pass the token embedding into Encoder of Transformer
    encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    encoder_output = transformer_encoder(token_embedding) # shape: [batch_size, token_sequence_len, model_dim]

    # -------------------------------------------------------------------
    # Step 5. classification
    cls_token_output = encoder_output[:, 0, :]
    linear_layer = nn.Linear(model_dim, num_classes)
    logits = linear_layer(cls_token_output)

    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(logits, lables)
    print(loss)
```


----------------------------
### Swin Transformer

Swin Transformer
- Patch Embedding: naive version and conv2d version
- Swin Transformer Block
    - Window Multi-Head Self-Attention (W-MHSA)
    - Shift Window Multi-Head Self-Attention (Swin-MHSA)
        - why and how to Shift window
        - window maksed to efficient computation 高效计算
        - reverse shift window
- Patch Merging
    - patch(token) number reduction
    - depth dimension expansion
- Classification

<center class="half">
  <img src="./images/swin_0.png" width="50%" /><img src="./images/swin.png" width="50%" />
  <img src="./images/swin_1.png" width="50%" /><img src="./images/swin_2.png" width="50%" />
</center>

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Swin Transformer
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-12
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
```


#### Step 1. 如何将图像转化为 token 即 tokenization in NLP; 生成 patch embedding

Method 1:
1. 基于 pytorch unfold API 将图片进行分块, 模拟卷积核心想法, 设置 kernel_size=patch_size, 得到分块后的 token(patch) 序列
2. token(patch) shape: [batch_size, num_patch, patch_depth]; patch_depth=patch_size*patch_size*in_channels
3. 将 patch tensor 与形状为 [patch_depth, model_dim_C] 的权重矩阵进行相乘, 即完成 patch embedding 过程
4. patch_embedding shape: [batch_size, num_patch, model_dim_C]

Method 2:
1. patch_depth == in_channels * patch_size * patch_size
2. model_dim_C == out_channels of torch.nn.Conv2d; 直接通过卷积方式进行 patch embedding
3. 将形状为 [patch_depth, model_dim_C] 的权重矩阵转化为 [model_dim_C, in_channels, patch_size, patch_size] 的卷积核
4. 利用 pytorch conv2d API 计算图像卷积后的输出特征图张量; shape:[batch_size, out_channels, out_height, out_width]
5. 将卷积输出特征图转为 [batch_size, num_patch, model_dim_C] 形状, 即得到 patch embedding


```python
# ------------------------------------------------------
def image2embedding_naive(image, patch_size, weight):
    # image shape: [B, C, H, W]
    patch_img = F.unfold(image, kernel_size=patch_size, stride=patch_size)
    patch = patch_img.transpose(-1, -2)

    patch_embedding = patch @ weight

    return patch_embedding # shape: [batch_size, token_sequence_len, model_dim]

# ------------------------------------------------------
def image2embedding_conv(image, kernel, stride):
    conv_output = F.conv2d(image, weight=kernel, stride=stride) # [B, out_channels, out_h, out_w]
    batch_size, out_channels, out_h, out_w = conv_output.shape # out_channels == model_dimension

    # kernel shape: [patch_depth, model_dim] ---> [out_channels, in_channels, kernel_h, kernel_w]
    # model_dim == out_channels; patch_depth = in_channels, patch_size, patch_size; kernel_size == patch_size
    patch_embedding = conv_output.reshape((batch_size, out_channels, out_h * out_w)).transpose(-1, -2)

    return patch_embedding # shape: [batch_size, token_sequence_len, model_dim]
``` 

#### Step 2. 如何构建 MHSA 并计算其复杂度

二维矩阵乘法: 规定记号: $A_{MN}$ 表示大小是 $M*N$ 的矩阵; 那么 $A_{MN} * B_{NL}$ 的时间复杂度是 $big-O(MNL)$; <br>
如果把乘法的过程用计算机语言表示出来，这一结论就会非常清晰：<br>
```cpp
C = np.zeros((M, L))
for m in range(M):
    for l in range(L):
        for n in range(N):
            C[m][l] += A[m][n] * B[n][l]
```

1. 基于输入张量 x (shape=[L, C]) 进行三个映射 (MLP layer shape=[C, C]) 分别得到 Q、K、V
    - 此步骤计算的复杂度为 $3LC^{2}$, 其中 L 为序列 token(patch) 的长度 (num_patch or seq_len), C 为特征的大小 (model_dim_C)
2. 将 q, k, v 拆分成多头机制(model_dim_C // num_head) multi-head, 每一个头都是独立计算的, 互不影响, 可以将其与 batch_size 维度进行统一看待
3. 计算 $qk^{T}$, 并考虑可能的掩码, 即让无效的两两位置间的能量为负无穷(-1e-9), masked 在 shift window MHSA 中用到, window MHSA 暂不需要
    - 此步骤计算的复杂度为 $L^{2}C$; (q-shape=[L, C], $k^{T}$-shape=[C, L])
4. 计算概率值(attention coefficient, shape=[L, L]) 与 v (shape=[L, C]) 的乘积
    - 此步骤计算的复杂度为 $L^{2}C$
4. 对输出 (shape=[L, C]) 进行再一次的映射 (shape=[C, C])
    - 此步骤计算的复杂度为 $LC^{2}$
5. 总体 MHSA 的计算复杂度为 $4LC^{2} + 2LC^{2}$

```python
# ------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head

        # self.proj_linear_layer_q = nn.Linear(model_dim, model_dim)
        # self.proj_linear_layer_k = nn.Linear(model_dim, model_dim)
        # self.proj_linear_layer_v = nn.Linear(model_dim, model_dim)
        self.proj_linear_layer = nn.Linear(model_dim, 3*model_dim)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self, input, additive_mask=None):
        batch_size, seq_len, model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim // num_head

        proj_output = self.proj_linear_layer(input)
        # 拆分 chunk [B, num_patch, 3*model_dim] ---> 3*[B, num_patch, model_dim]
        q, k, v = proj_output.chunk(3, dim=-1)

        # multi-head mechanism
        q = q.reshape(batch_size, seq_len, num_head, head_dim).transpose(1, 2) # [B, num_head, seq_len, head_dim]
        q = q.reshape(batch_size*num_head, seq_len, head_dim) # [B*num_head, seq_len, head_dim]
        
        k = k.reshape(batch_size, seq_len, num_head, head_dim).transpose(1, 2) # [B, num_head, seq_len, head_dim]
        k = k.reshape(batch_size*num_head, seq_len, head_dim) # [B*num_head, seq_len, head_dim]

        v = v.reshape(batch_size, seq_len, num_head, head_dim).transpose(1, 2) # [B, num_head, seq_len, head_dim]
        v = v.reshape(batch_size*num_head, seq_len, head_dim) # [B*num_head, seq_len, head_dim]

        if additive_mask is None:
            # [B*num_head, L, C] @ [B*num_head, C, L], scaled dot-product attention
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
        else: # 需要进行 masked 进行高效计算
            additive_mask = additive_mask.tile((num_head, 1, 1)) # 每一个独立的 head 都需要，故此复制 num_head
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim) + additive_mask, dim=-1)

        output = torch.bmm(attn_prob, v) # [B*num_head, seq_len, head_dim]
        output = output.reshape(batch_size, num_head, seq_len, head_dim).transpose(1, 2) # [B, seq_len, num_head, head_dim]
        output = output.reshape(batch_size, seq_len, model_dim)

        output = self.final_linear_layer(output)
        return attn_prob, output
```


#### Step 3. 如何构建 Window-based MHSA 并计算其复杂度
1. 将 patch 组成的图像进一步划分为一个个更大的 window 形式 (window=some patchs)
    - 首先需要将三维的 patch embedding 转为成图像格式
    - 使用 torch unfold API 将 patch 划分 window
2. 在每一个 window 内部计算 MHSA
    - 每一个 window 之间是互相独立计算的(可并行计算)(没有交互, so shift window), 则 window 数量其实可以与 batch_size 维度进行统一对待
    - 计算复杂度
        - 假设窗口的边长 window_size=W ($L=W^{2}$), 则计算每个窗口的总体复杂度为 $4W^{2}C^{2} + 2W^{4}C$
        - 假设 patch 的总数量为 L (num_patch), 则窗口的数量为 $\frac{L}{W^{2}}$
        - 故此 W-MHSA 的总体复杂度为 $4LC^{2} + 2LW^{2}C$
    - Window-based MHSA 和 ViT 中 Transformer Encoder 一致,不需要进行 mask
    - 将此计算结果转化为 window-based 的四维张量格式
3. MHSA and W-MHSA 的计算复杂度对比
    - W-MHSA: $4LC^{2} + 2LW^{2}C$; big-O($L^{2}$)
    - MHSA: $4LC^{2} + 2LC^{2}$; big-O(L)

```python
def window_multi_head_self_attention(patch_embedding, mhsa, window_size=4, num_head=2):
    num_patch_in_window = window_size * window_size
    batch_size, num_patch, patch_depth = patch_embedding.shape
    image_height = image_width = int(math.sqrt(num_patch))

    # [batch_size, num_patch, patch_depth] ---> [batch_size, patch_depth, num_patch]
    patch_embedding = patch_embedding.transpose(-1, -2)
    patch = patch_embedding.reshape(batch_size, patch_depth, image_height, image_width)
    window = F.unfold(patch, kernel_size=(window_size, window_size), stride=(window_size, window_size)).transpose(-1, -2)
    batch_size, num_window, patch_depth_time_num_patch_in_window = window.shape 

    window = window.reshape(batch_size*num_window, patch_depth, num_patch_in_window).transpose(-1, -2)
    attn_prob, output = mhsa(window) # [batch_size*num_window, num_patch_in_window, patch_depth]

    output = output.reshape(batch_size, num_window, num_patch_in_window, patch_depth)
    return output
```

#### Step 4. 如何构建 shift window MHSA 及其 Mask ?
1. 将上一步的 W-MHSA 的结果转换为图像格式
2. 假设已经做了新的 Window 划分, 这一步成为 shift-window
3. 为了保持 window 数量不变从而高效计算, 需要将图片的 patch 往左边和往上边各自滑动半个窗口大小的步长， 保持 patch 所属 window 类别不变
4. 将图像 patch 还原为 window 格式的数据张量
5. 由于 shift-window 后, 每个 window 虽然形状规整(便于高效计算),但是部分 window 中存在原本不属于同一个窗口的 patch, 因此需要 mask
6. 如何生成对应的 Mask ?
    - 首先构建一个 shift-window 的 patch 所属 window 的类别矩阵
    - 对该矩阵进行同样的往左和往上各自滑动半个窗口大小的步长的操作
    - 通过 unfold 操作得到 [B, num_window, num_patch_in_window] 形状的类别矩阵
    - 对该矩阵进行扩充维度 [B, num_window, num_patch_in_window, 1]
    - 将该矩阵与其转置矩阵进行做差, 得到同类关系矩阵 (结果 0 的位置上的 patch 所属同一个 window, 否则不属于同一个 window)
    - 对同类关系矩阵中的非零位置用负无穷数(-1e9)进行填充, 对于零的位置用 0 填充, 这样构建好了对应的 MHSA 所需要的 mask
    - 此 mask 的形状为 [B, num_window, num_patch_in_window, patch_depth]
7. 将 window 转换为三维格式, [B*num_window, num_patch_in_window, patch_depth]
8. 将三维格式的特征图和其对应的 mask 传入 MHSA 进行计算得到注意力输出
9. 将注意力输出转换为图像 patch 格式, [B, num_window, num_patch_in_window, patch_depth]
10. 为了恢复位置, 需要将图像的 patch 往右边和下边各自滑动半个窗口大小的步长, 完成 SW-MHSA 计算


```python
def shift_window_multi_head_self_attention(w_msa_output, mhsa, window_size=4, num_head=2):
    batch_size, num_windows, num_patch_in_window, patch_depth = w_msa_output.shape

    shifted_w_msa_input, additive_mask = shift_window(w_msa_output, window_size, shift_size=window_size//2, generate_mask=True)

    shifted_w_msa_input = shifted_w_msa_input.reshape(batch_size*num_windows, num_patch_in_window, patch_depth)

    _, output = mhsa(shifted_w_msa_input, additive_mask=additive_mask)
    # attn_prob, output = mhsa(shifted_w_msa_input, additive_mask=additive_mask)
    output = output.reshape(batch_size, num_windows, num_patch_in_window, patch_depth)

    output, _ = shift_window(output, window_size, shift_size=window_size//2, generate_mask=False)
    return output

def shift_window(w_msa_output, window_size, shift_size, generate_mask=False):
    """定义辅助函数 shift_window 即高效计算 SW-MSA"""
    batch_size, num_window, num_patch_in_window, patch_depth = w_msa_output.shape

    w_msa_output = window2image(w_msa_output) # [B, depth, h, w]
    batch_size, patch_depth, image_height, image_width = w_msa_output.shape

    rolled_w_msa_output = torch.roll(w_msa_output, shifts=(shift_size, shift_size), dims=(2, 3))

    shifted_w_msa_input = rolled_w_msa_output.reshape(batch_size, patch_depth,
                                                    int(math.sqrt(num_window)),
                                                    window_size,
                                                    int(math.sqrt(num_window)),
                                                    window_size)

    shifted_w_msa_input = shifted_w_msa_input.transpose(3, 4)
    shifted_w_msa_input = shifted_w_msa_input.reshape(batch_size, patch_depth, num_window*num_patch_in_window)
    shifted_w_msa_input = shifted_w_msa_input.transpose(-1, -2) # [B, num_window*num_patch_in_window, patch_depth]
    shifted_window = shifted_w_msa_input.reshape(batch_size, num_window, num_patch_in_window, patch_depth)

    if generate_mask:
        additive_mask = build_mask_for_shifted_wmsa(batch_size, image_height, image_width, window_size)
    else:
        additive_mask = None

    return shifted_window, additive_mask

def window2image(msa_output):
    """定义辅助函数 window2image, 将 transformer block 的结果转化为 image 格式"""
    batch_size, num_window, num_patch_in_window, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_width = image_height = int(math.sqrt(num_window)) * window_size

    msa_output = msa_output.reshape(batch_size, int(math.sqrt(num_window)), 
                                                int(math.sqrt(num_window)), 
                                                window_size,
                                                window_size,
                                                patch_depth)
    msa_output = msa_output.transpose(2, 3)
    image = msa_output.reshape(batch_size, image_height*image_width, patch_depth)

    image = image.transpose(-1, -2).reshape(batch_size, patch_depth, image_height, image_width) # 和卷积格式一致

    return image

def build_mask_for_shifted_wmsa(batch_size, image_height, image_width, window_size):
    """构建 shift window multi-head attention mask."""
    index_matrix = torch.zeros(image_height, image_width)

    for i in range(image_height):
        for j in range(image_width):
            row_times = (i + window_size // 2) // window_size
            col_times = (j + window_size // 2) // window_size
            index_matrix[i, j] = row_times*(image_height//window_size) + col_times + 1
    
    rolled_index_matrix = torch.roll(index_matrix, shifts=(-window_size//2, -window_size//2), dims=(0, 1))
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0) # [B, C, H, W]

    c = F.unfold(rolled_index_matrix, kernel_size=(window_size, window_size),
                                                    stride=(window_size, window_size)).transpose(-1, -2)
    c = c.tile(batch_size, 1, 1) # [B, num_window, num_patch_in_window]

    batch_size, num_window, num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1) # [B, num_window, num_patch_in_window, 1]
    c2 = (c1 - c1.transpose(-1, -2)) == 0 # [B, num_window, num_patch_in_window, num_patch_in_window]
    valid_matrix = c2.to(torch.float32)
    additive_mask = (1 - valid_matrix) * (-1e-9) # [B, num_window, num_patch_in_window, num_patch_in_window]

    additive_mask = additive_mask.reshape(batch_size*num_window, num_patch_in_window, num_patch_in_window)

    return additive_mask
```

#### Step 5. 如何构建 Patch Merging ?
1. 将 window 格式的特征转换为图像 patch 格式
2. 利用 unfold 操作, 按照 merge_size * merge_size 的大小得到新的 patch, shape=[B, num_patch_new, merge_size*merge_size*patch_depth_old]
3. 使用一个全连接层对 depth 进行降维成为原来的一半, convert[merge_size*merge_size*patch_depth_old]--->[merge_size*merge_size*patch_depth_old*0.5]
4. 输出的是 patch_embedding 形状格式: [batch_size, num_patch, patch_depth]
5. example: merge_size=2, 经过 PatchMerge, num_patch(seq_len) 减少为原来的 1/4, patch_depth(model_dim_C) 增大为原来的 2 倍数, 而不是 4 倍


```python
class PatchMerging(nn.Module):
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5):
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(model_dim*merge_size*merge_size, int(model_dim*merge_size*merge_size*output_depth_scale))

    def forward(self, input):
        batch_size, num_window, num_patch_in_window, patch_depth = input.shape
        window_size = int(math.sqrt(num_patch_in_window))

        input = window2image(input) # [B, patch_depth, image_h, image_w]

        merged_window = F.unfold(input, kernel_size=(self.merge_size, self.merge_size),
                                stride=(self.merge_size, self.merge_size)).transpose(-1, -2)
        merged_window = self.proj_layer(merged_window)

        return merged_window # [batch_size, num_patch_new, patch_depth_new]
```

#### Step 6. 如何构建 SwinTransformerBlock ?
1. 每个 swin block 包含 LayerNorm, Window-MHSA, MLP, Shift-Window-MHSA, shoutcut残差连接 模块
2. 输入是 shape:[batch_size, num_patch, model_dim_C] 的 patch embedding 格式
3. 每个 MLP 层包含两层, 对 model_dim_C 进行 4 倍升维度(4*model_dim_C), 然后有降低到 model_dim_C
4. 输出的是 window 数据格式, shape:[batch_size, num_window, num_patch_in_window, patch_depth]
5. 需要注意残差连接 shortcut 对形状的要求

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, model_dim, window_size, num_head):
        super(SwinTransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.layer_norm4 = nn.LayerNorm(model_dim)

        self.wsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.wsma_mlp2 = nn.Linear(4*model_dim, model_dim)
        self.swsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.swsma_mlp2 = nn.Linear(4*model_dim, model_dim)

        self.mhsa1 = MultiHeadSelfAttention(model_dim, num_head)
        self.mhsa2 = MultiHeadSelfAttention(model_dim, num_head)

    def forward(self, input):
        batch_size, num_patch, patch_depth = input.shape

        input1 = self.layer_norm1(input)
        w_msa_output = window_multi_head_self_attention(input1, self.mhsa1, window_size=4, num_head=2)
        batch_size, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
        w_msa_output = input + w_msa_output.reshape(batch_size, num_patch, patch_depth)
        output1 = self.wsma_mlp2(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        output1 += w_msa_output

        input2 = self.layer_norm3(output1)
        input2 = input2.reshape(batch_size, num_window, num_patch_in_window, patch_depth)
        sw_msa_output = shift_window_multi_head_self_attention(input2, self.mhsa2, window_size=4, num_head=2)
        sw_msa_output = output1 + sw_msa_output.reshape(batch_size, num_patch, patch_depth)
        output2 = self.swsma_mlp2(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        output2 += sw_msa_output

        output2 = output2.reshape(batch_size, num_window, num_patch_in_window, patch_depth)

        return output2
```

#### Step 7. 如何构建 SwinTransformerModel ?
1. 输入是图像数据 image
2. 首先对图像进行分块得到 patch embedding
3. 经过第一个 stage
4. 进行 patch merging, 在经过第二个 stage
5. 以此类推 ......
6. 对最后一个 block 的输出转换为 patch embedding 格式 [batch_size, num_patch, patch_depth]
7. 对 patch embedding 在时间维度进行平均池化, 并映射到分类层得到分类的 logits, 进行交叉熵损失优化

```python
class SwinTransformerModel(nn.Module):
    def __init__(self, input_image_channel=3, patch_size=4, model_dim_C=8, num_classes=10,
                window_size=4, num_head=2, merge_size=2):
        super(SwinTransformerModel, self).__init__()
        patch_depth = patch_size*patch_size*input_image_channel
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth, model_dim_C))
        self.block1 = SwinTransformerBlock(model_dim_C, window_size, num_head)
        self.block2 = SwinTransformerBlock(model_dim_C*2, window_size, num_head)
        self.block3 = SwinTransformerBlock(model_dim_C*4, window_size, num_head)
        self.block4 = SwinTransformerBlock(model_dim_C*8, window_size, num_head)

        self.patch_merging1 = PatchMerging(model_dim_C, merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C*2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C*4, merge_size)

        self.final_layer = nn.Linear(model_dim_C*8, num_classes)

    def forward(self, image):
        patch_embedding_naive = image2embedding_naive(image, self.patch_size, self.patch_embedding_weight)
        print(f"image to patch embedding shape: {patch_embedding_naive.shape}")

        # ---------------- stage 1 ----------------
        patch_embedding = patch_embedding_naive
        sw_msa_output = self.block1(patch_embedding)
        print(f"stage 1 output shape: {sw_msa_output.shape}")

        # ---------------- stage 2 ----------------
        merged_patch1 = self.patch_merging1(sw_msa_output)
        sw_msa_output_1 = self.block2(merged_patch1)
        print(f"stage 2 output shape: {sw_msa_output_1.shape}")
        
        # ---------------- stage 3 ----------------
        merged_patch2 = self.patch_merging2(sw_msa_output_1)
        sw_msa_output_2 = self.block3(merged_patch2)
        print(f"stage 3 output shape: {sw_msa_output_2.shape}")
        
        # ---------------- stage 4 ----------------
        merged_patch3 = self.patch_merging3(sw_msa_output_2)
        sw_msa_output_3 = self.block4(merged_patch3)
        print(f"stage 4 output shape: {sw_msa_output_3.shape}")
        
        # ---------------- final classification ----------------
        batch_size, num_window, num_patch_in_window, patch_depth = sw_msa_output_3.shape
        sw_msa_output_3 = sw_msa_output_3.reshape(batch_size, -1, patch_depth) # [batch_size, num_patch, patch_depth]
        print(f"final output shape: {sw_msa_output_3.shape}")

        pool_output = torch.mean(sw_msa_output_3, dim=1) # [batch_size, patch_depth]
        logits = self.final_layer(pool_output) # [batch_size, num_classes]
        print(f"logits: {logits.shape}")

        return logits


# --------------------------
if __name__ == "__main__":
    """ Step 8. 测试模型代码 """
    batch_size, in_channels, image_h, image_w = 16, 3, 256, 256
    patch_size = 4
    model_dim_C = 512
    # max_num_token = 16
    num_classes = 10
    window_size = 4
    num_head = 8
    merge_size = 2

    patch_depth = patch_size*patch_size*in_channels
    image = torch.randn(batch_size, in_channels, image_h, image_w)

    model = SwinTransformerModel(in_channels, patch_size, model_dim_C, num_classes, window_size, num_head, merge_size)

    logits = model(image)
    print(logits)
```


----------------------------
### Masked AutoEncoder

<center class="half">
  <img src="./images/mae.png"/>
</center>

> [MAE original source code on GitHub of FacebookResearch](https://github.com/facebookresearch/mae)

**Masked AutoEncoder, MAE**

1. data processing
    - image2tensor: max-mix normalization; image-min / max-min
    - augment: crop; resize; flip
    - convert: [0-255] ---> [0-1]
    - normalize: [image-mean_global_channel] / std_global_channel
    - ImageNet1k: mean[0.485, 0.456, 0.406]; std[0.229, 0.224, 0.225]; ~N(0, 1)
2. model architecture
    - encoder: image2patch2embedding; position embedding; random masking(shuffle); class token; Transformer Blocks
    - decoder: projection layer; unshuffle; position embedding; Transformer Blocks; regression layer; mse loss function(noem pixel)
3. forward functions
    - forward encoder
    - forward decoder
    - forward loss
4. training
    - dataset and dataloader
    - model and optimizer
    - load model and checkpoint
    - train on epoch and train on step
    - save model and save checkpoint
5. finetuning
    - strong augmentation
    - build encdoer + BN + MLP calssifier head
    - interpolate position embedding
    - load pre-trained model (stric=False)
    - update all parameters
    - AdamW optimizer
    - label smoothing cross-entropy loss
6. linear probing
    - weak augmentation
    - build encoder +BN(no affine) + MLP classifier head
    - interpolate position embedding
    - load pre-trained model (strict=False)
    - only update parameters of MLP classifier head
    - LARS optimizer
    - cross-entropy loss
7. evaluation
    - with torch.no_grad() to efficient
    - model.eval to accurate BN/dropout (R-dropout paper)
    - top-1 adn top-5


----------------------------
### Position Embedding

**Position Embedding**
- Transformer
    - one-dimension absolute position encoding
    - sin/cos constant
- Vision Transformer
    - one-dimension absolute position encoding
    - learning and trainable
- Swin Transformer
    - two-dimension relative position encoding
    - learning and trainable
- Masked AutoEncoder
    - two-dimension absolute position encoding
    - sin/cos constant

```python
#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Position Embedding
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-13
"""

import torch
import torch.nn as nn

# position embedding (PE) for Transformer
def create_1d_absolute_sincos_embedding(n_pos_vec, dim):
    # n_pos_vec: torch.arange(n_pos)
    assert dim % 2 == 0, "wrong dimension!" 
    position_embedding = torch.zeros(torch.numel(n_pos_vec), dim, dtype=torch.float)
    # position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float)

    omega = torch.arange(dim/2, dtype=torch.float)
    omega /= dim / 2
    omega = 1. / (10000 ** omega)

    out = n_pos_vec[:, None] @ omega[None, :]
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    position_embedding[:, 0::2] = emb_sin
    position_embedding[:, 1::2] = emb_cos

    return position_embedding


# position embedding (PE) for Vision Transformer
def create_1d_absolute_trainable_embedding(n_pos_vec, dim):
    position_embedding = nn.Embedding(torch.numel(n_pos_vec), dim)
    # position_embedding = nn.Embedding(n_pos_vec.nueml(), dim)
    nn.init.constant_(position_embedding.weight, 0.)

    return position_embedding


# position embedding (PE) for Swin Transformer
def create_2d_relative_bias_trainable_embedding(num_head, height, width, dim):
    # width: 5, [0, 1, 2, 3, 4]; bias=[-width+1, width-1], 2*width-1 
    # height: 5, [0, 1, 2, 3, 4]; bias=[-height+1, height-1], 2*height-1 
    position_embedding = nn.Embedding((2*width-1)*(2*height-1), num_head)
    nn.init.constant_(position_embedding.weight, 0.)

    def get_2d_relative_position_index(height, width):
        coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width))) # [2, height, width]
        coords_flatten = torch.flatten(coords, 1) # [2, heigt*width]

        relative_coords_bias = coords_flatten[:, :, None] -  coords_flatten[:, None, :] # [2, heigt*width, heigt*width]
        relative_coords_bias[0, :, :] += height - 1
        relative_coords_bias[1, :, :] += width - 1

        relative_coords_bias[0, :, :] *= relative_coords_bias[1, :, :].max() + 1

        return relative_coords_bias.sum(0) # [heigt*width, heigt*width]

    relative_position_bias = get_2d_relative_position_index(height, width)
    bais_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height*width, height*width, num_head) # [heigt*width, heigt*width, num_head]

    bais_embedding = bais_embedding.permute(2, 0, 1).unsqueeze(0) # [1, num_head, heigt*width, heigt*width]
    return bais_embedding


# position embedding (PE) for Mask AutoEncoder
def create_2d_absolute_sincos_embedding(height, width, dim):
    assert dim % 4 == 0, "wrong dimension!" 
    position_embedding = torch.zeros(height*width, dim)

    coords = torch.stack(torch.meshgrid(torch.arange(height, dtype=torch.float), torch.arange(width, dtype=torch.float)))

    height_embedding = create_1d_absolute_sincos_embedding(torch.flatten(coords[0]), dim//2) # [H*W, dim//2]
    width_embedding = create_1d_absolute_sincos_embedding(torch.flatten(coords[1]), dim//2) # [H*W, dim//2]

    position_embedding[:, :dim//2] = height_embedding
    position_embedding[:, dim//2:] = width_embedding

    return position_embedding


# --------------------------
if __name__ == "__main__":
    n_pos = 4
    dim = 4
    num_head = 2
    height = 4
    width = 4
    n_pos_vec = torch.arange(n_pos, dtype=torch.float)    

    # Transformer
    pe = create_1d_absolute_sincos_embedding(n_pos_vec, dim)
    print(pe.shape)

    # Vision Transformer
    pe_vit = create_1d_absolute_trainable_embedding(n_pos_vec, dim)
    print(pe_vit.weight.shape)

    # Swin Transformer
    pe_swin = create_2d_relative_bias_trainable_embedding(num_head, height, width, dim)
    print(pe_swin.shape)

    # Masked AutoEncoder
    pe_mae = create_2d_absolute_sincos_embedding(height, width, dim)
    print(pe_mae.shape)
```


### Reference
----------------------------

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need," NeurIPS'2017

[Transformer Paper on NeurIPS'2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
&emsp;&emsp;[Transformer Paper on arXiv'2017](https://arxiv.org/abs/1706.03762)
&emsp;&emsp;[Paper Original Code on GitHub](https://github.com/tensorflow/tensor2tensor)
&emsp;&emsp;[Paper Code on GitHub](https://github.com/huggingface/transformers)

[2] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR'2021

[ViT Paper on ICLR'2021](https://iclr.cc/virtual/2021/poster/3013)
&emsp;&emsp;[ViT Paper on arXiv'2020](https://arxiv.org/abs/2010.11929)
&emsp;&emsp;[ViT Original Code on GitHub](https://github.com/google-research/vision_transformer)
&emsp;&emsp;[Huggingface Transformer Code on GitHub](https://github.com/huggingface/transformers)

[3] Ze Liu, Yutong Lin, Yue Cao, Han Hu, et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV'2021

[Swin Transformer Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)
&emsp;&emsp;[Swin Transformer Paper on arXiv'2021](https://arxiv.org/abs/2103.14030)
&emsp;&emsp;[Swin Transformer Original Code on GitHub](https://github.com/microsoft/Swin-Transformer)

[4] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte, "SwinIR: Image Restoration Using Swin Transformer," ICCV'2021

[SwinIR Paper on ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html)
&emsp;&emsp;[SwinIR Paper on arXiv'2021](https://arxiv.org/abs/2108.10257)
&emsp;&emsp;[SwinIR Original Code on GitHub](https://github.com/jingyunliang/swinir)

[5] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, "Masked Autoencoders Are Scalable Vision Learners," CVPR'2022

[MAE Paper on CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)
&emsp;&emsp;[MAE Paper on arXiv'2021](https://arxiv.org/abs/2111.06377)
&emsp;&emsp;[MAE Original Code on GitHub](https://github.com/facebookresearch/mae)
