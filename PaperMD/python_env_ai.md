# Python environment for artificial intelligence (AI) Deep Learning (DL) and Machine Learning (ML)

## Python interpreter 解释器

> python download site : https://www.python.org/

## Python package manager 包管理器

> pip package manager and conda package manager

> 推荐使用 conda 进行 python 环境隔离, 使用 pip 进行 python 各种库和包的管理.

> conda 有两个平台, Anaconda and Miniconda, 推荐只用 miniconda, 因为只需要 conda 进行 python 环境隔离.

> Anaconda 默认里面有很多预先安装的科学计算库, 包括 python 解释器和 conda 管理器.

> Download sit: https://www.anaconda.com/

> Miniconda 默认里面只有 python 解释器 (pip 集成在里面)和 conda 管理器

> Download sit: https://docs.conda.io/en/latest/miniconda.html

```shell
# 1. conda 创建新的 python 环境
conda create --name pytorch_env python=3.8

# 2. 查看 conda 管理的 python 环境
conda info --envs

# 3. conda 激活对应的 python 环境
conda activate pytorch_env

# 4. 在对应的 conda 隔离 pytorch_env 环境中安装需要的包和库
pip install numpy, pandas, matplotlib
pip install torch

# 5. 退出 conda 的 python 环境
conda deactivate

# 6. 删除 conda python 环境
conda remove --name pytorch_env --all
```

> pip 对应的包和库在 pypi 这个仓库下, 配置 pypi 镜像源.

```shell
# pypi mirrors source for pip

# 清华大学的 pypi 镜像源
# 1. https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

# 清华大学的 conda 镜像源
# 2. https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/ 

# 阿里巴巴 pypi 镜像源
# 3. https://developer.aliyun.com/mirror/pypi

# 华为云镜像源
# 4. https://mirrors.huaweicloud.com/home

# 临时使用 pip 镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy

# 一键永久配置 pip 镜像源
python -m pip install --upgrade pip

# 清华大学镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里巴巴镜像源
pip config set global.index-url https://developer.aliyun.com/mirror/pypi/simple

```

## Python editor 编辑器

> VSCode for Python, 学习的时候或者编写脚本的时候推荐使用

> VSCode 插件以及技巧推荐 : https://gitee.com/weili_yzzcq/linux_-os/tree/main/vscode

> VSCode 插件以及技巧推荐 : https://github.com/2694048168/Linux_OS/tree/main/vscode

> Download site: https://code.visualstudio.com/

> PyCharm Community for Python, 做项目调式的使用推荐使用

> Download site: https://www.jetbrains.com/pycharm/


## Python env. for AI and Machine Learning 

```shell
# 1. create env.
conda create --name ai_ml_env python=3.8

# 2. install packages
conda activate ai_ml_env
pip install numpy, pandas, matplotlib
pip install -U scikit-learn

# 3. vscode for python scripts

# 4. python requirements.txt
# https://pip.pypa.io/en/stable/cli/pip_freeze/
# 导出 python 配置环境
pip freeze > requirements.txt

# 安装 python 配置环境
pip install -r requirements.txt

```

## Python env. for Deep Learning with PyTorch 

```shell
# 1. create env.
conda create --name pytorch_env python=3.8

# 2. install packages
conda activate pytorch_env
pip install numpy, pandas, matplotlib

# install torch 官网进行选择
# https://pytorch.org/get-started/locally/

# GPU for PyTorch
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# CPU for PyTorch
pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

pip install torchkeras

# 5. vscode for python scripts

# 6. python requirements.txt
# https://pip.pypa.io/en/stable/cli/pip_freeze/
# 导出 python 配置环境
pip freeze > requirements.txt

# 安装 python 配置环境
pip install -r requirements.txt

```

## Python env. for Deep Learning with TensorFlow

```shell
# 1. create env.
conda create --name tensorflow_env python=3.8

# 2. install packages
conda activate tensorflow_env
pip install numpy, pandas, matplotlib

# install tensorflow 官网进行选择
# https://tensorflow.google.cn/install/pip
# TensorFlow 2.0 CPU and GPU 都支持, TensorFlow 1.0 的 CPU and GPU version are separate.
pip install tensorflow
pip install tensorboard

# 5. vscode for python scripts

# 6. python requirements.txt
# https://pip.pypa.io/en/stable/cli/pip_freeze/
# 导出 python 配置环境
pip freeze > requirements.txt

# 安装 python 配置环境
pip install -r requirements.txt

```