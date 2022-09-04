# Python environment for Artificial Intelligence (AI) Deep Learning (DL) and Machine Learning (ML)

## Python interpreter 解释器

- Python download site : https://www.python.org/
- Anaconda download site: https://www.anaconda.com/
- Miniconda download site: https://docs.conda.io/en/latest/miniconda.html

## Python package manager 包管理器

- pip package manager and conda package manager
- 推荐使用 conda 进行 python 环境隔离, 使用 pip 进行 python 各种库和包的管理.
- conda 有两个平台, Anaconda and Miniconda, 推荐只用 miniconda, 因为只需要 conda 进行 python 环境隔离.
- Anaconda 默认里面有很多预先安装的科学计算库, 包括 python 解释器和 conda 管理器.
- Miniconda 默认里面只有 python 解释器 (pip 集成在里面)和 conda 管理器

**Conda 包管理环境常用命令**
```shell
# 1. conda 创建新的 python 环境
conda create --name pytorch_env python=3.8

# 2. 查看 conda 管理的 python 环境
conda info --envs

# 3. conda 激活对应的 python 环境
conda activate pytorch_env

# 4. 在对应的 conda 隔离 pytorch_env 环境中安装需要的包和库
pip install numpy, pandas, matplotlib

# 5. 退出 conda 的 python 环境
conda deactivate

# 6. 删除 conda python 环境
conda remove --name pytorch_env --all
```

**pip 对应的包和库在 pypi 这个仓库下, 配置 pypi 镜像源**
- [南京大学 pypi 镜像源](https://mirrors.nju.edu.cn/help/pypi)
- [清华大学 pypi 镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [阿里巴巴 pypi 镜像源](https://developer.aliyun.com/mirror/pypi)
- [华为云 pypi 镜像源](https://mirrors.huaweicloud.com/home)
- [腾讯云 pypi 镜像源](https://mirrors.tencent.com/help/pypi.html)

```shell
# pypi mirrors source for pip
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

- VSCode for Python, 学习的时候或者编写脚本的时候推荐使用
- JupyterLab for Python, 学习的时候或者做教学演示最佳推荐
- VSCode 插件以及技巧推荐 : https://gitee.com/weili_yzzcq/linux_-os/tree/main/vscode
- VSCode 插件以及技巧推荐 : https://github.com/2694048168/Linux_OS/tree/main/vscode
- Vscode Download site: https://code.visualstudio.com/
- PyCharm Community for Python, 做项目调式的使用推荐使用
- PyCharm Download site: https://www.jetbrains.com/pycharm/


## Python env. for AI and Machine Learning and Deep Learning

```shell
# 1. create env.
conda create --name ai_ml_env python=3.8
conda info --envs

# 2. install packages
conda activate ai_ml_env
pip install numpy, pandas, matplotlib
pip install -U scikit-learn

# 3. python requirements.txt
# https://pip.pypa.io/en/stable/cli/pip_freeze/
# 导出 python 配置环境
pip freeze > requirements.txt
# 4. 安装 python 配置环境
pip install -r requirements.txt

# ------------------------------------------
conda create --name pytorch_env python=3.8
# 2. install packages
conda activate pytorch_env
pip install numpy, pandas, matplotlib
# install torch 官网进行选择
# https://pytorch.org/get-started/locally/
# GPU for PyTorch
# CPU for PyTorch
pip install torchkeras

# ------------------------------------------
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
```