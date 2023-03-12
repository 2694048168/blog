## Python for Deep Learning and Machine Learning

**Python interpreter**
- [Python download site](https://www.python.org/)
- [Anaconda download site](https://www.anaconda.com/)
- [Miniconda download site](https://docs.conda.io/en/latest/miniconda.html)

**Python package manager**
- pip package manager and conda package manager
- 推荐使用 conda 进行 python 环境隔离, 使用 pip 进行 python 各种库和包的管理
- conda 有两个平台, Anaconda and Miniconda, 推荐只用 miniconda, 因为只需要 conda 进行 python 环境隔离
- Anaconda 默认里面有很多预先安装的科学计算库, 包括 python 解释器和 conda 管理器
- Miniconda 默认里面只有 python 解释器 (pip 集成在里面)和 conda 管理器(not forgetting path2env)
- Windows Terminal + Powershell + oh-my-posh on Windows Platform(not windows powershell)
- Ubuntu Termianal + Tmux + Zsh + oh-my-zsh on Linux Platform
- iTerm2 + Tmux + Zsh + oh-my-zsh on Mac OS Platform
- bing and google search engine, not baidu

**Conda 包管理环境常用命令**
```shell
# 1. conda 创建新的 python 环境
conda create --name pytorch_SR python=3.10

# 2. 查看 conda 管理的 python 环境
conda info --envs

# 3. conda 激活对应的 python 环境
conda activate pytorch_SR

# 4. 在对应的 conda 隔离 pytorch_SR 环境中安装需要的包和库
pip install numpy, pandas, matplotlib

# 5. 退出 conda 的 python 环境
conda deactivate

# 6. 删除 conda python 环境
conda remove --name pytorch_SR --all
```

**Conda 包管理环境常用命令**
```shell
# 查看 pip 包管理器版本
pip --vesion

# 更新 pip 包管理器
python -m pip install –upgrade pip

# pip 搜索 python package
# 网址: https://pypi.org/search/
pip search numpy

# pip 安装 python package online and offline
pip install numpy
pip install torch-1.13.0+cu117-cp311-cp311-linux_x86_64.whl

# pip 卸载 python package
pip uninstall numpy

# pip 查看 python package 相关信息
pip show numpy

# pip 显示该环境下所安装的所有 python package
pip list --help
pip list

# pip 导出该环境下所安装的所有 python package 到配置文件
# Output installed packages in requirements format.
pip freeze > requirements.txt

# pip 根据其配置文件安装环境
pip install --help
pip install -r requirements.txt
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
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://developer.aliyun.com/mirror/pypi/simple

# 也可以配置文件 .pip/config? how to modify？
# 太麻烦了, 一条命令可以搞定的事情, why not？
```

**Python editor**
- VSCode for Python, 学习的时候或者编写脚本的时候推荐使用(Best Recommendation)
- JupyterLab for Python, 学习的时候或者做教学演示最佳效果
- [VSCode 插件以及技巧推荐 Gitee](https://gitee.com/weili_yzzcq/linux_-os/tree/main/vscode)
- [VSCode 插件以及技巧推荐 GitHub](https://github.com/2694048168/Linux_OS/tree/main/vscode)
- [Vscode Download site](https://code.visualstudio.com/)
- PyCharm Community for Python, the Project and Paper Code with Debug(Best Recommendation)
- [PyCharm Download site](https://www.jetbrains.com/pycharm/)


**PyTorch and TensorFlow Configuration for Deep Learning**

> we can write a shell or python script to configure.

```shell
# 1. create env. python version 3.11
conda create --name pytorch python=3.10
conda info --envs

# 2. install packages
conda activate pytorch
pip install -r requirements.txt
pip install scipy numpy pandas matplotlib

# 3. install torch 官网进行选择
# https://pytorch.org/get-started/locally/
# https://download.pytorch.org/whl/ || torch and torchvision version matching
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-1.13.0+cu117-cp311-cp311-linux_x86_64.whl

conda dectivate

# ------------------------------------------
# 1. create env. python version 3.11
conda create --name tensorflow python=3.10

# 2. install packages
conda activate tensorflow
pip install scipy numpy pandas matplotlib

# 3. install tensorflow 官网进行选择
# TensorFlow 支持 Nvidia GPU 需要单独配置 CUDA and cuDNN 等环境
# https://tensorflow.google.cn/install/pip
# https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-2.6.0-cp39-cp39-win_amd64.whl
pip install tensorflow
pip install tensorflow_gpu-2.6.0-cp39-cp39-win_amd64.whl
pip install tensorboard
```

**VSCode and Pycharm**

- Ctrl + Shift + P ---> Select Interpreter ---> configure python Interpreter
- File ---> Settings ---> Project ---> Python Interpreter ---> add & configure
- anyway, google search
