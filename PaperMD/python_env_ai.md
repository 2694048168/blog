## Python for Deep Learning form Scratch

- &ensp;<span style="color:MediumPurple">Title</span>: Python for Deep Learning form Scratch
- &ensp;<span style="color:Moccasin">Tags</span>: Python; Miniconda; Git; GitHub
- &ensp;<span style="color:PaleVioletRed">Type</span>: Mini-Survey
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [黎为](https://2694048168.github.io/blog/#/)
- &ensp;<span style="color:DarkMagenta">DateTime</span>: 2023-04-15

> <span style="color:Red">Talk is cheap, show me the code. Quote from Linus Torvalds</span>

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
- [Tmux introduction](https://www.bilibili.com/video/BV1da4y1p7e1/)
- [Configuration of Terminal](https://github.com/2694048168/Linux_OS/tree/main/Configuration)

**Conda 包管理环境常用命令**
```shell
# 1. conda 创建新的 python 环境
conda create --name pytorch_SR python=3.11

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

**pip 包管理常用命令**
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
# https://tensorflow.google.cn/install/source_windows?hl=en#gpu
pip install tensorflow==2.10.0
pip install torch-1.13.0+cu117-cp311-cp311-linux_x86_64.whl
pip install torch-2.0.0+cu117-cp311-cp311-win_amd64.whl

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

**Python 项目管理虚拟环境**

> 无法创建特定 Python 版本的虚拟环境, 但是关于该虚拟环境和项目进行绑定.

```shell
# https://docs.python.org/3/library/venv.html
python -m venv path2venv
python 3.11 -m venv .venv

# VSCode 可以自动识别该虚拟环境
```


**pip 对应的包和库在 pypi 这个仓库下, 配置 pypi 镜像源**
- [南京大学 pypi 镜像源](https://mirrors.nju.edu.cn/help/pypi)
- [清华大学 pypi 镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [阿里巴巴 pypi 镜像源](https://developer.aliyun.com/mirror/pypi)
- [华为云 pypi 镜像源](https://mirrors.huaweicloud.com/home)
- [腾讯云 pypi 镜像源](https://mirrors.tencent.com/help/pypi.html)
- [上海交通大学 conda 镜像源](https://mirrors.sjtug.sjtu.edu.cn/docs/anaconda)
- [清华大学 conda 镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

```shell
# pypi mirrors source for pip
# 临时使用 pip 镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy

# 一键永久配置 pip 镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://developer.aliyun.com/mirror/pypi/simple

# 也可以配置文件 .pip/config? how to modify？
# 太麻烦了, 一条命令可以搞定的事情, why not？

# ---------------------
# 配置完后需要清除缓存
conda clean -i
# ---------------------
# 编辑 ~/.condarc
default_channels:
  - https://mirror.sjtu.edu.cn/anaconda/pkgs/r
  - https://mirror.sjtu.edu.cn/anaconda/pkgs/main
custom_channels:
  conda-forge: https://mirror.sjtu.edu.cn/anaconda/cloud/
  pytorch: https://mirror.sjtu.edu.cn/anaconda/cloud/
channels:
  - defaults
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

**VSCode and Pycharm Interpreter**

- Ctrl + Shift + P ---> Select Interpreter ---> configure python Interpreter
- File ---> Settings ---> Project ---> Python Interpreter ---> add & configure
- anyway, google search
- [DDP Exmaple PyTorch](https://github.com/pytorch/examples/tree/main/distributed/ddp)
- [DDP Tutorial PyTorch](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series)
- 多GPU分布式训练: PyTorch DDP ---> Huggingface Accelerate library
- [Huggingface Accelerate](https://github.com/huggingface/accelerate) 

**Git and GitHub**

- [1小时学会 Git 视频](https://www.bilibili.com/video/BV1JM4y1k7Pq/)
- [Learning Git quick start](https://www.bookstack.cn/read/learngit-basic/336eaf68268808a4.md)
- [GitHub Proxy](https://ghproxy.com/)

<img src="https://git-scm.com/images/about/index1@2x.png" width="50%" /><img src="https://git-scm.com/images/about/index2@2x.png" width="50%" />

> download and install Git

```powershell
# 安装或者升级 Git by PowerShell on Windows Terminal
winget install --id Git.Git -e --source winget
```

```shell
# download for Windows
# https://git-scm.com/download/win
# https://gitforwindows.org/
# 正常软件安装, 安装时需要注意环境变量的添加,以及使用者
# 否则需要自己手动配置环境变量,才能在终端使用

# 全局配置 git 基本信息(--system, --global, --local)
git config --global user.name "Wei Li"
git config --global user.email "weili_yzzcq@163.com"
git config --global core.editor "code --wait"
# 以默认的编辑器, 查看全局配置文件
git config --global -e
# Windows "\r\n(CRLF)" ---> true
# Linux and Mac "\n(LR)" ---> input
git config --global core.autocrlf true # Windows
git config --global core.autocrlf input # Mac or Linux

# 查看 git 配置信息
git config --global --list
git config --list

# 设置和恢复 git HTTPS 协议代理(SSH协议单独配置)
git config --global http.proxy 'socks5://127.0.0.1:7891' 
git config --global https.proxy 'socks5://127.0.0.1:7891'
git config --global --unset http.proxy
git config --global --unset https.proxy

# git SSH协议单独配置文件写入内容(~/.ssh/config)
# 全局 
ProxyCommand nc -X 5 -x 127.0.0.1:7891 %h %p
# 只为特定域名设定
Host github.com
    ProxyCommand nc -X 5 -x 127.0.0.1:7891 %h %p

# ----------------------
# install git on Debian and Ubuntu(apt or nala)
# https://gitlab.com/volian/nala/-/wikis/Installation
# you can select the fast mirrors by nala
sudo nala fetch
sudo nala update && sudo nala upgrade
sudo nala install git

# CentOS
sudo yum install git
# Arch Linux and Manjaro
sudo pacman -Sy git

# install git on MacOS(brew)
# https://brew.sh/
# https://gitee.com/cunkai/HomebrewCN/
brew install git
```

> we can write a shell or python script to configure.

```shell
# https://www.bilibili.com/video/BV1r3411F7kn/
# 增删查改基本操作
# how to read the help information of Command line
git --help

# git clone github repo.
git clone https://ghproxy.com/https://github.com/2694048168/ComputerVisionDeepLearning.git
# HTTPS and SSH 协议(涉及科学上网方式)
git clone https://github.com/2694048168/ComputerVisionDeepLearning.git
# all branchs default or special branch
git clone --branch <branchname> <remote-repo-url>
git clone -b <branchname> <remote-repo-url>
# 本地分支且只能跟踪此分支
git clone --branch <branchname> --single-branch <remote-repo-url>
git clone -b <branchname> --single-branch <remote-repo-url>

# git 仓库项目中依然引用其他 git 仓库时候, 需要递归克隆
git clone --recursive https://github.com/NVlabs/instant-ngp.git
cd instant-ngp
git clone --recursive https://github.com/NVlabs/instant-ngp.git -b master NGP
cd NGP

# simple pipeline: git add & commint & push
mkdir git_folder
cd git_folder

# 生成隐藏文件夹 .git/ 用于管理和记录和 Git 相关的信息
# 初始化 git 代码仓库版本控制(默认为master or main branch)
git init .
# git push -u origin <本地分支名> github 空白仓库的创建后有指引

# 查看 git 仓库里面的状态
git status

# 提交源代码文件到暂缓区
git add new_file
git add .

# 提交源代码文件到本地仓库
git commit # 以默认编辑器打开提交信息文件, 顶部commit
git commit -m "commit information"
# add and commit
git commit -am "commit information"
git commit -a # 打开默认编辑器填写提交描述

# 提交源代码文件到远程仓库(Gitee or GitHub, et al.)
# git push origin_repo_name<default main branch>
# git push origin <本地分支名>:<远程分支名>
git push origin_repo_name
# 注: 文件大小有限制, GitHub中模型的权重文件都是给的链接形式

# 更新本地仓库(library from other)
# learning from --help option
git pull
git fetch

# 对比工作区和暂缓区的文件差异
git diff
# 对比暂缓区的版本文件间差异
git diff --staged
# 利用一些可视化工具进行对比查看, 如 VSCode
# git difftool --tool-help
git config --global diff.tool vscode
git config --global difftool.vscode.cmd "code --wait --diff $LOCAL $REMOTE"
git config --global -e
git difftool
git difftool --staged

# 冲突解决 conflict
# git mergetool --tool-help
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd "code --wait $MERGED"
# '$' 变量符号好像无法解析, 需要检查一下配置文件, 添加完整内容
git config --global -e

# 查看 git 项目关联的远程仓库信息
git remote --verbose
git remote -v
git remote --help
# 添加/移除远程仓库并指定名称为 'mirror'
git remote add mirror https://github.com/NVlabs/instant-ngp.git
git remote remove mirror

# 查看 git 提交日志, q 推出日志查看状态
git log
git log oneline
git log --pretty=oneline
git show HEAD
git show "hash-id"
# 'tree' 表示是一个目录, 'blob' 表示是一个文件
# 'tag' and 'commit'
git ls-tree HEAD

# 让 git 忽略文件方式, 有规则
# https://github.com/github/gitignore
touch .gitignore

# 如何和 GitHub and Gitee 进行关联
# 分支操作; 
# 从暂缓区删除文件; 从本地仓库删除文件; 从远程仓库删除文件; 从磁盘上删除文件
rm file.txt
git commit -am "delete file.txt"
git ls-files # 显示暂存区的文件

git rm -h
git rm file
git commit m "delete file.txt"

mv file.txt main.cpp
git mv file.txt main.cpp
```


**Tmux 分离会话和终端 训练模型**

```shell
# 1. 安装 git tmux, 可以通过源码安装方式获取最新？
sudo nala update && sudo nala upgrade
sudo nala install tmux git cmake
# 查看 tmux 版本
tmux -V

# 2. 新建一个 session, 并设置 session 的名称为 'training'
tmux new -s training

# and then training model on Server
python train.py

# 分离会话, 此时模型的训练依然需要进行
# or enter 'Ctrl + b', 'd'
tmux detach

# 查看当前所有 session
tmux ls
tmux list-session

# 接入已有的会话 'training'
tmux attach -t training

# -------------------------
# kill session 'training'
tmux kill-session -t training

# tmux switch 命令用于切换会话
tmux new -s model
tmux new -s task
tmux
tmux switch -t task
tmux switch -t model

# -------------------------------------
# 3. 安装 NeoVim, 以满足 LunarVim 的需求
```
