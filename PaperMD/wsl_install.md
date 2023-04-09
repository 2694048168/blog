## WSL2 install on Windows

**WSL2 install**

- [WSL install](https://learn.microsoft.com/zh-cn/windows/wsl/install)
- [WSL2 Linux 内核更新包](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)

```powershell
# 管理员打开 powershell
wsl --install

# install Ubuntu-22.04
wsl --install -d Ubuntu-22.04

# 下载并更新 WSL2 内核, 直接运行下载好的文件
# 在 Windows Terminal 可以直接打开 Ubuntu-22.04 终端
# 设置 用户名和密码, 同时可以使用 VSCode 进行连接 WSL
```

**Ubuntu 基本配置**

```bash
# 0. 设置 root 密码
sudo password

# 1. 更新 Ubuntu 镜像源
# 通过 VSCode 连接便以编辑配置文件
su root
pwd
touch sources.list
rm /etc/apt/sources.list
mv ./sources.list /etc/apt/
# Ctrl + d 退出 root 登录, 回到 'weili' 用户

# 2. 更新索引和所有软件
sudo apt update
sudo apt install nala
sudo nala update && sudo nala upgrade

# 利用 nala 继续更换镜像源
# you can select the fast mirrors by nala
sudo nala fetch
```

**zsh 配置**

```shell
# 1. 检查当前可用的 shell
cat /etc/shells

# 查看当前使用的 shell
echo $SHELL
# 2. 安装 zsh shell
sudo nala install zsh -y

# 3. 查看 shell 版本 切换默认使用 zsh
# 需要重新 log out 才能生效
zsh --version
which zsh
chsh -s /usr/bin/zsh

# https://gitee.com/mirrors/oh-my-zsh
# 4. 安装 oh-my-zsh, 配置 zsh
# sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
git clone --recursive https://gitee.com/mirrors/oh-my-zsh.git
# /home/weili/oh-my-zsh/tools
cd oh-my-zsh/tools/
bash install.sh
rm -rf oh-my-zsh

# 5. [zsh 一系列插件支持](https://github.com/zsh-users)
# 下载 zsh-syntax-highlighting 语法高亮插件
git clone --recursive https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-syntax-highlighting

# 下载 zsh-autosuggestions 自动提示插件
git clone --recursive https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-autosuggestions

# 下载 zsh-completions 自动补全命令插件
git clone --recursive https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions


# 6. 配置 .zshrc文件 更换默认主题为: alanpeabody
vim ~/.zshrc
# 添加内容, 建议使用 VSCode 连接 WSL
ZSH_THEME="alanpeabody"
plugins=(git zsh-syntax-highlighting zsh-autosuggestions zsh-completions)
# config .zshrc by adding the following line before source "$ZSH/oh-my-zsh.sh":
fpath+=${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions/src

# 配置生效 或者重启即可
source ~/.zshrc
```

**SSH 配置和连接**

```shell
# 1. install openssh-server
sudo nala install openssh-server
sudo apt install openssh-server

# 2. 启动 ssh 服务
sudo service ssh start
# * Starting OpenBSD Secure Shell server sshd

# 3. 查看 ssh 服务状态
service ssh status
# * sshd is running

# 4. Windows Termianal 通过ssh连接 WSL
ip addr
ssh weili@172.26.155.245
# enter 'yes' and enter 'password' of user 'weili'

# 5. 将ssh服务设置为开机自启
sudo systemctl enable ssh
# Synchronizing state of ssh.service with SysV service script with
#  /lib/systemd/systemd-sysv-install.
# Executing: /lib/systemd/systemd-sysv-install enable ssh
```

**Tmux + LunarVim 配置**

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
# https://github.com/neovim/neovim
# https://gitee.com/mirrors/neovim
sudo nala install ./nvim-linux64.deb
nvim --version

# 4. 配置 SpaceVim, 便于终端操作文件
# https://spacevim.org/cn/quick-start-guide/
curl -sLf https://spacevim.org/cn/install.sh | bash

# https://www.lunarvim.org/docs/installation
# https://github.com/lunarvim/lunarvim
# https://gitee.com/mirrors/lunarvim
# 4. 配置 LunarVim, 便于终端操作文件
LV_BRANCH='release-1.2/neovim-0.8' zsh <(curl -s https://raw.githubusercontent.com/lunarvim/lunarvim/fc6873809934917b470bff1b072171879899a36b/utils/installer/install.sh)

git clone --recursive https://gitee.com/mirrors/lunarvim.git
# /home/weili/lunarvim/utils/installer
cd lunarvim/utils/installer/
ll
zsh install.sh

# 卸载 LunarVim
rm -rf ~/.config/lvim ~/.local/share/lunarvim ~/.local/bin/lvim ~/.config/lvim.old
```
