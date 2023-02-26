# 利用 vcpkg from Microsoft and find_package of CMake 进行 C++ 包管理

- &ensp;<span style="color:MediumPurple">Title</span>: 利用 vcpkg from Microsoft and find_package of CMake 进行 C++ 包管理
- &ensp;<span style="color:Moccasin">Tags</span>: vcpkg; C++ Library Manager; CMake; find_package;
- &ensp;<span style="color:PaleVioletRed">Type</span>: Technology Blog
- &ensp;<span style="color:DarkSeaGreen">Author</span>: [Wei Li](https://2694048168.github.io/blog/#/) (weili_yzzcq@163.com)
- &ensp;<span style="color:DarkMagenta">Revision of DateTime</span>: 2023-02-26;

> vcpkg that is C++ library manager from Microsoft and find_package command from CMake.

## Overview of vcpkg and find_package
1. **Install vcpkg**
2. **Use of vcpkg**
3. **find_package in CMakeLists.txt**

### **Install vcpkg and use**
[vcpkg from Microsoft on Github](https://github.com/microsoft/vcpkg)
[vcpkg from Microsoft](https://vcpkg.io/en/getting-started.html)

**Quick Start on Windows**
```shell
# 从 github 克隆 vcpkg
git clone https://github.com/microsoft/vcpkg

# 运行脚本按照 vcpkg 可执行文件
.\vcpkg\bootstrap-vcpkg.bat

# 检索所需要安装的 C++ library
cd vcpkg
.\vcpkg search eigen

# 根据检索结果安装所需 C++ library, 默认是安装 x86 版本
./vcpkg install eigen3
./vcpkg install eigen3:x64-windows
./vcpkg install eigen3 --triplet=x64-windows

# 根据安装后的提示 eigen3 provides CMake targets:
# 在 CMake 中使用该 C++ library
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

# 卸载或者删除 C++ libray
./vcpkg remove eigen3
./vcpkg remove eigen3:x64-windows
./vcpkg remove eigen3 --triplet=x64-windows

# ==== 配置 vcpkg 系统环境变量 ====
# Path 新增系统环境变量
Path=D:\Development\vcpkg

# 解决 vcpkg 安装库时候的默认为 x64 版本
VCPKG_DEFAULT_TRIPLET=x64-windows

# 解决 vcpkg 在 VSCode 中全局使用
CMAKE_TOOLCHAIN_FILE=D:\Development\vcpkg\scripts\buildsystems\vcpkg.cmake
```

**在 CMake 中使用 vcpkg**

- 将 vcpkg 作为一个子模块

当希望将 vcpkg 作为一个子模块加入到的工程中时, 可以在第一个 project() 调用之前将以下内容添加到 CMakeLists.txt 中,而无需将 CMAKE_TOOLCHAIN_FILE 传递给 CMake 调用.
```shell
# 使用此方式无需设置 CMAKE_TOOLCHAIN_FILE 即可使用 vcpkg
set(CMAKE_TOOLCHAIN_FILE 
    "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file"
)
```

- Visual Studio Code 中的 CMake Tools

将以下内容添加到您的工作区的 settings.json 中将使 CMake Tools 自动使用 vcpkg 中的第三方库:
```shell
# vcpkg root=D:/Development/vcpkg
{
  "cmake.configureSettings": {
    "CMAKE_TOOLCHAIN_FILE": "[vcpkg root]/scripts/buildsystems/vcpkg.cmake"
  }
}
```

- Visual Studio CMake 工程中使用 vcpkg

打开 CMake 设置选项，将 vcpkg toolchain 文件路径在 CMake toolchain file 中：
```shell
# vcpkg root=D:/Development/vcpkg
[vcpkg root]/scripts/buildsystems/vcpkg.cmake
```

### **find_package in CMakeLists.txt**

**find_package 的简单用法**
```CMake
# 搜索一个名为 PackageName 的包
find_package(PackageName)

# 对于项目来说,有些包是可选的,但有些是必要的,
# 对于必要的,可以指定 REQUIRED, 这样找不到对应的包就会报错
find_package(PackageName REQUIRED)

# 有些包其中有许多的组件, 可以指定 COMPONENTS 来只使用某些组件
find_package(PackageName COMPONENTS REQUIRED search_modes)
```

**find_package 搜索的是什么**
- find_package 本质是检索的是 '.cmake' 文件(search_modes)
- findPackageName.cmake 文件; 第一种模式称之为 MODULE 模式
- PackageNameConfig.cmake 文件; 第二种模式称之为 CONFIG 模式
- 可以指定 MODULE 或 CONFIG 关键字来指定只找某种模式的; 否则是两种模式都找，先找第一中模式文件,找不到再找第二中模式文件
- '.cmake' 文件里其实设定的就是库和头文件路径的位置(定义变量)

```CMake
# '.cmake' file 定义的变量, 使用库前可以查看该文件确认变量
PackageName_FOUND # 找到了就是 True, 没找到就是未设定
PackageName_INCLUDE_DIR #即头文件路径
PackageName_LIBRARY # 即库文件

# 检索的路径以及方式
set(Eigen3_ROOT path/to/Eigen3)
set(Eigen3_DIR path/to/Eigen3)
list(APPEND CMAKE_MODULE_PATH /path/to/Eigen3Config.cmake)
find_package(PackageName [PATHS path1 [path2 ... ]])
find_package(PackageName [HINTS path1 [path2 ... ]])
```
