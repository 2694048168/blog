## Modern CMake for Modern C++

> 软件和工具无时无刻不在更新, 我们应该尽可能使用新版本的工具, 肯定解决了实践中的一些痛点问题. the more modern, the more modern.

### 现代的 CMake 简述
- 始终安装并使用最新的 CMake, 建议 CMake 版本比使用的编译器要更新, 同时应该比使用的所有库(尤其是 Boost)都要更新
- 构建项目
```shell
cmake -S . -B build
cmake --build build
cmake --build build -v -j16
```

- 执行安装
```shell
cmake --build build --target install
cmake --install build
```

- 指定编译器
```shell
# 设置 CMakeLists.txt 的 cache 缓存变量
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# 开启 cland 自动补全等功能的文件导出 
# "build\compile_commands.json"
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

- 指定生成器
```shell
cmake --build build -G Ninja
cmake --build build -G "Visual Studio 17 2022"
cmake --build build -G "Unix Makefiles"
cmake --build build -G "MinGW Makefiles"
```

- 设置选项
```shell
cmake  --help

# Create or update a cmake cache entry.
# CMake 支持缓存选项
cmake -DCMAKE_CXX_COMPILER=clang++

# 使用 -L 列出所有选项
cmake -LH

# 标准选项
cmake -DCMAKE_BUILD_TYPE=Release
cmake -DCMAKE_BUILD_TYPE=Debug

cmake -DCMAKE_INSTALL_PREFIX=/usr/local
cmake -DCMAKE_INSTALL_PREFIX=~/.local

cmake -DBUILD_SHARED_LIBS=ON 或 OFF 来控制共享库的默认值
```

- CMake 行为准则(Do's and Don'ts)

### 基础知识简介
- 最低版本要求, CMake的版本与它的特性(policies)相互关联
```shell
cmake_minimum_required(VERSION 3.8...3.26)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
```

- 顶层 CMakelists 设置项目
```shell
project(ProjectName
    VERSION 1.0
    DESCRIPTION "Modern CMake project"
    LANGUAGES CXX 
)
# supported languages: CXX | C | Fortran | ASM | CUDA | CSharp | SWIFT, 默认是 C CXX
```

- 生成可执行文件(target)
```shell
add_executable(target_name main.cpp main.hpp)
```

- 生成一个库(target)
```shell
# CMake 将会通过 'BUILD_SHARED_LIBS' 的值
# 来选择构建 STATIC 还是 SHARED 类型的库
add_library(lib_name STATIC main.cpp main.hpp)
add_library(lib_name SHARED main.cpp main.hpp)
add_library(lib_name MODULE main.cpp main.hpp)

# 需要生成一个虚构的目标, 一个不需要编译的目标,
# 只有一个头文件的库, 这被叫做 INTERFACE 库
add_library(lib_name INTERFACE one.hpp two.hpp)

# 一个 ALIAS 库, 只是给已有的目标起一个别名,
# 这么做的一个好处是可以制作名称中带有 :: 的库
add_library(lib_name ALIAS main.cpp main.hpp)
```

- 目标信息
```shell
target_include_directories(target_name
    PUBLIC | PRIVATE | INTERFACE
        "include"
)
# 目标可以有包含的目录、链接库（或链接目标）、
# 编译选项、编译定义、编译特性等
target_link_libraries(target_name
    PRIVATE
        "dep_lib_file"
)
```

- 变量和缓存
- CMake 编程
- CMake 与源代码交互(配置文件和读入文件两种方式)
- 如何组织项目结构
```
. Project_Name
|—— .gitignore
|—— LICENCE
|—— CMakeLists.txt
|—— cmake
|   |—— FindSomeLib.cmake
|   |—— something_else.cmake
|—— include
|   |—— module
|   |—— |—— lib.hpp
|—— src
|   |—— CMakeLists.txt
|   |—— lib.cpp
|—— apps
|   |—— CMakeLists.txt
|   |—— app.cpp
|—— tests
|   |—— CMakeLists.txt
|   |—— testlib.cpp
|—— docs
|   |—— CMakeLists.txt
|—— extern
|   |—— googletest
|   |—— git_submodule_way
|—— scripts
|   |—— helper.py
|   |—— build.sh
|   |—— build.bat
|—— README.md
```

- [CMakeExamplesTutorial](https://github.com/2694048168/C-and-C-plus-plus/tree/master/CMakeTutorial)
- [CMakeClangVSCode](https://github.com/2694048168/C-and-C-plus-plus/tree/master/CMakeClangVcpkg)
- [Modern-CMake 教程](https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/)
- [Effective Modern CMake](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1)
