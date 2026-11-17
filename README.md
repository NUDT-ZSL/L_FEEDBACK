# CharView: 确定性 Windows 软渲染角色预览器

CharView 是一个从零实现的 Windows 桌面角色资产预览程序。场景、骨骼、网格、切线空间法线图、材质和光源全部由程序确定性生成，不依赖外部模型或纹理。渲染完全由 CPU 软光栅器完成，Win32 GDI 只负责把最终 sRGB 帧缓冲显示到窗口。

![CharView 渲染结果](artifacts/frame.png)

## 已实现功能

- 指令缓冲式渲染流程：清屏、阴影深度绘制、批量绘制、逐三角形透明绘制、线框绘制。
- 顶点变换、D3D 风格裁剪空间、近/远/四侧裁剪、背面剔除、像素中心采样、深度缓冲、透视校正插值。
- 15 骨骼角色骨架，每顶点最多 4 个骨骼影响，位置/法线/切线线性混合，固定 30 Hz 时间步。
- 256x256 程序化切线空间法线图，双线性采样、经纬重复包裹、镜像 UV 的 `tangent.w` 手性处理。
- 1 个方向光 + 2 个点光源；方向光 2 级级联阴影，两个点光源各 6 面立方体阴影；3x3 PCF。
- 1.25 单位均匀网格空间划分；每帧更新可见单元集合及可见面索引，统计并输出视锥剔除比例。
- 不透明材质按渲染状态/材质分组；透明三角形按相机空间深度从后向前排序，深度相等时按材质 ID、面 ID 稳定打破平局。
- Win32/GDI 窗口、尺寸变化重建、暂停/继续、单帧步进、线框/实体切换、鼠标轨道与滚轮缩放。
- 当前帧导出 PPM(P6) 与 24 位 BMP；路径和快捷键可通过命令行配置；写盘后重新校验文件头。

## Windows 构建

环境：

- Visual Studio 2022 或更新版本，安装 “使用 C++ 的桌面开发” 工作负载。
- Windows SDK（提供 `DirectXMath.h`）。
- CMake 3.20 或更新版本。

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

也可以使用 `CMakePresets.json`：

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-release
ctest --preset windows-release
```

生成的可执行文件：

- `build\Release\charview.exe`：Win32 GDI 交互程序。
- `build\Release\charview_headless.exe`：无窗口渲染/导出，适合回归验证。
- `build\Release\charview_tests.exe`：确定性、管线、导出和缩放测试。

## 运行与交互

```powershell
.\build\Release\charview.exe --width 1280 --height 720 `
  --ppm artifacts\frame.ppm --bmp artifacts\frame.bmp
```

默认快捷键：

- `Space`：暂停/继续。
- `N`：暂停时前进一个固定 1/30 秒动画 tick。
- `F`：线框/实体模式切换。
- `P`：导出 PPM(P6)。
- `B`：导出 24 位 BMP。
- 鼠标左键拖动：环绕相机；滚轮：缩放。
- `Esc`：退出。

导出快捷键可用 `--ppm-key`、`--bmp-key` 改为其他 ASCII 字母。导出始终使用当前已经显示/即将显示的帧缓冲，不额外重新着色。

## 命令行

```text
--width N              帧缓冲宽度，64..4096，默认 960
--height N             帧缓冲高度，64..4096，默认 640
--tick N               初始动画 tick（固定 1/30 秒）
--cascade-res N        方向光级联阴影分辨率，默认 1024
--cube-res N           点光源立方体单面分辨率，默认 256
--yaw DEG              相机环绕角
--pitch DEG            相机俯仰角
--distance N           相机轨道距离
--no-shadows           禁用阴影生成和采样
--ppm PATH             PPM 导出路径
--bmp PATH             BMP 导出路径
--ppm-key C            PPM 导出快捷键
--bmp-key C            BMP 导出快捷键
--paused               启动后暂停
--wireframe            启动线框模式
--help                 帮助
```

无窗口验证示例：

```powershell
.\build\Release\charview_headless.exe --width 960 --height 640 --tick 9 `
  --cascade-res 1024 --cube-res 256 `
  --ppm artifacts\frame.ppm --bmp artifacts\frame.bmp
```

## 确定性与渲染规则摘要

- 不启用 `/fp:fast`；MSVC 使用 `/fp:precise`。
- 单线程执行，不使用原子累加、随机数、系统时间参与姿态或着色。
- 动画只由整数 tick 决定，姿态使用固定 `1/30` 秒时间步。
- 所有排序都有全序键，不依赖 `std::sort` 对相等元素的偶然顺序。
- 像素中心固定为 `(x + 0.5, y + 0.5)`；深度测试固定为 `LessEqual`；覆盖边使用包含规则。
- 最终颜色在浮点线性空间完成光照和透明混合，随后做 Reinhard 色调映射和 sRGB 传输函数编码为 8 位。

完整规则见 [docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 验证

本地实际执行结果、哈希、截图和交叉编译结果见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。核心测试覆盖 48 项检查，包括：

- 两个独立 Renderer 输出逐字节相同。
- 同一 tick 重新渲染、缩放后恢复同尺寸均得到相同哈希。
- 朝向视锥外的相机将三角形全部剔除。
- 透明排序键有效，2 级 CSM + 12 个点光源立方体面共 14 个阴影图。
- 阴影开关改变最终像素。
- PPM/BMP 导出后重新读取，RGB 数据与帧缓冲完全一致，坏文件头被拒绝。

## 代码结构

```text
src/cv_math.h/.cpp       确定性向量/矩阵、相机、视锥
src/assets.h/.cpp        程序化网格、骨骼、材质、法线图、光源
src/visibility.h/.cpp    均匀网格、可见单元集合、CPU 视锥剔除
src/raster.h/.cpp        裁剪、光栅化、深度测试、透明混合、线框
src/renderer.h/.cpp      指令缓冲、蒙皮、CSM/点阴影、PCF、光照、排序、解析
src/image_io.h/.cpp      sRGB 帧缓冲、PPM/BMP 导出与文件头校验
src/win32_main.cpp       Win32/GDI、输入、固定时间步、导出
src/headless_main.cpp    无窗口渲染与统计日志
tests/test_main.cpp      自包含回归测试
```

## 当前环境说明

本工作区运行在 WSL/Linux，因此实际验证包含 Linux 原生构建、回归测试、ASan/UBSan，以及 Windows x64 PE 交叉编译。Win32 窗口逻辑已成功编译为 `PE32+` 可执行文件，但本环境没有可交互的 Windows 桌面，因此最终窗口交互验收仍需在 Windows 上执行。具体要求见 `docs/VERIFICATION.md`。

