# SkelPreview

一个用于离线检查角色绑定结果的桌面向骨骼动画预览工具。工程从零实现自定义
`SKA1` 二进制资产读取、采样、TRS 层级求值、逆绑定与线性混合蒙皮，并提供
OpenGL 4.1 实时预览和逐字节可复现的 CPU 离线渲染。

## 功能

- 解析节点树、逐节点动画轨道、逆绑定矩阵、最多四骨骼权重和网格拓扑。
- 父链环、越界索引、非单调采样时间、零四元数、非法权重等均有稳定错误码，
  失败时退出码非零且不暴露部分加载状态。
- 六种资产声明的 T/R/S 合成顺序；四元数始终归一化。
- 法线通过每个蒙皮矩阵的逆转置变换，支持非均匀缩放。
- 权重和不为 1 时统一按和归一化并计数，不删除权重或改变拓扑。
- 响应鼠标旋转/滚轮缩放，`W` 切换线框；窗口创建失败自动回退离线渲染。
- 离线输出 `depth_XXXX.pfm`、`color_XXXX.ppm` 和 JSON 统计报告。
- 内置确定性正常/异常资产生成器，以及单元、数值、端到端确定性测试。

## 依赖与构建

正式依赖由仓库根目录的 `vcpkg.json` 声明：`glm`、`stb`、`glfw3` 和
`nlohmann-json`。CMake 要求 3.25 或更新版本。

### Windows / Visual Studio 2022

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\tools\vcpkg
C:\tools\vcpkg\bootstrap-vcpkg.bat
$env:VCPKG_ROOT = "C:\tools\vcpkg"
C:\tools\vcpkg\vcpkg install --triplet x64-windows

cmake --preset vs2022
cmake --build --preset vs2022
ctest --preset vs2022
```

也可显式传入
`-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake`。

### Linux（当前验证环境）

没有 sudo 权限时，将 Ubuntu 包下载并解压到项目本地前缀：

```bash
./scripts/bootstrap_deps_linux.sh
cmake -S . -B build/linux-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/.deps/usr/usr"
cmake --build build/linux-release -j4
ctest --test-dir build/linux-release --output-on-failure
```

也可使用 vcpkg 清单构建。构建后运行阶段不需要网络；`SKA1` 样本全部由
`asset_generator` 确定性生成。

## 使用

生成正常资产和异常资产：

```bash
build/linux-release/asset_generator --list
build/linux-release/asset_generator --preset normal --out demo/normal.ska
build/linux-release/asset_generator --preset cycle --out demo/cycle.ska
```

逐帧离线渲染：

```bash
build/linux-release/skelpreview offline demo/normal.ska \
  --frame 3 --width 640 --height 480 \
  --eye 4.5,3.4,6.5 --target 0,1.3,0 --fov 45 \
  --output-dir demo/frames \
  --report demo/frames/report.json
```

`--start-frame N --end-frame M` 可渲染闭区间。帧号至少补零到四位。

实时预览：

```bash
build/linux-release/skelpreview preview demo/normal.ska
```

左键拖动旋转，滚轮缩放，`W` 线框，空格暂停，`R` 复位，`Esc` 退出。
`--force-offline` 可用于显式验证回退路径。

测试模式：

```bash
build/linux-release/skelpreview selftest build/selftest
build/linux-release/skelpreview_tests build/unit-tests
```

## 实际验证结果

在当前 Ubuntu 26.04、GCC 15.2、CMake 3.30.5 环境下：

- `cmake --build build/test1 -j4` 成功构建全部目标。
- `skelpreview_tests`：`85/85 checks passed, 0 failed`（含 stb_image 回读 PPM）。
- `ctest`：`3/3` 测试通过（`acceptance`、`unit_and_numeric`、
  `cli_selftest`）。
- 端到端接受测试通过：正常、循环引用、越界网格索引、越界骨骼、非单调时间、
  零四元数、超限权重数、零权重和、非法合成顺序、截断和退化骨骼。
- 异常退出码实测：`18/27/26/20/22/23/25/28/14`；退化骨骼产生 16 个法线
  NaN，返回可分失败码 `70`。
- 同一帧两次离线运行的 320x240 `color_0003.ppm` 与
  `depth_0003.pfm` SHA-256 分别一致：
  `047074e84fd94c9b8e8125a15aa8b72455490d003698ef10c6470476d1ba2a2`、
  `69a7c7ec28875b6cf5695b7176b7cdbf58cca8e2c0230a263c50e6ab8678a69e`。
- 两次运行 JSON 去除 `timing` 与逐帧 `elapsed_ms` 后完全一致。
- 强制离线与无显示环境下自动回退均成功生成 PFM、PPM 和报告。

更详细的格式、错误码与设计见 [docs/FORMAT.md](docs/FORMAT.md) 和
[docs/TECHNICAL.md](docs/TECHNICAL.md)。
