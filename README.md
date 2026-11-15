# PBR Offline Renderer

一个面向游戏/影视前期制作的离线 PBR 参考渲染器。程序使用 OpenGL 4.5 core，
从 RGBA32F 帧缓冲读取线性结果，输出 Radiance HDR 与经曝光/色调映射/sRGB 转换的
512x512 PNG，并提供基于固定基准统计 JSON 的离线验证模式。

## 功能

- Cook-Torrance 微表面 BRDF，支持金属度-粗糙度工作流。
- 各向异性 GGX 法线分布与 Smith 几何项。
- 5 种示例材质：纯金属、粗糙金属、光滑介电质、粗糙介电质、各向异性金属。
- 4 个点光源与 2 个矩形区域光，区域光按功率 CDF 重要性选择并做面积采样。
- 启动时从内置环境描述确定性积分 9 个 L2 球谐系数，用于间接漫反射与镜面环境光。
- 物理相机：焦距、传感器尺寸、f 数光圈、对焦距离、曝光时间。
- 镜头圆盘抖动景深、曝光时间内的子帧时间累积运动模糊、确定性双边滤波降噪。
- 线性 HDR（带曝光/色调映射参数注释）和 Reinhard/ACES/Uncharted2 的 PNG 预览。
- 逐材质平均颜色、对数高光直方图和镜面主方向统计，可与基准 JSON 对比并生成文本报告。
- 严格参数与 JSON 校验，退出码区分参数、场景加载、渲染/IO 错误。

## 目录

```text
assets/scene.json             示例场景与内置环境描述
assets/reference_stats.json   默认基准配置生成的参考统计
src/                          渲染器、场景解析、色彩管线、统计与验证
third_party/stb_image.h       随项目提供
third_party/stb_image_write.h 随项目提供
third_party/nlohmann/json.hpp nlohmann/json 3.11.3 单头文件
third_party/glad/             OpenGL 4.5 core 函数加载器
tests/                        Linux shell 与 Windows PowerShell 验收脚本
docs/TECHNICAL.md             实现与算法说明
```

## Windows 构建

要求 Windows 10/11、Visual Studio 2019/2022 的 C++ 桌面开发工具集、CMake 3.16+
和支持 OpenGL 4.5 的显卡驱动。运行期不需要网络。

### GLFW 的三种本地准备方式

1. vcpkg（推荐）：

```bat
vcpkg install glfw3:x64-windows
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
```

2. GLFW 源码：将 GLFW 3.3+ 源码解压到 `third_party/glfw/`，使
   `third_party/glfw/CMakeLists.txt` 存在。配置时不访问网络。

3. 已编译的 GLFW：显式传入头文件和库路径：

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DGLFW_INCLUDE_DIR=C:/deps/glfw/include ^
  -DGLFW_LIBRARY=C:/deps/glfw/lib/glfw3.lib
```

`stb_image.h`、`stb_image_write.h`、`nlohmann/json.hpp` 已放在 `third_party/`，
离线可直接编译。若替换这些单头文件，保持文件名和目录不变即可。

### 编译

```bat
cmake --build build --config Release
cd build\Release
.\pbr_renderer.exe --scene assets\scene.json --output output\reference
```

构建后 CMake 会把 `assets/` 复制到可执行文件目录，因此从 `build\Release` 运行也能
找到默认场景。也可以在项目根目录运行 `.\build\Release\pbr_renderer.exe ...`。

## 命令行

```text
--scene <file>           场景 JSON，默认 assets/scene.json
--output <prefix>        输出前缀，默认 output/render
--mode <render|validate> 默认 render
--samples <1..4096>         每个像素的直接光采样数，默认 256
--exposure <EV>            预览曝光，范围 [-10, 10]，默认 0
--tonemap <curve>          reinhard | aces | uncharted2，默认 aces
--subframes <1..128>       时间累积子帧数，默认 8
--reference <file>         参考统计 JSON，默认 assets/reference_stats.json
--report <file>            验证报告，默认 output/validation_report.txt
--write-reference <file>   将本次渲染统计写成新的参考 JSON
--help                     显示帮助
```

`--output` 默认是文件前缀。如果传入已存在的目录或以 `/`、`\` 结尾的目录路径，
程序会使用其中的 `render` 作为前缀。例如：

```bat
.\pbr_renderer.exe --output output\shot --exposure 1 --tonemap aces
```

会生成：

```text
output\shot_ev+1_aces.hdr
output\shot_ev+1_aces.png
```

HDR 始终保存未曝光的场景线性辐射值；`--exposure` 和 `--tonemap` 只影响 PNG。
HDR 文件头会记录 `exposure_ev`、`tonemap`、曝光倍率、采样数、子帧数和场景路径。

## 场景与材质

示例 `assets/scene.json` 满足以下约束：

- 相机字段 `position/target/up/focal_length_mm/sensor_size_mm/f_number/focus_distance/exposure_time`。
- `materials` 至少包含所需材质；每个材质要求 `name/base_color/metallic/roughness/anisotropy`。
  `anisotropy_rotation_degrees` 和 `ior` 可选。
- `spheres` 的 `material_id` 必须在材质数组范围内；也可用 `material` 名称引用。
  名称和 ID 同时出现时必须一致。
- 点光类型 `point` 需要 `position/color/power`。
- 矩形光类型 `rect` 需要 `center/normal/tangent/half_size/radiance`，normal 与 tangent 必须正交。
- 仅支持程序内置的解析采样环境描述，不读取环境贴图或外部模型。

缺失字段、类型错误、范围错误和材质 ID 越界都会输出具体 JSON 路径并退出 2。

## 渲染与验证

基准渲染（默认 256 采样、8 子帧、512x512）：

```bat
.\pbr_renderer.exe --scene assets\scene.json --output output\reference ^
  --samples 256 --subframes 8
```

重新生成参考统计（固定基准配置）：

```bat
.\pbr_renderer.exe --scene assets\scene.json --output output\baseline ^
  --samples 256 --subframes 8 --write-reference assets\reference_stats.json
```

验证模式会先完成同一渲染，再计算统计并对比基准：

```bat
.\pbr_renderer.exe --scene assets\scene.json --output output\validated ^
  --mode validate --samples 256 --subframes 8 ^
  --reference assets\reference_stats.json --report output\validation_report.txt
```

验证指标：

- 每材质平均线性 RGB 的欧氏距离。
- 每材质亮度 `log2` 16 桶高光直方图的 L1 距离。
- 每材质按亮度平方加权的主镜面反射方向夹角。

阈值存放在参考 JSON 中。超限只让报告显示 `FAIL`，进程仍以 0 退出；指标、材质名和
当前/参考值会逐条写入报告。

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 渲染/验证流程完成（验证超限也返回 0） |
| 1 | 命令行参数非法 |
| 2 | 场景文件无法读取、JSON 错误、缺字段、类型错误、材质 ID 越界 |
| 3 | OpenGL 初始化、渲染、参考文件读取、报告或 HDR/PNG IO 失败 |

## 测试

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File tests\run_tests.ps1
```

Linux 无头环境（使用随项目提供的 EGL 头文件与系统 Mesa 运行库）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./tests/run_tests.sh ./build/pbr_renderer
```

测试覆盖 1/2/3 退出码、JSON 缺字段/类型/ID 越界、HDR 逐字节复跑一致、3 档曝光
× 3 种曲线、512x512 输出、超阈值报告和基准验证通过。

## 随附验证产物

- `artifacts/final_reference_ev0_aces.hdr`：默认 256 采样/8 子帧的线性 HDR。
- `artifacts/final_reference_ev0_aces.png`：默认 ACES 0 EV 预览。
- `artifacts/validation_report.txt`：基准验证报告。
- `artifacts/test_results.txt`：本机完整测试输出。
- `artifacts/verification.txt`：环境、命令、哈希、HDR 回读和错误码记录。

## 本机实际验证记录

本次交付环境没有 Windows，实际执行在 Linux + Mesa llvmpipe 的无头 EGL 上下文：

- `OpenGL: 4.5 (Core Profile) Mesa 26.0.8 / llvmpipe`
- CMake 3.30.5 + GCC 15.2 构建成功。
- 默认 256 采样、8 子帧渲染耗时约 2.3 秒。
- 同一输入连续两次输出的 HDR SHA-256 完全相同，PNG 也逐字节一致。
- `stb_image` 成功回读默认 HDR：`512x512`，最小 0.000002，最大 16.5625。
- 默认基准验证报告：`SUMMARY: PASS (15/15 metrics within threshold)`。
- `tests/run_tests.sh`：`ALL TESTS PASSED (27 checks)`。

Windows 分支使用 GLFW 隐藏窗口和相同的 OpenGL 4.5 着色器/渲染路径；PowerShell
测试脚本已随项目提供，但未在此 Linux 容器内实机运行 Windows。

详细算法和确定性说明见 `docs/TECHNICAL.md`。
