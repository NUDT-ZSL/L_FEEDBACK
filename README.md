# Terrain LOD Stitching

一个从零实现的 Windows 桌面地形 LOD 示例。程序加载由内置确定性 fBm 噪声生成的
512×512 高度场，使用视锥裁剪和屏幕空间误差构建四叉树 LOD，通过顶点位置钳制消除
不同 LOD 边界上的裂缝，并把可见叶节点按 LOD 层级实例化合批。

项目同时提供：

- GLFW 3.4 + OpenGL 4.6 桌面渲染器，渲染到离屏 FBO 后可写 PPM。
- 确定性软件光栅化后端，供无显示器/无 GPU 的 CI 验证。
- 固定相机关键帧回放，每帧输出一行结构化 JSON。
- 线框、LOD 着色、接缝高亮三种调试视图。

## Windows 构建

前置条件：

1. Visual Studio 2022，含“使用 C++ 的桌面开发”工作负载。
2. vcpkg，安装 GLFW 3.4 和 GLM 1.0.1。
3. 支持 OpenGL 4.6 Core Profile 的显卡和驱动。

```powershell
vcpkg install glfw3:x64-windows glm:x64-windows
$env:VCPKG_ROOT = "C:\src\vcpkg"

cmake --preset windows-vcpkg
cmake --build --preset windows-vcpkg
ctest --test-dir build/windows -C Release --output-on-failure
```

生成的 `build\windows\Release\terrain_lod.exe` 是控制台/桌面混合程序：命令行模式输出
JSON 和错误码，交互模式打开 GLFW 窗口。

不安装 GLFW/GLM 时仍可构建无窗口核心与软件后端：

```bash
cmake -S . -B build/local \
  -DTERRAIN_BUILD_GL=OFF \
  -DTERRAIN_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/local -j
```

可选 Linux/EGL 冒烟目标（不依赖窗口系统）：

```bash
cmake -S . -B build/egl -DTERRAIN_BUILD_GL=ON \
  -DTERRAIN_BUILD_EGL_SMOKE=ON
cmake --build build/egl -j
./build/egl/gl_egl_smoke artifacts/egl_gl.ppm
```

## 回放与截图

```powershell
.\build\windows\Release\terrain_lod.exe `
  --camera-path assets\camera_path.txt `
  --time-step 0.1 --frames 161 `
  --debug-view seam `
  --width 960 --height 540 `
  --headless `
  --screenshot artifacts\seam.ppm
```

`--headless` 使用隐藏 GLFW 窗口和离屏 FBO。若显式指定 `--backend software`，则使用
确定性软件光栅化；`auto` 在 GL 初始化失败时自动回退到软件后端。

相机路径是 UTF-8 文本，每行六个有限数值：

```text
# time  positionX  positionY  positionZ  yawDegrees  pitchDegrees
0.0   256.0      190.0     -90.0       -8.0        -18.0
3.0   210.0      130.0      80.0      -25.0        -22.0
```

`#` 后为注释。时间必须非负且严格递增，pitch 必须在 `(-90, 90)` 内。

## 命令行

```text
--seed N               uint32，默认 1337
--amplitude F          (0, 4096]，默认 48
--frequency F          (0, 64]，默认 2.4
--octaves N            [1, 12]，默认 5
--sse F                (0, 1000]，默认 16
--camera-path FILE     固定相机关键帧文件
--time-step F          (0, 60] 秒，默认 0.1
--frames N             [1, 100000]，默认 1
--screenshot FILE      最后一帧写入 PPM P6
--capture-dir DIR      每帧写入 DIR/frame_NNNNNN.ppm
--debug-view V         normal | wireframe | lod | seam
--backend V            auto | gl | software
--width N / --height N [64, 4096]
--cache-capacity N     [16, 65536]
--headless             隐藏 GLFW 窗口
--interactive          打开交互窗口
```

非法数值参数向 stderr 输出 `E_INVALID_PARAM` 并以退出码 2 结束。相机路径缺失、
不可读或格式错误输出 `E_BAD_PATH` 并以退出码 3 结束。渲染后端失败为退出码 4。

交互窗口按键：

- `1` 普通着色，`2` 线框，`3` LOD 层级色，`4` 接缝高亮。
- `W/A/S/D` 平移，`Q/E` 升降，方向键旋转，`P` 截图，`Esc` 退出。

## 验证

`scripts/run_validation.sh` 会自动构建、运行核心测试、双跑 161 帧确定性回放、检查
JSON 字段、生成 960×540 的 PPM/PNG 截图并验证错误码。实测结果保存在
`artifacts/validation/summary.txt`，GL 链接/环境记录在
`artifacts/validation/gl_validation.txt`。

详细算法与 GPU 资源布局见 `docs/TECHNICAL.md`。
