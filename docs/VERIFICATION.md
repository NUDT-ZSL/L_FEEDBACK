# 实际验证记录

验证日期：2026-11-17  
工作区：`/home/liang/feedback/uid176`

## 环境

- WSL2 Linux x86_64。
- GCC 15.2.0。
- CMake 3.30.5。
- 无 Docker、无外部模型/纹理。
- 额外使用独立 LLVM-MinGW Clang 23.1.5 做 Windows x64 交叉编译检查。

## 原生构建与测试

命令：

```bash
cmake -S . -B build-make -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DCHARVIEW_BUILD_TESTS=ON
cmake --build build-make -j 4
(cd build-make && ./charview_tests)
```

结果：

```text
checks=48 failures=0 hash=0x40ab84743e9db71b
```

当前环境未安装 `ctest` 命令，因此直接运行测试可执行文件；CMake 已注册同一个 `charview_deterministic_checks` 测试，Windows/完整 CMake 环境可直接运行 `ctest`。

测试覆盖点：

- 顶点权重归一化、索引范围、网格拓扑和材质数量。
- 镜像 UV 的 `tangent.w=-1`。
- 固定 tick 骨骼矩阵逐位重复。
- 同一 tick CPU 蒙皮重复。
- 两个独立 Renderer 输出逐字节一致。
- 网格可见单元集合、视锥剔除及比例。
- 透明排序、状态合并、14 个阴影图。
- 动画、阴影和线框开关改变输出。
- 相机朝外时三角形全部被剔除。
- 257x199 缩放后恢复 320x240 输出原哈希。
- PPM/BMP 文件头校验、回读像素与帧缓冲完全一致。
- PPM 首像素通道为空白字符时不被误吞。
- 错误 PPM/BMP 文件头被拒绝。

测试输出文件：`artifacts/test_output.txt`。

## 重复运行确定性

同一命令运行两次，分别输出到不同 PPM/BMP：

```bash
./build-make/charview_headless --width 640 --height 400 \
  --cascade-res 512 --cube-res 128 --tick 9 \
  --ppm artifacts/repeat_a.ppm --bmp artifacts/repeat_a.bmp
./build-make/charview_headless --width 640 --height 400 \
  --cascade-res 512 --cube-res 128 --tick 9 \
  --ppm artifacts/repeat_b.ppm --bmp artifacts/repeat_b.bmp
```

`sha256sum` 结果：

```text
ec2905085b24261889d7d49ec54659aef2d235f905157cb41a9092119c6eab11  artifacts/repeat_a.ppm
ec2905085b24261889d7d49ec54659aef2d235f905157cb41a9092119c6eab11  artifacts/repeat_b.ppm
d4e7b87de9a62c0cbb845e33b58e8b87163385611674097cc0556df0b2b1d5df  artifacts/repeat_a.bmp
d4e7b87de9a62c0cbb845e33b58e8b87163385611674097cc0556df0b2b1d5df  artifacts/repeat_b.bmp
```

两次日志中的帧缓冲哈希均为 `0x5ec40f37e2738368`。仅 `timing` 行因 CPU 调度不同而不同。

## 最终画面导出

960x640、tick 0、级联 1024、立方体面 256：

```bash
./build-make/charview_headless --width 960 --height 640 \
  --cascade-res 1024 --cube-res 256 --tick 0 \
  --ppm artifacts/frame.ppm --bmp artifacts/frame.bmp
```

关键日志：

```text
[cull] total=9608 grid_candidates=8986 visible=8754 ratio=0.9111 cells=227/432
[draw] opaque=8102 transparent=652 commands=659+wire:0 transparent_order=ok shadow_maps=14 shadow_triangles=100744
[export] ppm=artifacts/frame.ppm hash=0xbf45cf68d19c3a69
[export] bmp=artifacts/frame.bmp hash=0xbf45cf68d19c3a69
[image] 960x640 framebuffer_hash=0xbf45cf68d19c3a69
```

产物：

- `artifacts/frame.png`：由同一 PPM 无损转换用于查看。
- `artifacts/frame.ppm`
- `artifacts/frame.bmp`

背面/透明披风视图：

```bash
./build-make/charview_headless --width 640 --height 400 \
  --cascade-res 512 --cube-res 128 --tick 9 \
  --yaw 180 --pitch 10 --distance 5.2 \
  --ppm artifacts/back.ppm --bmp artifacts/back.bmp
```

结果是透明披风正常显示，`transparent_order=ok`，PPM/BMP 同为 `0xc47eb86749bfb4f9`。截图：`artifacts/back.png`。

线框模式验证：

```bash
./build-make/charview_headless --width 640 --height 400 \
  --cascade-res 512 --cube-res 128 --tick 0 --wireframe \
  --ppm artifacts/wireframe.ppm --bmp artifacts/wireframe.bmp
```

截图：`artifacts/wireframe.png`；该帧 PPM/BMP 同为 `0x82284c315fa5c318`。

## Sanitizer

命令：

```bash
cmake -S . -B build-asan -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug \
  -DCHARVIEW_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j 4
(cd build-asan && ./charview_tests)
```

结果：48 项全部通过，ASan/UBSan 未报告越界、未定义行为或泄漏。

## Windows 交叉编译

使用 LLVM-MinGW Clang 23.1.5、Windows UCRT 头文件/库，在 WSL 内交叉编译：

```text
build-win/charview.exe:          PE32+ executable for MS Windows 6.00 (console), x86-64
build-win/charview_headless.exe: PE32+ executable for MS Windows 6.00 (console), x86-64
```

导入表包含 `GDI32.dll`、`USER32.dll`、`KERNEL32.dll`，`DirectXMath.h` 编译通过。该步骤验证了 Win32/GDI 源文件的 API 使用和链接路径。

限制：本 WSL 环境没有 Windows 桌面和 Wine，未实际点击窗口、拖拽缩放或观察 GDI 显示。因此最终“窗口显示与导出逐像素一致”的人工 GUI 验收需在 Windows 上按 README 构建运行；核心的同一帧缓冲、导出回读与逐像素一致测试已经在本地自动验证。

## 建议的 Windows 人工验收

1. 以 960x640 启动 `charview.exe`，拖动窗口并最大化/还原，确认画面持续更新且无崩溃。
2. `Space` 暂停，`N` 连续步进，确认姿态按固定 tick 变化。
3. 鼠标拖动到背面观察透明披风，转到正面观察透明面罩。
4. `F` 切到线框，确认网格与地面线框；再按 `F` 恢复。
5. 按 `P`/`B`，检查 `frame.ppm`、`frame.bmp` 文件头以及和窗口当前的画面一致。
6. 观察控制台/`charview.log` 中的 `[cull]`、`[draw]` 日志。

