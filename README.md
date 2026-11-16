# 确定性离屏级联阴影渲染器

这是一个 Windows / Direct3D 11 的纯离屏渲染程序。程序不创建 `HWND` 或可见窗口，
在 1280x720 颜色缓冲和深度缓冲上渲染确定性程序化场景，并输出 PNG 序列帧与 JSON
统计文件。

场景包含：

- 256x256 顶点网格地形，高度来自固定种子的二维值噪声 fBm。
- 两个由多段胶囊体组成的角色，每段胶囊体在 GPU 顶点着色器中按正弦时间函数偏移。
- 一个方向光。
- 三级级联阴影贴图（CSM），每级 1024x1024。
- 主通道 3x3 PCF、级联边界淡入淡出、方向光漫反射与阴影衰减。

## 构建

要求 Windows 10/11、Visual Studio 2019 或更高版本的 C++ 桌面工作负载，
以及 Windows 10 SDK 10.0.19041.0 或更高版本。

最简单的方式是在仓库根目录运行：

```bat
scripts\build_windows.bat
```

输出为：

```text
build\Release\deterministic_csm.exe
build\Release\shaders\scene.hlsl
```

也可以从 VS 开发者命令提示符手动构建：

```bat
cmake -S . -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release --parallel
```

CMake 的 post-build 步骤会把 `shaders/scene.hlsl` 复制到可执行文件旁。
程序运行时从该位置加载 HLSL，并通过 `D3DCompile` 编译 Shader Model 5.0。

## 运行

```bat
build\Release\deterministic_csm.exe --frames 60 --output-dir output --simulate-device-loss-at 30
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--frames N` | `60` | 帧数，必须大于 0 |
| `--output-dir DIR` | `output` | PNG/JSON 输出目录 |
| `--cascade-splits A,B,C` | `0.1,0.3,0.6` | 三个严格递增且位于 `(0,1)` 的视深度比例 |
| `--slope-offset X` | `2.0` | 阴影深度斜率缩放偏移 |
| `--const-offset X` | `0.0015` | 阴影归一化裁剪深度常量偏移 |
| `--simulate-device-loss-at N` | 无 | 在第 N 帧完整输出后执行一次设备销毁与重建 |

分割距离按摄像机近/远平面线性计算：

```text
split[i] = near + ratio[i] * (far - near)
```

默认摄像机的 `near=0.5`、`far=140`，因此距离为：

```text
0.5, 14.45, 42.35, 140.0
```

第一级覆盖 `[0.5,14.45]`，第二级覆盖 `[14.45,42.35]`，第三级覆盖
`[42.35,140.0]`。每级阴影贴图都有独立的深度切片和紧包围该视锥段的灯光正交矩阵。

动画时间固定为：

```text
animationTime = frameIndex / 60.0
```

不使用墙钟时间，也不依赖窗口状态或帧率。

## 输出

对于 60 帧，输出目录包含：

```text
frame_0000.png
frame_0000.json
...
frame_0059.png
frame_0059.json
stats.json
```

每帧 JSON 记录：

- 帧号与动画时间戳。
- 绘制调用数（3 次阴影级联绘制 + 1 次主通道绘制）。
- 三级阴影贴图尺寸。
- 级联比例、分割距离、每级近远范围。
- 当帧斜率偏移和常量偏移。
- 当前累计设备恢复次数。

`stats.json` 汇总相同信息，并包含几何顶点/索引数量、固定步长、设备丢失注入帧号与
最终恢复次数。

PNG 由 Windows Imaging Component 的 PNG 编码器写出。没有时间戳、随机数或
未初始化内存进入图像或统计文件。

## 实现结构

```text
src/Options.*     命令行解析与范围校验
src/Scene.*       固定种子地形、胶囊角色、相机与灯光
src/Renderer.*    DX11 设备、离屏目标、CSM、PCF、读回与设备恢复
src/WicPng.*      WIC PNG 编码
src/Stats.*       确定性 JSON 序列化
shaders/scene.hlsl 共享顶点动画、阴影深度与主通道着色器
tests/            输出目录与双跑一致性校验
scripts/          Windows 构建与验收脚本
```

全部地形和角色顶点/索引在初始化时由 CPU 生成一次，并一次性上传到默认用途的
GPU 顶点/索引缓冲区。设备重建时使用同一份 CPU 场景数据重新上传。角色动画不是
CPU 每帧修改顶点，而是在同一个 `AnimateVertex()` HLSL 函数中由 `MainVS` 和
`ShadowVS` 共同调用，因此阴影通道与主通道使用完全相同的时间与顶点变换。

阴影通道通过 `SV_Depth` 应用：

```text
bias = slopeOffset * max(|ddx(depth)|, |ddy(depth)|) + constOffset
```

两个参数位于场景常量缓冲区，并随每帧统计写出。

主通道根据视空间深度计算三个级联权重。级联边界附近用 `smoothstep` 淡出旧级联、
淡入新级联，并对两个有效级联分别做 3x3 PCF 后按权重合成。

## 设备丢失恢复

渲染循环在阴影绘制、主通道绘制、颜色复制和 staging 读回后检查
`ID3D11Device::GetDeviceRemovedReason()`。遇到
`DXGI_ERROR_DEVICE_REMOVED`、`DXGI_ERROR_DEVICE_RESET` 等结果时：

1. 解绑并释放 SRV、DSV、RTV、阴影数组、深度缓冲、离屏颜色目标（交换链替身）、
   顶点/索引缓冲、状态对象和着色器。
2. 释放设备上下文与设备。
3. 重新创建设备、离屏颜色目标、深度缓冲、阴影数组、场景缓冲与管线对象。
4. 使用相同帧号和动画时间重试当前帧；只有成功完成才写 PNG/JSON。

`--simulate-device-loss-at N` 在第 N 帧已经完整写出后真实销毁旧 `ID3D11Device`
及其全部设备对象，再走与错误检测完全相同的重建函数，然后从第 N+1 帧继续。
它不会伪造帧号、跳过帧或重复写已经完成的帧。该注入方式不依赖管理员权限或
全局 GPU TDR，不会影响机器上的其他进程。

## 确定性

- 地形种子固定为 `0x51A7C3E9`，没有外部模型、贴图或数据。
- 帧时间、相机、灯光和动画频率固定。
- 所有场景几何只生成一次；渲染使用同一顶点缓冲和同一着色器变换。
- 不使用系统时间、随机设备或可见窗口。
- JSON 使用固定键顺序、经典 locale 和固定小数位数。
- WIC 写入固定 32bpp BGRA 像素和固定分辨率，不附加动态元数据。

同一机器、同一驱动和同一参数重复运行会得到逐字节一致的 PNG、每帧 JSON 和
`stats.json`。实际双跑结果见 [VERIFICATION.md](VERIFICATION.md)。

## 验收

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_acceptance.ps1 `
  -Exe build\Release\deterministic_csm.exe
```

该脚本执行两次 60 帧渲染、第 30 帧后设备重建，检查 60 张 PNG 全部不同、
两次输出逐字节一致、帧序和时间戳连续、统计字段正确，并验证多组非法参数
返回非零退出码。

如果本机有 Python 3.10+，也可以独立校验两个输出目录：

```bat
python tests\verify_outputs.py run_a run_b --frames 60 --loss-frame 30
```

程序对无效参数向 stderr 输出错误并返回 2；运行期一般错误返回 1；不可恢复的
设备丢失返回 3；正常完成返回 0。
