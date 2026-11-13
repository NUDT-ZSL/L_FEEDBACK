# lf_baker

`lf_baker` 是一个确定性的离线材质烘焙器。它读取 JSON 场景，用自实现软光栅器把球面参考几何渲染为逐方向 RGBA8888 帧，并通过内容寻址的分层缓存支持中断恢复和增量重烤。

项目仅依赖 C++17 标准库和 `third_party/stb_image_write.h`。没有网络、GPU、Docker 或运行时第三方库依赖。

## Windows 构建

前置条件：

- Visual Studio 2022 或 Visual Studio Build Tools 2022，包含“使用 C++ 的桌面开发”工作负载。
- CMake 3.20 或更高版本，可在 `PATH` 中调用。
- 可选：Python 3，用于端到端测试；缺少 Python 时仍会构建并运行 C++ 单元测试。

在仓库根目录运行：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

生成的程序为 `build\Release\lf_baker.exe`。

Linux 下同样支持：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 命令行

```text
lf_baker --scene <scene.json> --output <目录> --cache <目录>
```

也支持 `--scene=<path>` 形式。`--help` 打印用法。

运行期间逐帧输出：

```text
FRAME <1-based-sequence>/<total>
```

结束输出：

```text
SUMMARY <本次现场渲染帧数> <本次缓存复用帧数> <总帧数>
```

示例：

```powershell
.\build\Release\lf_baker.exe `
  --scene samples\scene.json `
  --output out `
  --cache cache
```

## 场景格式

主 schema：

```json
{
  "resolution": { "width": 96, "height": 96 },
  "samples": [
    { "direction": [0.0, 0.0, 1.0] },
    { "direction": [0.7071068, 0.0, 0.7071068] }
  ],
  "layers": [
    {
      "baseColor": [0.78, 0.20, 0.12],
      "roughness": 0.35,
      "normalStrength": 0.18,
      "emissionColor": [0.02, 0.005, 0.0]
    }
  ],
  "lights": [
    {
      "direction": [0.6, 0.8, 0.35],
      "color": [1.0, 0.95, 0.86],
      "intensity": 1.0
    }
  ]
}
```

- `samples[i].direction` 是从球心指向相机的方向；数组顺序对应输出帧顺序。若对象显式给出 `index`，所有索引必须组成唯一的 `0..N-1` 序列。也可直接写成 `[x, y, z]`。
- `baseColor`、`emissionColor`、光照 `color` 可写归一化 `[0, 1]` RGB，或 0-255 美术色；任一分量大于 1 时整组颜色按 255 归一化。颜色可写为 3 或 4 元数组，第四个分量忽略。
- `roughness` 和 `normalStrength` 均必须在 `[0, 1]`。
- `lights[i].direction` 是从表面指向光源的单位方向语义；内部会归一化。`color`/`intensity` 可省略，默认白色/`1.0`。
- 最多 32 个材质层、4096 帧，分辨率单边最多 8192。
- 支持常用 snake_case/camelCase 别名，例如 `base_color`、`normal_strength`、`emissionColor`、`sampleDirections`。

JSON 结构/类型/缺失字段错误返回 `E_SCENE_PARSE`；数值范围、层数、帧数和方向零向量错误返回 `E_SCENE_VALID`。

## 渲染语义

- 相机位于采样方向 `dir * 3.2`，看向原点；45° 垂直视场角，近/远平面为 `0.1/100`。
- 球面网格为 48 纬度段 × 96 经度段。顶点变换、透视除法、齐次视锥裁剪、覆盖测试和深度缓存均使用双精度。
- 光栅化只在整数像素中心 `(x + 0.5, y + 0.5)` 采样；深度测试使用严格小于，三角形按固定索引顺序提交。
- 片元属性做透视校正插值。每层按数组顺序在切线空间施加程序化法线扰动，再依次叠加漫反射、粗糙度控制的高光、发射色。所有累加顺序固定。
- 最终 RGB 使用固定 Reinhard 映射 `x / (1 + x)`，再量化为 8-bit；球面覆盖像素 alpha 为 255，背景为 `(0, 0, 0, 0)`。

球面外无抗锯齿，便于不同运行顺序、缓存命中与恢复路径产生完全相同的像素。

把 `samples` 设为 `±X/±Y/±Z` 六个方向即可输出立方体贴图六面所需的序列帧；程序不额外拼接 atlas。

## 缓存

缓存布局：

```text
<cache>/<global-key>/<frame-subkey>/frame_%06d.rgba
<cache>/<global-key>/<frame-subkey>/frame_%06d.meta.json
```

- `global-key`：SHA-256，覆盖分辨率、帧索引列表及固定管线常量。
- `frame-subkey`：SHA-256，覆盖材质层、光照列表和该帧采样方向。
- 方向微调只改变对应帧的子键；材质或光照变化会改变所有帧的子键；分辨率或帧列表变化产生新的全局键。
- 元数据记录 `frame_index`、`subkey`、尺寸、字节数和 RGBA 像素的 SHA-256。缺失元数据、临时文件、长度不符或校验值不符均按缓存缺失处理。
- 每帧先写同目录临时文件，再执行原子重命名；元数据在像素文件之后提交，作为“已完成帧”的标记。
- 启动时清理遗留 `.tmp-*` 半成品；已提交且键完全匹配的帧直接复用。

## 输出

输出目录包含：

```text
frame_000000.png
frame_000001.png
...
manifest.json
```

PNG 固定为 8-bit RGBA、固定压缩参数、无时间戳/文本块。清单按帧索引排序：

```json
{
  "frames": [
    {
      "index": 0,
      "path": "frame_000000.png",
      "average_luminance": 43.36
    }
  ]
}
```

`average_luminance` 是整个 RGBA 图像 RGB 分量的算术均值（0-255，固定两位小数），包含透明背景的 RGB 零值。缓存命中路径从已验证缓存像素重编码 PNG，因此与现场渲染字节一致。

## 中断与退出码

- `Ctrl+C`/`SIGINT`：设置停止标志，完成当前帧的缓存提交和 PNG 写出后以退出码 0 结束；未完成时不发布清单。
- 强制终止（例如 `SIGKILL`，无法捕获）：不保证退出输出；再次运行会复用已提交帧，只重渲缺失帧。
- 两种恢复路径在全部帧完成后生成同一份 PNG 序列和清单。

| 退出码 | 符号 | 含义 |
|---:|---|---|
| 1 | `E_RUNTIME` | 未分类运行时错误 |
| 2 | `E_SCENE_PARSE` | 文件、JSON、缺失字段或类型错误 |
| 3 | `E_SCENE_VALID` | 字段值、层数、帧数或方向校验失败 |
| 4 | `E_OUTPUT_IO` | 输出目录不可写或写出失败 |
| 5 | `E_CACHE_IO` | 缓存目录不可写或写出失败 |
| 64 | `E_USAGE` | 命令行参数错误 |

场景错误和输出/缓存目录预检均发生在渲染前，不会生成部分帧。

## 验证

自动化测试覆盖：

- SHA-256 已知向量与 JSON 严格解析。
- 首轮烘焙、全缓存命中、方向单帧失效、材质全子键失效、分辨率全失效。
- PNG 结构/过滤解码/平均亮度、清单顺序与两位小数。
- 缺失元数据、损坏像素校验、残留临时文件的无失败修复。
- `E_SCENE_PARSE`、`E_SCENE_VALID`、`E_OUTPUT_IO`、`E_CACHE_IO`。
- Ctrl+C 退出码 0、SIGKILL 后恢复、正序与恢复输出逐字节比较。

实际执行结果见 [VERIFICATION.md](VERIFICATION.md)。

## 第三方代码

- `third_party/stb_image_write.h`：stb 单头文件 PNG 编码器，文件 SHA-256 为 `a8e7f77f0ccfdc9bac09a71268b131931ff81aafca82c393aeb6aaba10d9e7ca`。
