# 技术规格

## 1. 坐标、单位与确定性

- 坐标系：右手系，Y 轴向上，长度单位均为米（m）。
- 水平/垂直视场角：度（deg）。
- 相机近/远裁剪面：米。
- 场景颜色、光源颜色、反射率：线性 `[0,1]`，不做 sRGB 预转换。
- 方向光 `intensity`：相对辐照度，无量纲，范围 `[0,100]`。
- 点光 `intensity`：相对辐射强度，范围 `[0,1000]`，随距离按
  `1/(1+d^2)` 和 range window 衰减。
- 介质系数：`density` 为相对密度；`absorption/scattering` 为
  `density=1` 时的每米系数。
- 噪声：`32x32x32` little-endian float32，按 `(x + y*32 + z*1024)` 排列，
  每个值有限且位于 `[0,1]`，3D 采样使用 repeat + linear。
- PNG：RGBA8，sRGB 编码，逐像素按行写出。

确定性的基础是固定 JSON、固定 `.raw` 字节、固定可执行文件、固定设备和驱动。
程序对同一进程的两个独立帧、以及两次独立进程执行，检查 PNG 字节哈希。相同
输入下时间统计不要求相同。

## 2. 场景 JSON

当前 `schemaVersion` 固定为 `1`。版本缺失、非整数或不等于 `1` 时输出
`E_SCHEMA`，退出码 `2`。JSON 语法错误、字段缺失、类型错误、重复键或数值越界
同样属于 `E_SCHEMA`。

### 2.1 顶层字段

| 字段 | 类型 | 范围/约束 |
|---|---|---|
| `schemaVersion` | integer | 必须为 `1` |
| `name` | string | 可选，默认 `unnamed` |
| `camera` | object | 相机 |
| `output` | object | 输出 |
| `noise` | object | 噪声纹理 |
| `volume` | object | 体积介质 |
| `directionalLight` | object | 方向光 |
| `pointLight` | object | 点光 |
| `shadows` | object | 阴影设置 |
| `geometry` | object | 地面与 0..16 个球体 |

### 2.2 `camera`

| 字段 | 范围 |
|---|---|
| `position`, `target`, `up` | 有限 `[x,y,z]`；target 不得等于 position；up 不得与视线平行 |
| `fovYDegrees` | `[10,120]` |
| `nearMeters` | `[0.01,10]` |
| `farMeters` | `[1,1000]`，且大于 near |

### 2.3 `output`

| 字段 | 范围 |
|---|---|
| `width` | `[64,3840]` |
| `height` | `[64,2160]`，总像素不超过 `8,294,400` |
| `exposure` | `[0.01,32]` |
| `gain` | `[0,8]` |
| `background` | 可选，3 个线性分量，各 `[0,1]` |
| `pngPath` | 必填，相对场景目录解析 |
| `statsPath` | 可选，默认 `stats.json` |
| `referencePngPath` | 可选，默认 `render.reference.png` |

命令行 `--out/--stats/--reference-out` 优先于 JSON 路径。

### 2.4 `noise`

| 字段 | 约束 |
|---|---|
| `path` | `.raw` 路径，相对场景目录解析 |
| `dimension` | 必须为 `32` |
| `seed` | 非负 32 位整数；样本生成器固定使用该 seed |

`.raw` 必须恰好为 `32*32*32*4 = 131072` 字节；尺寸不符或样本非有限/越界为
`E_SCHEMA`。

### 2.5 `volume`

| 字段 | 范围 |
|---|---|
| `boundsMin/boundsMax` | 各轴 min < max |
| `density` | `[0.0001,2.0]` |
| `absorption` | `[0,20]` |
| `scattering` | `[0,20]` |
| `anisotropy` | `[-0.95,0.95]` |
| `stepMeters` | `[0.01,1.0]` |
| `maxSteps` | `[8,2048]` |
| `noiseFrequencyPerMeter` | `[0.001,4.0]` |

### 2.6 光源与阴影

- `directionalLight.direction`：从光源指向场景的非零向量，着色器内归一化；
  `color` 各分量 `[0,1]`；`intensity` `[0,100]`。
- `pointLight.position`：米；`color` 各分量 `[0,1]`；`intensity` `[0,1000]`；
  `rangeMeters` `[0.1,100]`。
- `shadows.resolution`：`[128,4096]` 像素，方向和点光阴影共用。
- `shadows.pcfKernel`：奇数 `[1,9]`，在方向和点光 PCF 中同时生效。
- `shadows.directionalBias`、`pointBias`：`[0,0.05]`，同时用于深度管道
  constant/slope bias 和查询深度偏移参数。

### 2.7 `geometry`

- `ground.height`：米；`sizeMeters` `[1,100]`；`albedo` 各分量 `[0,1]`。
- `spheres`：最多 16 个；`center` 米；`radius` `[0.01,20]`；`albedo`
  各分量 `[0,1]`。

## 3. 渲染管线

固定顺序：

1. 方向光正交阴影图：单张 `D32_SFLOAT`，R32 视图。
2. 点光立方体阴影图：`VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT` 的 6 层深度图，
   每面 90 度透视投影、独立 view/framebuffer。
3. G-buffer：`RGBA16F` 不透明颜色 + `D32_SFLOAT` 深度，深度测试/写入。
4. 计算着色器体积积分：以 G-buffer 深度为截止距离，在体积 AABB 内按
   `ceil(segmentLength/stepMeters)` 分层、`maxSteps` 截断，中点采样。
5. 全屏合成：`(G-buffer 颜色 + 体积散射) * exposure * gain`，线性空间
   Reinhard `x/(1+x)`，再逐通道 sRGB transfer，写入 `RGBA8_UNORM`。
6. 拷贝到 host，编码确定性 PNG，并写统计 JSON。

体积积分：

- 密度扰动为 `density * clamp(0.35 + 1.30*noise, 0.15, 1.8)`。
- `sigmaS = scattering * modulatedDensity`，
  `sigmaA = absorption * modulatedDensity`。
- Henyey-Greenstein 相位函数使用 JSON `anisotropy`。
- 每个采样点分别查询方向光 PCF 和点光立方体 PCF。
- `transmittance` 从前向后累积；散射贡献乘当前透射率。
- 每个像素的实际执行步数通过 storage buffer `atomicAdd` 统计。

## 4. PCF 与立方体接缝处理

方向光：

- 阴影坐标 `uv` 超出 `[0,1]` 时直接返回全阴影。
- 核内 PCF tap 钳制到 `[0.5/res, 1-0.5/res]` 的边界纹素中心。

点光：

- 按主轴向和符号选择 6 个面之一，使用与渲染矩阵一致的 `sc/tc` 映射。
- PCF tap 越出当前面时，将未钳制的 `uv` 转回方向，再重新选面，从而跨共享
  边缘采样相邻面，避免硬断层。
- 最终采样坐标钳制到半纹素边界；面深度使用同一 near/far 的投影深度，因此
  跨面比较保持连续。
- 深度查询关闭硬件 compare，使用 nearest 深度加显式 PCF，避免过滤插值污染
  深度比较。

## 5. 双缓冲与资源管理

- 每个主帧拥有独立 UBO、采样计数器、G-buffer 颜色/深度、体积图、输出图、
  readback buffer、descriptor set、command buffer 和 fence。
- 阴影图和噪声图只生成一次，由两个 descriptor set 共享。
- 所有 Vulkan 对象和内存分配经自定义计数包装；清理后打印
  `cleanup: resources_live=0 allocator_live=0`。
- 错误注入发生在部分资源已分配之后，仍走统一清理路径。

## 6. 业务错误码

| 业务码 | 退出码 | 触发条件 |
|---|---:|---|
| `E_SCHEMA` | 2 | schema 版本不匹配；JSON 语法/字段/类型/范围/重复键错误；噪声尺寸或值非法 |
| `E_SHADER` | 3 | SPIR-V 缺失、长度/魔数非法；`--inject shader` 注入编译失败 |
| `E_ALLOC` | 4 | `VK_ERROR_OUT_OF_HOST_MEMORY`/`VK_ERROR_OUT_OF_DEVICE_MEMORY`；`--inject alloc` |
| `E_DESC` | 5 | descriptor pool 耗尽/碎片化；`--inject desc` |
| `E_DEVICE_LOST` | 6 | 任一受检查 Vulkan 调用返回 `VK_ERROR_DEVICE_LOST`；`--inject device-lost` |
| `E_IO` | 7 | 文件打开/读取/输出失败 |
| `E_VULKAN` | 8 | 其他受检查 Vulkan 失败 |
| `E_UNSUPPORTED` | 9 | loader < 1.3、无合格设备、所需格式特性缺失 |
| `E_NONDETERMINISTIC` | 10 | 两个主帧 PNG/像素不一致 |
| `E_REFERENCE` | 11 | 参考亮度误差大于 `0.02` |
| `E_RECOVERY` | 12 | 恢复目录/文件错误；清理后自定义资源计数非零 |
| `E_USAGE` | 13 | 命令行参数错误 |
| `E_INTERNAL` | 1 | 未分类 C++ 异常 |

错误信息写入 stderr，格式为 `业务码: 描述`。错误注入路径同样打印资源清理
日志，并要求 `resources_live=0`。

## 7. 设备丢失持久化恢复

默认恢复目录：

- Windows：`%TEMP%\volume-light-renderer-recovery\`
- Linux：`$TMPDIR/volume-light-renderer-recovery/`（通常为 `/tmp/...`）

可用 `--recovery-dir` 覆盖。固定文件名：

- `volume_light_recovery.rgba`：最近一次完整主帧的 RGBA8、逐行 top-down 数据，
  大小严格为 `width*height*4`。
- `volume_light_recovery.json`：`complete`、scene 路径/SHA-256、帧号、宽高、
  像素 SHA-256。
- `volume_light_recovery.ready`：最后写入的就绪标记。

每完成一个主帧就原子替换完整帧；`ready` 最后写。若设备丢失发生在后续帧，
已存在的上一完整帧不会被部分数据覆盖。检测到 `VK_ERROR_DEVICE_LOST` 时：

1. 不再写入当前未完成帧；
2. stderr 输出 `E_DEVICE_LOST`，附恢复帧号和路径；
3. 以退出码 `6` 结束；
4. 保留上述固定文件。

下次启动如果发现 `ready` 和完整尺寸的 `.rgba`，在 stderr 打印
`recovery: complete frame already exists ...` 及 metadata。程序不会自动恢复或
自动消费该帧；用户可读取 raw 数据，或在一次成功运行后通过
`--clear-recovery` 删除三个恢复文件。

## 8. 固定亮度参考帧

参考来源完全固定为“同一可执行实现”的以下配置：

- 与主输出相同的场景 camera、lights、medium、noise seed、阴影分辨率、PCF、
  exposure 和 gain；
- `stepMeters = 0.030`；
- `maxSteps = 512`；
- 单独渲染并输出 `reference.png`。

主输出与参考输出均先按 IEC 61966-2-1 sRGB transfer 解码到线性值，再用
Rec.709 权重计算亮度：

`Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear`

统计量为所有像素 `|Y_main - Y_reference|` 的算术平均，阈值 `0.02`。该值写入
`stats.json` 的 `reference.meanAbsoluteLuminanceError`，同时记录 reference
参数与 SHA-256。

## 9. CLI

```text
volume_renderer --scene <scene.json> [options]
  --out <png>
  --stats <json>
  --reference-out <png>
  --shader-dir <dir>
  --recovery-dir <dir>
  --inject shader|alloc|desc|device-lost
  --volume-step <meters>
  --volume-max-steps <n>
  --frames <n>              # 默认 2，最少 2
  --no-reference
  --clear-recovery
  --validation on|off
```

## 10. 统计 JSON

`stats.json` 至少包含：

- 场景路径、schema 版本、场景 SHA-256；
- 设备名、Vulkan API 版本、驱动版本、验证层开关；
- PNG 路径/宽高/SHA-256、两个主帧哈希、`framesIdentical`；
- 两个主帧的体积采样总数、每像素平均采样数、参考帧采样数；
- 阴影图渲染耗时、两个主帧耗时、参考帧耗时；
- 固定参考来源、步长、最大步数、亮度误差、阈值、是否通过；
- 原始像素哈希 A/B 和 `sameInputSamePng`。

时间使用 CPU steady-clock 包围 `vkQueueSubmit + vkWaitForFences`，单位毫秒；
它用于性能统计，不参与图像确定性判定。
