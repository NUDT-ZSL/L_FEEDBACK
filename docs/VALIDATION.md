# 实际验证记录

## 1. 环境

| 项目 | 值 |
|---|---|
| 日期 | 2026-11-16 |
| OS | Ubuntu 26.04.1 LTS |
| 编译器 | GCC 15.2.0 |
| CMake | 3.30.5 |
| Vulkan loader | 1.4.341 |
| Vulkan 实现 | Mesa lavapipe，API 1.4.335 |
| 验证层 | `VK_LAYER_KHRONOS_validation` 1.4.341，实际启用 |
| GPU | llvmpipe (LLVM 21.1.8, 256 bits) |

当前执行环境不是 Windows。程序使用标准 Vulkan 1.3、C++20 和
`#ifdef _WIN32` 分配器分支，CMake 提供 MSVC 2022 配置；Windows 实机编译/运行
需按 README 的命令执行。本文件记录的是本机 Vulkan 1.3+ 路径的真实运行结果。

## 2. 构建

```bash
export VULKAN_SDK=/home/liang/.cache/uid90-tools/sdk
export PATH=/home/liang/.cache/uid90-tools/root/usr/bin:$PATH
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-linux -j 4
```

构建结果：

- `volume_renderer`、`scene_gen` 链接成功。
- 6 个 GLSL 阶段成功编译为 SPIR-V：
  `shadow.vert`、`gbuffer.vert`、`gbuffer.frag`、`fullscreen.vert`、
  `composite.frag`、`volume.comp`。

## 3. 确定性样本

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export VK_LAYER_PATH=/home/liang/.cache/uid90-tools/root/usr/share/vulkan/explicit_layer.d

build-linux/scene_gen sample
build-linux/volume_renderer \
  --scene sample/scene.json \
  --out validation/render.png \
  --stats validation/stats.json \
  --reference-out validation/reference.png \
  --recovery-dir validation/recovery \
  --shader-dir build-linux/shaders \
  --validation on
```

关键输出：

```text
device: llvmpipe (LLVM 21.1.8, 256 bits) api=1.4.335 validation=on
frames_identical=true
volume_samples_frame_a=6659011
volume_samples_frame_b=6659011
mean_abs_luminance_error=0.000278 threshold=0.020000
cleanup: resources_live=0 allocator_live=0 device_lost=false
```

产物哈希：

```text
scene.json SHA-256  b83825fda81b233929e567911696c079e1818dc5a53489a73e33f30c0450f4bc
noise.raw SHA-256   673263c5a4852a3cda4a4b9ec0ffa8f01a0f09bff771c3e88f6e5d7c2ee5979d
render.png SHA-256  4b4a5e6082ddb6b4061aed6e98d1913d3b913e356cab5559ec5ad88705927c62
reference.png       138e6f97c3d49d47c63c9dcd5ec77b74139ea5093a30bd57bd8b97e21533d1cf
```

`stats.json` 中两个主帧 PNG 哈希和原始像素哈希均相同：

```text
frameASha256 = frameBSha256 = 4b4a5e6082ddb6b4061aed6e98d1913d3b913e356cab5559ec5ad88705927c62
pixel A      = pixel B      = f8997139a83898197c1b4eb97c65886dbc290e151f939f0724c39f72933b2f3e
```

另外在独立输出目录、独立进程再次执行同一场景，得到同一 PNG SHA-256。

## 4. 固定亮度参考

参考配置固定为同一实现、同一噪声 seed、`stepMeters=0.030`、
`maxSteps=512`、相同 camera/lights/medium/shadow/PCF/exposure。

实测：

```text
reference.meanAbsoluteLuminanceError = 0.000278
threshold                            = 0.020000
passed                               = true
```

度量方式为 sRGB 解码后的 Rec.709 线性亮度逐像素平均绝对误差。

## 5. 步长收敛

使用 `--volume-max-steps 768 --no-reference`，分别以
`0.18/0.12/0.08 m` 相对 `0.03 m` 高质量帧计算 Rec.709 亮度误差：

| step (m) | 平均绝对亮度误差 |
|---:|---:|
| 0.18 | 0.00046063 |
| 0.12 | 0.00027783 |
| 0.08 | 0.00015525 |

结果随步长减小严格单调下降，且均远小于 `0.02`；采样使用像素射线上的确定性
中点分层，没有逐帧随机抖动。

## 6. 业务错误码

实际命令均带独立恢复目录，验证层关闭以隔离错误注入：

| 注入/条件 | stderr 业务码 | 退出码 | 清理日志 |
|---|---:|---:|---|
| schemaVersion=99 | `E_SCHEMA` | 2 | `resources_live=0 allocator_live=0` |
| `--inject shader` | `E_SHADER` | 3 | `resources_live=0 allocator_live=0` |
| `--inject alloc` | `E_ALLOC` | 4 | `resources_live=0 allocator_live=0` |
| `--inject desc` | `E_DESC` | 5 | `resources_live=0 allocator_live=0` |
| `--inject device-lost` | `E_DEVICE_LOST` | 6 | `resources_live=0 allocator_live=0` |

注入后的实际 stderr 摘要：

```text
E_SCHEMA: schema: unsupported schemaVersion 99 (expected 1)
E_SHADER: injected shader compilation failure while loading shadow.vert.spv
E_ALLOC: injected Vulkan memory allocation failure (VkResult -1)
E_DESC: injected descriptor set exhaustion after allocating 1 frame set(s)
E_DEVICE_LOST: injected VK_ERROR_DEVICE_LOST after persisting the last complete frame
```

## 7. 设备丢失恢复

`--inject device-lost` 在主帧完成后返回业务码 6。恢复目录内容：

```text
volume_light_recovery.rgba   331776 bytes = 384*216*4
volume_light_recovery.json   complete=true, frameIndex=1, pixelSha256=...
volume_light_recovery.ready  最后写入
```

以同一恢复目录下一次启动时，stderr 实际出现：

```text
recovery: complete frame already exists at .../volume_light_recovery.rgba
recovery-metadata: { ... "complete": true ... }
```

成功运行并指定 `--clear-recovery` 后，三个固定文件全部删除。

## 8. 自动验收

```bash
ctest --test-dir build-linux --output-on-failure
```

结果：

```text
1/1 Test #1: volume_renderer_acceptance ....... Passed  5.79 sec
100% tests passed, 0 tests failed out of 1
```

测试脚本实际覆盖：

- 内置生成器两次产出的 JSON/raw 逐字节一致，raw 尺寸为 131072；
- 独立进程 PNG 哈希一致；
- 双主帧 PNG/像素哈希一致；
- 统计字段完整、采样数非零、阴影时间为正；
- 固定参考亮度误差 `<=0.02`；
- `E_SCHEMA/E_SHADER/E_ALLOC/E_DESC/E_DEVICE_LOST`；
- 注入后资源计数为 0；
- 设备丢失恢复文件、下次启动提示和显式清理；
- 三档步长单调收敛。

脚本最终输出：

```text
PASS
png_sha256=4b4a5e6082ddb6b4061aed6e98d1913d3b913e356cab5559ec5ad88705927c62
reference_luminance_error=0.000278
step_errors=[0.00046063, 0.00027783, 0.00015525]
injection_codes=E_SHADER,E_ALLOC,E_DESC,E_DEVICE_LOST
recovery=verified
```

## 9. 复现边界

上述哈希是在固定 loader/驱动/设备/lavapipe 浮点实现下获得。验收强制检查
同机双帧与同机独立进程逐字节一致。跨 GPU/驱动不能承诺 SPIR-V 浮点结果逐位
相同；若验收要求跨设备 golden hash，应固定设备和驱动版本，或使用提供的
固定参考帧与误差阈值。
