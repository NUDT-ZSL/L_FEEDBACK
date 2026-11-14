# 教学用离屏延迟渲染器（PyOpenGL + GLFW 隐藏窗口）

一个完全离屏、确定性的两阶段延迟渲染器。它通过 GLFW 创建**隐藏窗口**承载
OpenGL 3.3 Core 上下文，不需要真实窗口显示；几何阶段把世界位置、法线、反照率
/材质写入多渲染目标（MRT）G-Buffer，光照阶段读取 G-Buffer 做简化
Cook-Torrance BRDF 着色，输出 PNG 帧序列与 `manifest.json` 日志。

## 环境准备

- Python 3.10+
- 支持 OpenGL 3.3+ 的驱动（开发机使用 Mesa llvmpipe 软件光栅，GL 4.5）
- 安装依赖：

```bash
pip install -r requirements.txt
# 等价于：pip install numpy pyopengl glfw pillow
```

## 快速开始

```bash
# 默认：seed=42，float32，512x512，21 帧，第 10 帧后重建 G-Buffer
python -m deferred_renderer --out out/demo

# 半精度近似 BRDF
python -m deferred_renderer --fast-approx --out out/demo_fast

# 强制某附件使用“不支持”格式，触发降级链路
python -m deferred_renderer \
  --force-format world_position=UNSUPPORTED --out out/demo_fallback

# 不重建 G-Buffer 的对照运行（用于逐像素比对）
python -m deferred_renderer --no-rebuild --out out/demo_control
```

## 命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--seed` | `42` | `numpy.random.default_rng(seed)` 的确定性种子 |
| `--frames` | `21` | 渲染帧数（帧号从 `0000` 连续递增） |
| `--width` / `--height` | `512` | 帧尺寸 |
| `--out` | `out/seed_<seed>_<mode>` | 输出目录 |
| `--fast-approx` | 关 | 光照 BRDF 中间量走 GLSL fp16 仿真 |
| `--force-format` | 无 | `附件名=格式`，可重复，如 `world_normal=UNSUPPORTED` |
| `--no-rebuild` | 关 | 关闭第 10 帧后的 G-Buffer 重建（对照实验） |

合法附件名：`world_position`、`world_normal`、`albedo_material`。
可选格式 token：`RGBA32F`、`RGB32F`、`RGBA16F`、`RGB16F`、`RGBA8`，
以及保证触发失败的 `UNSUPPORTED`（映射到非法 internalformat）。

## G-Buffer 附件契约（槽位必须一一对应）

| 槽位 | 几何着色器输出 | 附件名 | 内部格式 | 光照采样器 |
| --- | --- | --- | --- | --- |
| 0 | `gPosition` | `world_position` | `RGBA32F` | `gPosition` |
| 1 | `gNormal` | `world_normal` | `RGB16F` | `gNormal` |
| 2 | `gAlbedoMaterial` | `albedo_material` | `RGBA8` | `gAlbedoMat` |

- 位置附件 `xyz` 存世界坐标，`w` 存线性视距。
- 反照率附件 `RGB` 存反照率，`A` 以 8 位 unorm 编码材质 id，
  光照阶段解码后再查粗糙度/金属度/自发光。
- 几何阶段启用深度测试（`LEQUAL`）与背面剔除（`GL_BACK` + CCW）。

## 产物

- `frame_XXXX.png`：默认帧缓冲回读后翻转 Y 写出的 RGBA PNG。
- `scene.json`：确定性场景（相机、材质、球体、点光源、平行光）。
- `manifest.json`：GPU 厂商/OpenGL 版本、精度模式、请求与实际附件格式、
  每帧 PNG 与 G-Buffer 哈希、`events`（重建）、`warnings`（降级）、`errors`。
- `errors.jsonl`：结构化错误/降级日志（每行一条 JSON）。

`png_sha256` 直接对**磁盘上的 PNG 字节**取 SHA-256，可用外部工具复核。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 退出码约定

- `0`：渲染成功，包括发生格式降级但成功恢复的情况。
- `1`：无法继续渲染（如默认格式也失败的 `RENDER_FBO_INCOMPLETE`，
  或 G-Buffer 回读不一致的 `RENDER_STATE_MISMATCH`）。进程不会崩溃或挂起。

更多实现细节与验证记录见 `docs/TECHNICAL_NOTES.md`。
