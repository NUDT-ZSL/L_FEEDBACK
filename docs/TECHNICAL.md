# 技术说明

## 渲染顺序

每一帧创建一组临时 FBO/纹理，并在 `finally` 中释放：

1. **光空间预通道**：从平行光视线方向以正交投影绘制建筑，写入 `R32UI` ID 附件和 `GL_DEPTH_COMPONENT24` 深度附件。
2. **相机预通道**：以插值后摄像机的 view-projection 绘制建筑，写入独立 `R32UI` ID 附件和 24 位深度附件。此深度也用于后续不透明深度测试和玻璃遮挡。
3. **不透明底色**：全屏生成程序化天空，再使用相机预通道深度绘制建筑底色。
4. **体积光通道**：全屏积分到独立 `RGBA16F` 纹理。RGB 为雾环境项和散射项，A 为到射线终点的透射率。
5. **体积合成**：`base.rgb * volume.a + volume.rgb`。此时玻璃尚未绘制。
6. **半透明通道**：按远到近顺序绘制正面朝向的玻璃，开启深度测试、关闭深度写入，使用 `SRC_ALPHA, ONE_MINUS_SRC_ALPHA`。
7. **量化读回**：以 `RGBA16F` 读回，CPU 用固定公式量化到 8 位并编码 PNG。

## 深度与 ID 附件

moderngl 的 depth texture/renderbuffer 使用 `GL_DEPTH_COMPONENT24`；本项目使用 `ctx.depth_texture(...)`，因此深度由固定精度 24 位附件承载。

建筑 ID 使用 `GL_R32UI`：

- 相机预通道 ID FBO：用于保留每像素最近建筑 ID，并与相机深度一致。
- 光空间预通道 ID FBO：体积光每步投影到光空间后，先读取 ID；ID 为 0 时不遮挡，非 0 时再比较光空间深度和 bias。

两个 ID 附件及深度附件都位于每帧新创建的资源集中，并在通道开始时显式清为 ID=0、depth=1。资源集在帧末释放，不跨帧复用。

## 体积光积分

体积通道对每个屏幕像素执行：

```text
ray = normalize(reconstruct_world(ndc, far_depth) - camera_position)
ray_end = distance_to_camera_depth_buffer_or_far
step = (far - near) / steps
```

对第 `i` 个采样区间：

```text
t0 = near + i * step
t1 = min(t0 + step, ray_end)
dt = t1 - t0
t  = (t0 + t1) / 2
p  = camera_position + ray * t
v  = shadow_visibility(p)
T  = exp(-sigma_t * t)
L  += T * sigma_s * HG(-dot(ray, L_dir), g) * light_color * intensity * v * dt
```

其中：

```text
sigma_t = fog.density * fog.extinction
sigma_s = sigma_t * fog.scattering
```

`HG` 为 Henyey-Greenstein 相位函数。最终体积纹理为：

```text
volume.rgb = L + fog.color * fog.ambient * (1 - exp(-sigma_t * ray_end))
volume.a   = exp(-sigma_t * ray_end)
```

`step` 每帧根据插值后的 `near`/`far` 重新计算，因此步长随摄像机近远平面变化。`fog.steps` 同时控制采样数量和积分精度。

光空间遮挡测试：

```text
light_ndc = light_view_proj * world_position
id        = texelFetch(shadow_id, light_uv)
if id == 0: visible
else: visible = sample_light_depth <= stored_light_depth + shadow.bias
```

这使体积光在建筑阴影内变暗，而不是只按相机深度削减整段雾。

## 玻璃排序和混合

每个面板中心的世界位置转换到当前摄像机前向坐标：

```text
view_depth = dot(panel_center - camera_position, camera_forward)
depth_key  = round(view_depth, 6)
```

排序规则：

1. `depth_key` 降序（远到近）。
2. `depth_key` 相等时，`panel_id` 升序。

统计 JSON 为所有面板记录 `order_index`、`panel_id`、`view_depth`、`depth_key`、`front_facing`、`behind_camera` 和 `rendered`。实际绘制顺序记录在 `drawn_order_indices` 和 `drawn_panel_ids`。

面板法线由多边形绕序计算。`dot(normal, camera_position - center) > 0` 才是正面；背面面板不发出 draw call。绘制时深度测试为 `<=`，深度写入关闭。面板 alpha 为 `1 - transmittance`，因此最终近似为：

```text
glass_over(background) = glass_color * (1 - transmittance)
                       + background * transmittance
```

由于体积光已在独立通道先合成，玻璃 alpha 会衰减其后方已经合成的雾散射，并叠加自身颜色；体积光不会被追加到已完成玻璃混合的画面上。

## 摄像机插值

帧时间为：

```text
count = floor((end_time - start_time) * fps + 1e-9) + 1
time_i = start_time + i / fps
```

对帧时间 `t` 找到满足 `t0 <= t <= t1` 的关键帧段：

```text
alpha = (t - t0) / (t1 - t0)
position = (1 - alpha) * p0 + alpha * p1
orientation = normalize(shortest_path_slerp(q0, q1, alpha))
fov, near, far = linear interpolation
```

若 `t` 恰好等于某个中间关键帧时间，采用以该关键帧为起点的后一段（`alpha=0`）；若等于最后一个关键帧时间，则采用最后一段且 `alpha=1`。这与实现中的 `bisect_right` 规则一致。

四元数使用 `[x, y, z, w]`。若 `dot(q0, q1) < 0`，先把 `q1` 取反以走短弧。每帧统计中记录段索引、段端点时间、`alpha`、实际位置、实际朝向、FOV、near/far 和前/右/上方向，外部可直接复算。

## 确定性

- 样本生成不使用当前时间、随机种子或外部文件。
- 所有 GL 中间颜色缓冲为 `RGBA16F`，最终量化公式固定：

```text
channel8 = floor(clamp(channel, 0, 1) * 255 + 0.5)
```

- 统计浮点字段统一保留 6 位小数；负零统一为 0。
- 面板排序使用公开的 1e-6 深度量化键。
- PNG 使用固定 Pillow 参数：`compress_level=9`、`optimize=False`、不写元数据。
- 每帧统计无时间戳；只有聚合 `stats.json` 含 `generated_at`。

## 资源与并发

每个 worker 最多持有一个 standalone GL context 和一套静态程序/VBO。`--parallel-frames N` 时最多创建 N 个 worker context；帧索引进入队列，未开始的帧不会创建额外 context。

单帧创建的所有 FBO、颜色纹理和深度纹理由 `FrameResources.release()` 释放；释放前先解绑到默认 framebuffer，避免软件驱动保留无效绑定。帧结束时调用 `ctx.gc()`。

任一线程的单帧异常会转换为 `E_FRAME_RENDER`，context 中包含 `frame_index`，并设置停止标志，不再启动排队中的后续帧。

统计 JSON 和错误 context 中的 `frame_index` 均为 0 基；终端进度显示为 1 基。

## 校验模式

`render --validate-only`：

- 要求目标目录已有完整帧文件和 `stats.json`。
- 用相同参数在内存中重新渲染。
- PNG 使用原始字节比较。
- 逐帧 JSON 和聚合 JSON 规范化后比较；比较前移除 `generated_at`/`timestamp`。
- 不创建、不覆盖任何输出文件。
- 全部一致返回 0；缺失返回 40；不一致返回 41。

