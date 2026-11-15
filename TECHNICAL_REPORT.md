# 技术说明

## 1. 目标与约束

实现是一个确定性的 CPU 离线渲染器。输入是包含透视相机、单个任意方向平行光与非空球体列表的 JSON 场景；输出为未裁切的线性 HDR 数组和显示编码 PNG。

关键约束：

- 无 GPU、无显示设备、无 Docker；
- 不使用伪随机数生成器、系统时钟或随机种子；
- 每像素采样数 1..64；
- 1、2、4 线程结果逐像素完全一致；
- 线性 HDR 是统计与报告的唯一来源；
- 色彩显示仅由 ACES filmic 与 sRGB 编码决定。

## 2. 场景与业务校验

`matrender.scene.parse_scene` 将 JSON 转换为冻结 dataclass。校验覆盖：

- 必需对象和必需字段；
- 有限数值、整数字段、布尔字段；
- 相机 `near < far`、位置与目标不同、`up` 非零且不与视线平行；
- 光源方向非零、强度非负、颜色非负；
- 球体半径严格为正；
- `base_color` 每通道 `[0,1]`；
- `metallic ∈ [0,1]`、`roughness ∈ (0,1]`；
- `normal_perturbation ∈ [0,1]`；
- SSAO 半径严格为正、强度非负、采样数 `[1,64]`。

任何业务错误统一包装为 `SceneValidationError`。CLI 捕获后输出 `E_INVALID_SCENE`，返回码 0，并且在校验通过前不创建输出目录。

## 3. 光线与几何

相机使用右手 look-at 基：

```text
forward = normalize(target - position)
right   = normalize(cross(forward, up))
up      = cross(right, forward)
```

像素中心或子像素坐标先转为 NDC，再由 FOV 与纵横比生成归一化射线。球体求交使用二次方程：

```text
oc = origin - center
b  = dot(direction, oc)
c  = dot(oc, oc) - radius^2
t  = -b - sqrt(b^2 - c)   （若大于 near）
```

若第一交点不在 `near` 前，则取第二交点。所有球体遍历时以严格更小距离选择最近命中，索引相同时保持低索引优先。

## 4. PBR 光照

光源 `direction` 表示光线的传播方向，因此表面到光源方向为 `L = -normalize(direction)`。着色在线性 RGB 中进行。

基础量：

```text
alpha = max(roughness^2, 1e-4)
F0    = 0.04 * (1 - metallic) + base_color * metallic
```

GGX/Trowbridge-Reitz 法线分布：

```text
D = alpha^2 / (pi * ((N·H)^2 * (alpha^2 - 1) + 1)^2)
```

Schlick Fresnel：

```text
F = F0 + (1 - F0) * (1 - V·H)^5
```

直接光几何项使用 Schlick-GGX：

```text
k  = (roughness + 1)^2 / 8
G1 = NdX / (NdX * (1 - k) + k)
G  = G1(N·V) * G1(N·L)
```

镜面项与漫反射项：

```text
specular = D * G * F / (4 * (N·V) * (N·L))
diffuse  = (1 - F) * (1 - metallic) * base_color / pi
```

表面点沿几何法线偏移 `1e-5` 后向 `L` 发射阴影射线；任一球体在有限正距离内遮挡时，直接光贡献为零。

常量环境照明为 `ambient_color * ambient`，经 SSAO 因子 `AO` 调制：

```text
ambient_brdf = albedo * (1 - metallic) + F0
ambient      = ambient_irradiance * ambient_brdf * AO
```

最终线性值为 `ambient + direct`，不裁切，保留 HDR 高光。

### 法线扰动

每个球体以世界坐标定义解析高度场：

```text
h(u,v) = A * sin(10u + 0.7) * cos(13v - 0.4)
```

其中 `u/v` 是命中点在切线/副切线上的坐标。着色法线由高度场的两个解析偏导构造：

```text
N' = normalize(N - dh/du * T - dh/dv * B)
```

扰动只用于材质着色；阴影偏移和 SSAO 使用未扰动的几何法线，避免自阴影与 AO 噪声。

## 5. 确定性抗锯齿

采样数为 1 时使用像素中心。采样数为 `S > 1` 时：

```text
grid_width = ceil(sqrt(S))
gx = i mod grid_width
gy = i div grid_width
u = (gx + 0.05 + 0.90 * hash01(x, y, salt_x(i))) / grid_width
v = (gy + 0.05 + 0.90 * hash01(x, y, salt_y(i))) / grid_width
```

`hash01` 是对像素整数坐标、采样索引和固定盐值做 64 位整数混合，再取 24 位映射到 `[0,1)`。它不使用 NumPy/Python 随机数发生器、时间或操作系统熵。

每个子采样独立求交与着色，最后按采样顺序用 `float64` 累加并除以 `S`，最终一次性写为 `float32`。因此像素结果不依赖线程调度或分块执行顺序。

## 6. SSAO

主射线在像素中心生成 G-buffer：

- 视线距离 `depth`；
- 世界空间几何法线 `normal`；
- 最近球体索引；
- 归一化主射线方向。

SSAO 只消费深度、法线和相机参数。对每个命中像素 `P`：

1. 由法线构造切线基；
2. 用另一个确定性整数哈希旋转切线基；
3. 使用 32 个黄金角半球采样点（样例场景）；
4. 采样点为 `Q = P + (T*x + B*y + N*z) * radius * scale`；
5. 将 `Q` 投影回屏幕，读取该像素深度并重建邻域表面 `Qn`；
6. 若邻域视空间深度比采样点更靠近相机，且位于 `2 * radius` 深度范围内，则产生遮蔽；
7. 遮蔽贡献经 `bias = 0.02 * radius`、距离范围和强度归一化；
8. `AO = clamp(1 - intensity * occlusion, 0, 1)`。

遮蔽只乘到环境照明，不削弱直接光。半径、强度与 SSAO 采样数均可配置。

固定对比场景 `scenes/ssao_contrast.json` 使用高环境照明和相互重叠的球体形成稳定接触遮蔽。官方 384x256、2 spp 输出中，开启/关闭 SSAO 的线性图像平均绝对差实测为 **0.0244164925**，高于 0.02。

## 7. 色彩管理

生产显示变换使用 Narkowicz ACES filmic 拟合：

```text
x = max(linear_rgb, 0)
y = clamp((x * (2.51x + 0.03)) / (x * (2.43x + 0.59) + 0.14), 0, 1)
```

随后应用 sRGB IEC 61966-2-1 传递函数：

```text
s = 12.92y                              , y <= 0.0031308
s = 1.055 * y^(1/2.4) - 0.055           , y > 0.0031308
```

量化方式为 `round(s * 255)`，裁剪到 `[0,255]` 后写 PNG。

测试包含一个完全独立的标量参考实现，遍历 39 个 HDR 通道值。实测每通道最大量化误差为 **0**，要求上限为 2。

## 8. 分块多线程

分辨率固定后，图像按固定规则划分为 64x64（可配置）的 row-major 方格。最后一行/列允许不足一个方块。

执行模型：

- 场景球体参数、G-buffer、SSAO 因子作为只读数组共享；
- 每个 worker 为自己负责的块分配独立 `float64` 累加缓冲区；
- worker 完成全部采样和着色后返回块坐标与 `float32` 块；
- 主线程按 `executor.map` 的固定 row-major 输入顺序写回最终数组；
- 所有块合并后才生成整图统计、掩码和逐球体报告。

每个像素只由自己所属块计算，跨块没有浮点归约，因此线程数不会改变像素运算顺序。

实测 96x72、2 spp 下，2/4 线程相对 1 线程的最大逐像素差均为 **0.0**，报告对象完全相等。

## 9. 资源预检与降级

估算模型：

```text
estimate =
  W * H * (24 * samples + 24 + 60)
  + W * H * 24
  + threads * min(tile_size, W) * min(tile_size, H) * 128
```

其中每采样 RGB 缓冲为 `3 * float64 = 24` 字节。虽然实际实现按块临时分配，预检故意使用整帧每采样缓冲模型，以获得与调度无关的保守上界。

超过上限时：

1. 采样数逐次减半，最低到 1；
2. 若仍超限，宽度和高度各乘 0.75 向下取整，直到 1x1；
3. 每一步写入 `degradation_events`；
4. 向 stderr 输出 `E_RESOURCE_LIMIT`；
5. 不中止进程，继续渲染并完成色彩转换、PNG、日志和报告。

实测 96x72、8 spp 在 0.5 MiB 上限下，从估算 2,139,136 B 降级到 54x40、1 spp、350,656 B；进程返回 0 并写出全部输出。

## 10. 统计与材质报告

可见球体掩码来自像素中心主射线命中的最近球体。这样每个像素最多属于一个球体，区域不重叠，所有球体都有独立条目。

区域统计直接从合并后的线性 HDR 图像计算：

- `linear_hdr_min/max`：逐 RGB 通道最小值/最大值；
- `linear_scalar_min/max`：全部 RGB 标量最小值/最大值；
- `luminance_min/max`：Rec.709 亮度最小值/最大值；
- `average_luminance`：区域亮度均值；
- `max_luminance`：区域亮度最大值；
- `dominant_hue_deg`：区域平均线性 RGB 归一化后的 HSV 色相角。

报告不使用内部光照中间量，因此与“从 HDR 图像和对应掩码重新采样”的定义天然一致。官方 12 球体输出实测最大相对统计误差和最大色相误差均为 **0.0**。

## 11. 复现与限制

重新运行：

```bash
.venv/bin/python scripts/verify.py
```

该命令执行测试、生成官方全分辨率工件并重新写入 `verification/results.json`。

当前实现有意限定为：

- 球体几何，不含三角网格或纹理；
- 单个方向光；
- 常量环境照明；
- 屏幕空间近似 SSAO，不是路径追踪全局光照；
- ACES filmic 为 Narkowicz 近似，不是完整 ACES RRT/ODT 色彩管理链。

这些限制不影响本任务规定的材质阵列、回归一致性、色彩转换、SSAO 对比和资源降级验收。

