# 技术说明

## 1. 数据流

```text
scene.json
  -> 严格 JSON 递归下降解析
  -> 场景结构和数值校验
  -> 世界矩阵 / 法线矩阵
  -> 视图-投影与视锥平面提取
  -> 每网格世界 AABB 粗剔除
  -> 三角形齐次坐标裁剪
  -> 屏幕空间重心坐标光栅化
  -> 深度缓冲测试
  -> 纹理层混合、Lambert 光照
  -> Reinhard 色调映射、sRGB 伽马
  -> 24 位 BMP（bottom-up，BGR）
```

所有渲染数学使用 `double`。不启用 fast-math，不使用多线程，不依赖 GPU、
OpenGL、Direct3D 或第三方运行时。

## 2. DirectXMath 与精度

Windows 构建在 `include/raster/dxmath_compat.h` 中包含 Windows SDK 的
`<DirectXMath.h>`，入口处调用 `DirectX::XMLoadFloat3` / `XMVectorGet*` 做头文件
与链接可用性校验。DirectXMath 的向量寄存器类型为单精度，直接用于投影、裁剪、
重心插值会违反本任务的双精度及可复现要求，因此渲染核心使用项目内的
`Vec2/Vec3/Vec4/Mat4` 双精度实现。非 Windows 环境不声明 DirectXMath 头文件，
但执行完全相同的渲染代码。

## 3. JSON 与校验

解析器实现标准 JSON 对象、数组、字符串、转义、Unicode 代理对、布尔、空值和
数字。拒绝：

- 重复对象键、尾随内容、非法转义和未终止字符串/数组。
- `NaN`、`Infinity` 这类非 JSON token。
- 超出双精度范围并变成非有限值的数字。

场景层继续校验所有必需字段和数值：

- 相机：位置、方向或目标、上方向、FOV、近远平面。
- 灯：方向非零、颜色分量 `[0,1]`、强度有限且非负。
- 材质：漫反射颜色 `[0,1]`；层颜色可选；纹理路径可选；层权重有限且非负。
- 网格：位置/法线/UV 顶点数一致；顶点色支持 RGB 或 RGBA，数量一致；索引数为
  3 的倍数且不越界；材质索引不越界；模型矩阵有限且可逆。
- 分辨率：`1..16384`。

字段别名包括：

- 相机 `direction` / `forward` / `target` / `lookAt`；
- FOV `fovY` / `fov` / `fovDegrees` / `verticalFov`；
- 顶点 `positions` / `vertices`；
- UV `uvs` / `uv` / `texcoords`；
- 材质 `material` / `materialIndex`；
- 纹理 `baseTexture` / `baseMap` / `texture`；
- 模型 `model` / `world` / `transform`。

纹理路径相对于场景文件目录解析。纹理不存在、不是合法 BMP 或不受支持时，
对应层回退到该材质的层颜色；未显式提供层颜色时回退到 `diffuse`。警告写入
stderr，不改变统计和退出码。

## 4. 坐标与矩阵

相机采用左手系：

```text
zAxis = normalize(direction)
xAxis = normalize(cross(up, zAxis))
yAxis = cross(zAxis, xAxis)
```

投影为 Direct3D 风格左手透视投影，裁剪空间满足 `0 <= z <= w`：

```text
xScale = 1 / (aspect * tan(fovY / 2))
yScale = 1 / tan(fovY / 2)
zClip  = z * far / (far - near) - near * far / (far - near)
wClip  = z
```

模型和视图-投影矩阵均为行主序：

```text
clip = Projection * View * Model * position
```

法线使用模型矩阵的逆转置矩阵变换。模型矩阵是任意可逆 4x4 时，位置变换也做
齐次除法，保证数据一致性。

## 5. 视锥剔除

从视图-投影矩阵行向量 `r0..r3` 提取六个平面：

```text
near   = r2
far    = r3 - r2
left   = r3 + r0
right  = r3 - r0
bottom = r3 + r1
top    = r3 - r1
```

平面法线归一化。对每个变换后的网格顶点求世界 AABB，再对每个平面选择 AABB
的正顶点：

```text
p.x = n.x >= 0 ? max.x : min.x
p.y = n.y >= 0 ? max.y : min.y
p.z = n.z >= 0 ? max.z : min.z
dot(n, p) + d < -1e-9  => 完全在平面外
```

任一平面判定完全在外时整网格剔除，不提交任何三角形。判定使用保守的负容差，
避免边界误差误剔除。屏外网格只增加 `culledMeshes`，不写颜色或深度缓冲。

## 6. 齐次裁剪

每个三角形依次对六个裁剪平面做 Sutherland-Hodgman 裁剪：

```text
near:   z >= 0
far:    w - z >= 0
left:   x + w >= 0
right:  w - x >= 0
bottom: y + w >= 0
top:    w - y >= 0
```

交点参数为 `t = da / (da - db)`，并截断到 `[0,1]` 以控制边界数值误差。
位置、世界坐标、法线、UV、顶点颜色全部在裁剪空间中线性插值。裁剪后的凸多边形
按 `(0,i,i+1)` 扇形重新三角化，因此跨越近平面的三角形可生成两个或更多屏幕
三角形。

## 7. 光栅化与透视校正

裁剪空间转屏幕坐标：

```text
xNdc = clip.x / clip.w
yNdc = clip.y / clip.w
sx   = (xNdc * 0.5 + 0.5) * width
sy   = (0.5 - yNdc * 0.5) * height
```

采样点固定为像素中心 `(x + 0.5, y + 0.5)`。三角形在屏幕空间统一为正面积方向，
即使输入绕序不同也不做背面剔除。边函数为零时只接受 top-left 边：

```text
edge(a,b,p) = (b.x-a.x)*(p.y-a.y) - (b.y-a.y)*(p.x-a.x)
top-left: (dy == 0 && dx > 0) || (dx == 0 && dy < 0)
```

设屏幕重心为 `(b0,b1,b2)`，顶点逆深度为 `q_i = 1/w_i`，则透视正确属性为：

```text
denom = b0*q0 + b1*q1 + b2*q2
attr  = (b0*q0*attr0 + b1*q1*attr1 + b2*q2*attr2) / denom
```

世界坐标、法线、UV 和 RGBA 顶点色均按此公式插值；法线最后再次归一化。

## 8. 深度

深度键不使用屏幕空间线性的 NDC `z`，而是归一化逆视图深度：

```text
D = (1/w - 1/far) / (1/near - 1/far)
```

`D=1` 在近平面，`D=0` 在远平面，范围 `[0,1]`。`1/w` 在屏幕空间线性，因此
`D` 可直接用屏幕重心线性插值：

```text
Dfrag = b0*D0 + b1*D1 + b2*D2
```

深度缓冲初值为 `0`，仅当 `Dfrag > depthBuffer[pixel]` 时通过并写回。相同深度
按先提交的三角形保留，这使结果依赖固定输入顺序但不依赖运行顺序，因而可复现。

## 9. 纹理与材质

BMP 纹理加载器支持未压缩 24/32 位、正高度 bottom-up 和负高度 top-down 文件。
纹理在内存中统一为 top-down 浮点 RGB，UV 原点按左下角解释。

重复寻址和最近邻采样：

```text
u' = u - floor(u)
v' = v - floor(v)
ix = floor(u' * width)
iy = floor(v' * height)
```

层颜色：

```text
base    = (baseTexture   ? sample(baseTexture, uv)   : baseColor)   * baseWeight
detail  = (detailTexture ? sample(detailTexture, uv) : detailColor) * detailWeight
t       = clamp(vertexColor.a, 0, 1)
layered = base * (1 - t) + detail * t
albedo  = diffuse * layered * vertexColor.rgb
```

因此顶点 alpha 是两层混合参数，顶点 RGB 作为额外颜色调制。

## 10. 光照与颜色

方向光 `direction` 定义为光线传播方向，片元到光源方向为
`L = -normalize(direction)`。每个灯按 Lambert 模型累加：

```text
nDotL = max(0, dot(N, L))
lightSum += light.color * light.intensity * nDotL
```

所有灯累加后逐通道截断到 `[0,1]`，再加入固定 `0.1` 环境光。这里先截断灯光和
再加环境光，使最终线性辐射最多约为 `1.1`，Reinhard 仍能压缩高光：

```text
hdr = albedo * (clamp(lightSum, 0, 1) + 0.1)
```

后处理逐通道执行：

```text
mapped = hdr / (1 + hdr)
srgb   = pow(clamp(mapped, 0, 1), 1 / 2.2)
byte   = floor(srgb * 255 + 0.5)
```

BMP 按收到阶写入 `B,G,R`。文件使用 24 位、无压缩、bottom-up 行序，每行补齐到
4 字节。

## 11. 统计定义

- `totalTriangles`：所有网格源索引可组成的三角形总数，包含被剔除网格。
- `culledMeshes`：AABB 完全位于六视锥平面之外的网格数。
- `rasterizedTriangles`：通过齐次裁剪后实际提交光栅化的扇形三角形数；一个源
  三角形被裁剪后可能贡献多个。
- `totalFragments`：像素中心位于三角形内并通过 top-left 规则的候选片元数。
- `depthTestFailures`：候选片元中深度键小于等于当前深度缓冲的片元数。

这些计数均使用无符号 64 位整数。

## 12. 确定性策略

- 固定像素中心采样和 top-left 填充规则。
- 固定网格、三角形、灯、材质遍历顺序。
- 不使用随机数、时间、线程调度或哈希表迭代决定像素。
- 纹理采样固定最近邻与重复寻址。
- 所有渲染计算双精度，BMP 量化固定四舍五入。
- 画面与统计只由输入场景、分辨率和纹理字节决定。

同一构建和同一平台上的重复运行应当产生逐字节一致的 BMP。

## 13. 已知边界

- 不做背面剔除；双面材质都会参与深度测试。
- 不做抗锯齿、mipmap、各向异性过滤或阴影。
- 纹理值直接作为线性颜色参与着色，不额外执行纹理 sRGB 解码。
- 自相交共面三角形的深度相等时，按输入顺序保留先提交者。
- 极大分辨率的颜色和深度缓冲由内存容量限制；CLI/场景上限为 `16384x16384`。
