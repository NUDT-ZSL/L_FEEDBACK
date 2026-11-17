# 场景 JSON 格式

所有数值字段必须是有限浮点数。整数字段也以 JSON number 表示，但必须是无小数部分的整数。未知字段会被忽略；重复键、尾随逗号、非标准 JSON 字面量会被拒绝。

## 顶层

```json
{
  "version": 1,
  "image": {},
  "camera": {},
  "ambientColor": [0.03, 0.04, 0.05],
  "background": [0.005, 0.008, 0.015],
  "directionalLight": {},
  "pointLights": [],
  "materials": [],
  "meshes": [],
  "objects": [],
  "transparency": {},
  "limits": {}
}
```

- `version`：必填，目前只能为 `1`。
- `ambientColor`：必填，RGB 线性颜色，每通道 `[0,1]`。
- `background`：可选，RGB 线性背景，每通道 `[0,1]`；默认 `[0.01,0.015,0.025]`。
- `limits`：可选；`maxMemoryBytes` 为可选整数，默认 1 GiB，最小 1024。

## image

```json
{
  "width": 320,
  "height": 320,
  "viewport": {"x": 0, "y": 0, "width": 320, "height": 320}
}
```

输出图像最大边长 8192。视口必须完整位于图像内。投影宽高比使用 `viewport.width / viewport.height`，视口外像素使用背景色。

## camera

```json
{
  "position": [3.15, 2.25, 4.85],
  "target": [0.0, 1.0, 0.0],
  "up": [0.0, 1.0, 0.0],
  "fovYDegrees": 43.0,
  "near": 0.1,
  "far": 40.0
}
```

相机使用右手视图矩阵，`near > 0` 且 `far > near`。`position`、`target`、`up` 必须形成有效观察基。

## directionalLight

```json
{
  "direction": [0.42, 0.88, 0.55],
  "color": [1.0, 0.88, 0.70],
  "intensity": 1.7,
  "shadow": {
    "enabled": true,
    "mapSize": 512,
    "depthBias": 0.0015,
    "normalBias": 0.035,
    "pcfRadius": 1
  }
}
```

- `direction`：从表面指向光源，内部单位化。
- `mapSize`：整数，范围 `[16, 8192]`；大于 8192 返回错误码 8。
- `depthBias`：范围 `[-0.1, 0.1]`。
- `normalBias`：世界空间法线偏移，范围 `[0, 10]`。
- `pcfRadius`：`[0, 4]`，实际采样数为 `(2r+1)^2`。

阴影相机由全部实例的世界包围盒自动构造，覆盖整个场景。

## pointLights

最多 8 个：

```json
[
  {
    "position": [-1.8, 1.65, 1.6],
    "color": [0.15, 0.8, 1.0],
    "intensity": 10.0
  }
]
```

点光源强度按 `1 / max(distance^2, 1e-4)` 衰减，`distance` 是点光源位置到当前片元世界坐标的欧氏距离。

## materials

```json
[
  {
    "name": "spectral_aura",
    "baseColor": [0.03, 0.62, 0.88],
    "specularColor": [0.40, 0.95, 1.0],
    "shininess": 96.0,
    "opacity": 0.34,
    "normalMap": {
      "width": 1,
      "height": 1,
      "strength": 1.0,
      "data": [0.0, 0.0, 1.0]
    }
  }
]
```

- `baseColor`、`specularColor`：每通道 `[0,1]`。
- `shininess`：必须大于 0。
- `opacity`：`[0,1]`；`< 1` 定义为半透明。
- `normalMap`：可选。`width`、`height` 为 `[1,2048]`，`strength` 为 `[0,4]`，`data` 是扁平数组，长度必须为 `width * height * 3`。

法线贴图值按 `x,y,z` 元组存放在切线空间。采样顺序为双线性、wrap、strength 缩放、归一化。若插值 TBN 或采样结果退化，则确定性回退到几何法线/稳定切线基。

## meshes

```json
[
  {
    "name": "head_sphere",
    "vertices": [
      {
        "position": [0.0, 0.42, 0.0],
        "normal": [0.0, 1.0, 0.0],
        "uv": [0.5, 0.0]
      }
    ],
    "indices": [0, 1, 2]
  }
]
```

- 每个 mesh 最多 2,000,000 个顶点、6,000,000 个索引。
- 索引数必须非零且是 3 的倍数；所有索引必须小于该 mesh 顶点数，否则错误码 7。
- 世界空间逆时针绕序为正面；视口 Y 向下，因此屏幕空间负面积栅格化。
- 顶点法线允许非单位长度，内部归一化；零法线使用相邻面法线累加回退；全部退化时使用 `+Y`。
- 每三角形用 UV 微分计算 tangent/bitangent，累加到三个顶点；UV 行列式退化的三角形不贡献切线。最终缺失切线的顶点使用基于法线的稳定基。

## objects

```json
[
  {
    "name": "head",
    "mesh": 1,
    "material": 1,
    "model": [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 1.71, 0.02, 1
    ]
  }
]
```

- `mesh`、`material` 是数组索引；越界错误码 7。
- `model` 为 16 个行主序浮点数，按列向量约定执行 `M * p`。
- 模型矩阵必须可逆；法线使用逆转置矩阵变换。

## transparency

```json
{"depthWriteMaxOpacity": 0.5}
```

透明三角形的质心深度按以下规则检查与排序：

1. 计算相机空间质心 Z。
2. 若任意两个透明三角形的 Z 的 float 位模式相同，错误码 9。
3. 按 Z 升序（远到近）稳定排序。
4. 不透明 pass 完成后绘制透明 pass。
5. 透明片元先做 Z 测试；仅当 `opacity < depthWriteMaxOpacity` 时更新深度。

该判断按任务要求实现为“小于阈值时写”，不是常见的 alpha cutoff 语义。
