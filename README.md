# Deterministic VPL Global Illumination Previewer

这是一个面向 Windows / OpenGL 3.3 Core Profile 的离线静态场景渲染示例。场景、材质、灯光和固定随机种子全部由仓库脚本确定性生成，不依赖外部资产或下载。渲染器使用“多光源聚类 -> 虚拟点光源（VPL） -> 深度立方体贴图 -> 屏幕空间 HDR 累积 -> ACES/sRGB 后处理”的流程，并支持只更新受影响光源的增量路径。

## 快速运行（Windows）

要求 64 位 Python 3.10 或 3.11。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m proc_gi.generate --output scene.json --summary artifacts/scene_summary.json
python -m proc_gi.app
```

也可以直接运行：

```powershell
.\run.ps1
```

首次运行会生成 `scene.json`。后续启动从该 JSON 加载材质与灯光参数；几何仍由 `scene.json` 中的固定 `seed` 确定性重建。当前验证环境使用 WSLg + Mesa llvmpipe，也支持正常 Windows 显卡驱动。

## 交互

| 按键 | 功能 |
| --- | --- |
| `Tab` | 选择下一盏原灯 |
| 方向键 | 平移选中光源的 X/Z |
| `PageUp` / `PageDown` | 抬高/降低选中光源 |
| `+` / `-` | 调整选中光源强度 |
| `C` | 循环切换光源颜色 |
| `G` | 循环聚类上限与 VPL 渲染预算，包含超限降级场景 |
| `T` | 在 ACES 色调映射与线性调试视图间切换 |
| `S` | 保存 ACES 与线性调试 PNG |
| `R` | 强制全量重累积 |
| `Esc` | 退出 |

灯光变化会在下一次 `render()` 中完成增量重聚类、受影 VPL 立方体重建和 HDR 加减累积，更新延迟不超过一帧。相机移动会触发全量重累积，因为屏幕空间 G-buffer 已变化。

## 管线

1. **确定性场景**：凹多边形房间、地板/天花板、桌子、椅子、沙发、柜子和灯具均为程序化三角形网格。每个顶点含位置、法线、材质索引，固定 `seed=196`。
2. **材质分组**：所有几何装入同一个 VBO/IBO，每个材质有一个连续 `MeshGroup`，记录索引偏移、索引数、顶点偏移和顶点数。渲染时按组调用 `glDrawElements`，组内沿用同一 VAO/VBO/EBO 状态。
3. **材质分支**：`Lambertian`、`Oren-Nayar`、`Blinn-Phong` 在同一个片元着色器中分支。未知类型在配置加载时记录业务码 `MAT_UNKNOWN_TYPE` 并回退为 Lambertian；着色器也有最终 Lambertian 安全分支。
4. **VPL 聚类**：最多读取 256 盏点光/聚光灯，把世界空间位置和朝向组成特征向量，以强度加权确定性 k-means 聚类。输出位置、颜色、总强度、代表方向、方向范围、作用半径和 `source_count`。每帧重聚类时，未受影响簇直接复用；只把改动灯所在簇及新增灯重新聚类。
5. **立方体阴影缓存**：每个 VPL 有内容签名。缓存按签名比对：新增/改动签名创建新立方图，旧图保留到本帧增量减法完成后再释放，未变签名直接复用。立方图存储线性距离/Range，HDR 累积阶段做 5 次 PCF 比较。
6. **屏幕空间累积**：先渲染世界位置、法线+材质和深度 G-buffer。CPU 将 VPL 投影到屏幕空间并写入 16×16 tile 的 VPL 索引表；片元着色器按像素深度/法线与 tile 列表查找 VPL，使用距离平方衰减、平滑 Range 窗、聚光锥和阴影可见度累加漫反射/BRDF 响应。
7. **增量 HDR**：变动的 VPL 在同一 HDR 缓冲上先减去旧贡献、再加新贡献；其他 VPL 贡献从上一帧保留。多个光源重叠时使用物理可加的 BRDF 与能量通量，不做按灯数归一化，因此不会因重叠产生暗斑，也不会重复乘倍增系数。
8. **后处理**：HDR `RGBA16F` 只有一份。ACES + sRGB 输出用于正常画面，线性裁剪输出用于无色调映射调试视图，二者均从同一 HDR 源采样并可保存 PNG。
9. **重要性截断**：当聚类数量超过 `clustering.vpl_capacity` 时，按强度/亮度、视锥包含因子和场景遮挡率计算重要性，保留高重要性 VPL。`artifacts/render_warnings.json` 记录 `VPL_CAPACITY_TRUNCATED`、被截断 VPL ID、对应原灯 ID、评分与 `render_continues=true` 的降级状态。

## 配置

`scene.json` 同时包含：

- `seed`、`scene.room_polygon`、`scene.wall_height`
- `camera`
- `materials`：类型、albedo、roughness、specular、shininess
- `lights`：point/spot、位置、方向、颜色、强度、range、聚光角/软边
- `clustering`：`max_vpls`、`vpl_capacity`、位置/朝向权重、迭代次数、立方图尺寸、tile 尺寸
- `render`：分辨率、曝光、截图与告警路径

`max_vpls` 是聚类输出上限；`vpl_capacity` 是 GPU 渲染预算。二者分开后，可以直接复现“聚类结果超过渲染预算并降级”的验收路径，例如：

```powershell
python -m proc_gi.app --max-vpls 12 --vpl-capacity 8
```

## 验证

```powershell
python -m unittest discover -s tests -v
python -m proc_gi.validate
```

`proc_gi.validate` 会创建隐藏 OpenGL 3.3 Core 窗口并实际执行：

- 确定性网格、材质分组、法线绕序和场景生成重复性检查
- 首帧 VPL 立方图构建
- 单灯移动/强度变化的增量聚类，立方图复用/重建数量检查
- 增量 HDR 与强制全量重建 HDR 的误差比较（同时验证增量阴影一致性）
- 能量线性与重叠可加性
- ACES 与线性调试 PNG 输出
- 超限重要性截断和 JSON 降级告警

## 实际验证结果

本仓库已在本机 Python 3.11.18、Mesa 26.0.8（llvmpipe）、OpenGL 4.5 Core 上下文（代码请求 3.3 Core）下运行。详细机器可读结果见 `artifacts/validation_report.json`，当前结果摘要：

- `17` 个单元/集成测试全部通过。
- 场景：`1560` 顶点、`520` 三角形、`5` 个材质组、`24` 盏原灯、`16` 个 VPL。
- 单灯强度与位置改变：`cube_builds=1`、`cube_reused=15`、`delta_entries=2`（1 减 1 加），未全量重建。
- 增量 HDR 对比全量 HDR：最大绝对误差小于 `0.08`，实测 `0.0078125`；阴影一致性通过该项比较验证。
- HDR 缓冲无 NaN/Inf；ACES 与线性调试 PNG 内容不同。
- `max_vpls=12`、`vpl_capacity=8` 时继续渲染，接受 8 个 VPL，截断 4 个，并写出包含原灯 ID 的告警 JSON。
- 产物：`artifacts/frame_tonemapped.png`、`artifacts/frame_linear.png`、`artifacts/frame_after_incremental.png`、`artifacts/frame_degraded.png`、`artifacts/render_warnings.json`、`artifacts/vpl_clusters.json`。

## 业务码

| 业务码 | 行为 |
| --- | --- |
| `MAT_UNKNOWN_TYPE` | 未知材质回退 Lambertian，记录材质 ID/名称/原类型 |
| `LIGHT_LIMIT_TRUNCATED` | 超过 256 盏灯时按重要性截断并记录灯 ID |
| `VPL_CAPACITY_TRUNCATED` | 超过 VPL 渲染预算时按重要性截断，渲染继续 |

## 仓库结构

```text
proc_gi/
  scene_generator.py   确定性房间/家具/灯光配置
  geometry.py          单 VBO/IBO、材质索引分组和绕序修正
  clustering.py        位置+朝向聚类、VPL 签名、重要性截断
  cubemap_cache.py     立方图签名版本缓存与延迟释放
  lighting.py          CPU 光照参考与能量测试
  gl_resources.py      VAO/VBO/IBO、FBO、HDR、立方图、着色器封装
  renderer.py          G-buffer、tile 累积、增量更新、后处理
  app.py               Windows/OpenGL 交互入口
  validate.py          实际 OpenGL 验收与报告生成
  shaders/             3.30 版本顶点/片元着色器
tests/                 unittest 单元与 OpenGL 集成测试
scene.json             固定种子生成并随仓库提供的配置
```

## 已知边界

- 这是 VPL 近似的漫反射全局光照，不是路径追踪；镜面只使用 Blinn-Phong 局部响应，没有屏幕空间反射。
- 深度立方图使用线性距离颜色纹理 + 硬件深度附件，不依赖 GL 4.x 的立方图数组。
- 单次交互通过隐藏窗口验证；真实 Windows 窗口渲染需要桌面显示会话和现代显卡驱动。
