# 离线地形资产生成流程（terrain_gen）

从 YAML 配置确定性生成程序化地形，经多遍水力/热力腐蚀后输出实时渲染管线可用的
16 位高程图、OpenGL 切线空间法线贴图、16 位材质索引图和 JSON 元数据。所有输入
均由代码内的确定性噪声构造，不依赖任何外部数据或网络。

## 环境要求

- Python 3.10+（开发验证使用 Python 3.14）
- numpy >= 1.24、PyYAML >= 6.0、Pillow >= 9.0（测试另需 pytest）

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python -m terrain_gen examples/config_valid.yaml     # 单次生成，退出码 0
python -m terrain_gen examples/config_invalid.yaml   # 非法配置，stderr 输出 ERR_CFG，退出码 2
python -m terrain_gen examples/config_sweep.yaml     # 参数扫描，单项失败隔离，全部成功才返回 0
```

输出文件由 `output_prefix` 决定，缺失目录自动创建：

- `<prefix>_height.png`：16 位灰度高程（I;16，量化自 `[0,1]` 浮点高度场）
- `<prefix>_normal.png`：8 位 RGB 法线（OpenGL 编码，R=X、G=Y、B=Z，`[-1,1] -> [0,255]`）
- `<prefix>_material.png`：16 位材质索引（0 草地、1 岩石、2 沙地、3 雪地）
- `<prefix>_metadata.json`：输入配置、收敛诊断、高度统计、材质像素占比

参数扫描额外写 `<prefix>_sweep_summary.json`，扫描产物按
`<prefix>_it<迭代次数>_*` 命名。

## 配置项

| 键 | 范围/类型 | 说明 |
| --- | --- | --- |
| `width` / `height` | 整数 256–2048 | 地形尺寸 |
| `iterations` | 整数 1–50 | 腐蚀迭代次数 |
| `noise.seed` | 0–2^32-1 | PCG64 随机种子，决定分形噪声 |
| `noise.base_frequency` / `octaves` / `lacunarity` / `persistence` | 数值 | fBm 分形噪声参数 |
| `erosion.hydraulic_weight` / `thermal_weight` | `[0,1]` | 两类腐蚀权重，为 0 时跳过对应子流程 |
| `erosion.*` | 数值 | 降雨、管流系数、可蚀性、沉积、蒸发、坡度阈值 talus、热力搬运系数 |
| `sweep_iterations` | 1–50 的整数列表 | 可选；按升序逐项执行扫描 |
| `fault_at` | 整数（测试钩子） | 可选；在指定 1-based 迭代轮次抛出 `RuntimeError`，用于验证异常隔离 |

任何非法配置都会在 stderr 输出形如
`{"error_code": "ERR_CFG", "message": "..."}` 的 JSON 并以退出码 2 结束。

## 模块设计

- `terrain_gen/noise.py`：周期性 value-noise 晶格 + fBm 叠加，晶格在两个方向环绕，
  高度场天然可平铺；min-max 归一化到 `[0,1]`。随机源统一为
  `numpy.random.default_rng(seed)`（PCG64），同配置逐字节可复现。
- `terrain_gen/erosion.py`：
  - 水力子步骤：降雨 → 管流模型四向流量累积（出流按可用水量钳制）→ 水量平衡 →
    流速/坡度 → 输沙容量驱动的侵蚀与沉积 → 蒸发。每步地形更新后钳制负值，
    侵蚀量不超过单元格现有高度，杜绝负高度与显式格式振荡。
  - 热力子步骤：对超过 talus 坡度阈值的邻居按可接收量比例分配搬运量，所有
    移除/沉积累加到统一 delta 场，严格守恒（实测相对漂移约 1e-16）。
  - 全部使用 float64 与周期性 `np.roll` 边界；每轮记录 `max_erosion` 等收敛诊断。
- `terrain_gen/normals.py`：周期边界 Sobel 梯度构造单位法线，
  `N = normalize(-dz/dx, -dz/dy, +1)`，再量化到 8 位 RGB。
- `terrain_gen/materials.py`：高度 + Sobel 坡度联合分类（沙地低海拔、岩石陡坡、
  雪地高海拔平缓区、其余草地），输出各材质像素计数与占比。
- `terrain_gen/pipeline.py`：单次生成与扫描编排；扫描中单项异常被捕获，
  记录 `error_type` 后继续，汇总 `exit_code`（全成功 0，存在失败 1）。
- `terrain_gen/cli.py`：命令行入口与退出码策略。

## 验证结果

执行 `python -m pytest -q`：**48 个测试全部通过**，覆盖配置校验、噪声确定性与
可平铺接缝、水力非负/稳定、热力守恒、法线角度、材质统计、端到端 PNG/JSON、
扫描异常隔离与 CLI 退出码。

本地 256×256、8 次迭代的确定性配置实测：

- 同配置重复运行三张 PNG 逐字节一致（SHA-256 相同）；换种子高度图必然变化。
- 法线与 Sobel 梯度参考方向最大角度误差 **0.32°**（要求 < 2°，含 8 位量化）。
- 热力 50 轮、权重 1.0 压力测试相对总高度漂移约 **2e-16**（要求 <= 1e-6）。
- 材质占比 `grass 0.599 / rock 0.197 / sand 0.142 / snow 0.061`，占比和误差 0
  （要求 <= 0.0001），材质 PNG 含全部 4 个索引。
- 边界接缝处高度与法线连续；非法配置退出码 2 且 stderr 含 `ERR_CFG`；
  扫描 `[2,4,8]` + `fault_at=4` 时 2 成功、4/8 失败并隔离，汇总退出码 1。
- 2048×2048、3 次迭代端到端约 5.4 秒（参考机器）。
