# 体积雾离屏渲染器（volfog）

纯 CPU 的体积雾离屏渲染器：确定性程序化噪声生成三维密度场，单点光源光线步进做
散射积分，输出颜色、Beer-Lambert 透射率与深度三个通道，并保存为 PNG/NumPy 文件。

## 环境

- Python 3.9+（开发验证使用 3.12，系统自带 3.14 无 numba 支持时用 uv 安装 3.12）
- 依赖：`numpy`、`numba`、`pillow`（见 `requirements.txt`）

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 运行

```bash
.venv/bin/python -m volfog --mode single                       # 串行单次散射
.venv/bin/python -m volfog --mode multiple --parallel --threads 4
.venv/bin/python -m volfog --density-file field.npy            # 外部密度场
```

主要参数（均有默认值）：`--width/--height/--voxel-res`、`--step-size`、
`--max-samples`、`--terminate-threshold`、`--depth-threshold`、`--extinction`、
`--albedo`、`--phase-g`、`--light-samples`、`--multiple-strength`、相机与光源参数。

## 模块

- `volfog/density.py`：种子化置换表驱动的 3D 值噪声 + 4 倍频 FBM，min-max 归一化到
  [0,1]；外部 `.npy` 经同一套校验与归一化。
- `volfog/renderer.py`：Numba `njit` 内核。体素立方覆盖世界坐标 `[-1,1]^3`，
  三线性插值采样；串行与并行由同一份循环体以 `parallel=False/True` 编译两次。
- `volfog/pipeline.py`：扩展 Reinhard 色调映射 → 线性转 sRGB → 8 位量化，文件输出。
- `volfog/cli.py`：命令行与统一 JSON 错误。
- `tests/`、`scripts/acceptance.py`：单元测试与端到端验收。

## 数学约定

- 光线步进：样本位于每段中点 `t=(i+0.5)*step_size`，相机 → 最大距离
  `max_distance = step_size * max_samples` 方向累积。
- 透射率（Beer-Lambert）：`T *= exp(-sigma_t * density * step_size)`，与颜色积分在
  同一循环、同样本、同提前终止（`terminate_threshold`）下完成，逐样本对齐。
- 单次散射颜色：`T * (1-exp(-tau)) * albedo * L_light * phase_HG`，其中
  `L_light = intensity / dist^2 * T_shadow`，阴影段以固定 `--light-samples` 步进取
  Beer-Lambert 透射率，相位函数为 Henyey-Greenstein。
- 多次散射近似：在单次散射能量之外，对 3x3x3 邻域平均密度（代表经历至少一次额外
  散射事件后到达的各向同性再辐射能量）追加一项累积；透射率与深度规则不变。
- 深度：记录透射率首次低于 `--depth-threshold`（默认 0.01）时的行进距离；未命中写
  `max_distance`。16 位 PNG 的归一化基准为
  `depth / (step_size * max_samples)`（即配置的最大光线距离，见每个 `*_meta.json`）。
- 颜色管线：线性 RGB 积分 → 扩展 Reinhard（白点取帧内最大亮度）→ sRGB 传递函数 →
  8 位。颜色（RGB PNG）与透射率（灰度 PNG）文件名均带模式段，如
  `fog_single_color.png`、`fog_multiple_transmittance.png`；同时输出同名 `.npy`
  原始浮点数据供合成与精确比对。

## 输入校验

所有参数在渲染前统一校验。NaN/无穷大、负焦距、非法分辨率、非法密度场等均中止渲染，
stderr 输出单行 JSON，退出码 1：

```json
{"error_code":"E_INVALID_INPUT","param":"focal_length","message":"value must be positive"}
```

## 验证

```bash
.venv/bin/python -m unittest tests.test_renderer -v   # 8 个单元测试
.venv/bin/python scripts/acceptance.py                # 端到端验收
```

实测结果见 `VALIDATION.md`。
