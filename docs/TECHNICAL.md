# 技术说明与验证记录

## 1. 数据流

```
YAML 解析(material.py)
  -> 贴图加载/UV 采样(sampler.py, image_io.py)
  -> 权重归一化(blend.py，仅一次)
  -> 法线合成 + 重归一化(blend.py + deterministic.py)
  -> 基础色/粗糙度合并(blend.py)
  -> 切线基重建与连续性修正(basis.py)
  -> GGX 方向光渲染(shading.py)
  -> 原子落盘：法线 PNG / 反射 NPY+PNG / 材质缓存 MZC / result.json(pipeline.py)
```

## 2. 关键设计

### 2.1 权重单一事实来源

`normalized_weights()` 对标量层权重做一次归一化并校验总权重为正。同一数组被
法线合成、标量场合并、反射渲染和缓存 `layer_contributions[].normalized_weight`
共用，杜绝三处权重漂移。重复层按独立索引处理，允许同名层（见
`material_weighted.yaml` 的 `detail` / `detail_duplicate`）。

### 2.2 OpenGL(+Y) 法线解码与重归一化

PNG 的 RGB 按 `x=2r-1, y=2g-1, z=2b-1` 线性解码到切线空间向量（绿色为 +Y）。
合成末尾统一重归一化；平方项归约顺序由阶段 RNG 的置换决定
（`DeterministicContext.seeded_normalize`），避免不同 BLAS 归约顺序造成的 ULP 漂移。

### 2.3 混合模式语义

- `overlay`：法线仅混入切向细节 `(nx,ny,0)`，颜色用通道叠加公式
  （`2ab` 变暗 / `1-2(1-a)(1-b)` 变亮）。
- `mix`：按归一化权重线性插值。
- `weighted`：全加权场景走全局加权快速路径，法线只重归一化一次，结果与层顺序无关。

### 2.4 切线基重建与不连续修正

初始切线取法线切向投影并对法线正交化。光栅扫描（先行后列、左到右）时取左邻或
上邻为参考，比较两者在当前法线平面上投影的夹角；超过 `tangent_threshold_deg`
则翻转切线符号并重新正交化，副切线按 `B=cross(N,T)` 重建，保持右手系。
修正只作用于切线/副切线，法线数组不被写回，配合单位长度断言保证“法线长度与方向不变”。
修正标记 `correction_marker` 与阈值、计数、比例、最大夹角一并写入缓存。

### 2.5 GGX 反射

采用 Trowbridge-Reitz NDF、Schlick-GGX Smith 几何项、Schlick Fresnel（F0=0.04）
与 Lambert 漫反射互补。所有方向（光、视线、半角向量）进入计算前单位化，半角向量
归一化同样走种子控制的归约。每盏灯输出线性 HDR `float32` NPY 与 tonemap 预览 PNG。

### 2.6 确定性

- 种子经 `SeedSequence` 按阶段派生 RNG；批量子种子按输入位置 `spawn`，与并发数无关。
- 关键浮点归约（法线/半角归一化、加权求和）使用种子置换后的固定累加顺序。
- PNG 固定 `compresslevel=6`、不写时间戳；缓存 ZIP 条目固定时间戳 1980-01-01、
  固定压缩级别、条目名排序；JSON 排序键 + 固定分隔符；NPY 本身无时间戳。

### 2.7 失败原子性与错误码一致性

每个材质先在输出根下的隐藏临时目录构建，全部成功后 `rename` 替换目标目录；
任何异常都清理临时目录、目标目录与空输出根。错误码与退出码在 `errors.py` 单点定义，
单文件 CLI 与批量 worker 引用同一份映射，批量失败项进入汇总但不产生半成品文件。

## 3. 实际验证结果

环境：Python 3.14.4，numpy 2.5.4，pyyaml 6.0.3，pillow 13.0.0（Linux 容器；
代码仅使用跨平台标准库与三方库，Windows 10/11 + Python 3.10+ 同样适用）。

### 3.1 测试套件

`python -m unittest tests.test_pipeline -v`：15 个用例全部通过，覆盖
合成输出、单位法线、缓存权重一致性、切线正交/右手系、接缝修正、GGX 输出、
四类错误码、批量顺序与隔离、种子字节确定性。

### 3.2 字节级确定性

同一 `material_mixed.yaml` 以 `--seed 20260101` 渲染两次，`diff -r` 输出为空，
法线 PNG、三张反射 NPY/PNG、`material_cache.mzc`、`result.json` 逐字节一致。
批量同种子两次运行，23 个数据文件 SHA-256 全部相同；汇总 JSON 仅 `output_dir`
绝对路径随输出根变化（剔除后完全一致）。

### 3.3 错误码与退出码

| 样本 | 错误码 | 退出码 |
| --- | --- | --- |
| `case_bad_mode.yaml` | `E_MIX_MODE` | 4 |
| `case_uv_oob.yaml` | `E_UV_OUT_OF_BOUNDS` | 5 |
| `case_missing_texture.yaml` | `E_INPUT_MISSING` | 2 |
| `case_invalid_layer.yaml` | `E_LAYER_INVALID` | 3 |

stderr 为单行结构化 JSON（含 `error_code/exit_code/condition/message/details`），
失败后无任何输出目录残留。

### 3.4 批量汇总

5 项输入（2 成功 3 失败）、`-j 2`：汇总按输入顺序记录
`OK / E_MIX_MODE / OK / E_UV_OUT_OF_BOUNDS / E_INPUT_MISSING`，
成功项退出码 0、失败项为对应业务码；整体退出码 1（部分失败），失败项目录不存在。

### 3.5 切线修正端到端

`material_seam.yaml`（阈值 5 度，含垂直接缝法线）：缓存记录
`correction_count=2048`、`correction_ratio=0.5`、`max_adjacent_angle_deg=180`，
标记数组有 2048 个修正像素，法线保持单位长度且与切线正交、副切线满足右手系。

## 4. 复现实验命令

```bash
python scripts/make_samples.py -o samples
python -m unittest discover -s tests -v
python -m matrelight render samples/material_mixed.yaml -o run_a --seed 20260101
python -m matrelight render samples/material_mixed.yaml -o run_b --seed 20260101
diff -r run_a run_b
python -m matrelight batch samples/material_mixed.yaml samples/material_weighted.yaml \
    samples/material_seam.yaml samples/case_bad_mode.yaml -o batch_out -j 2 --seed 20260101
```
