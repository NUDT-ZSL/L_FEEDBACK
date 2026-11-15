# 验证记录

环境：Ubuntu 26.04，CPython 3.12.16（uv 安装；系统 Python 3.14 暂无 numba wheel），
numpy 2.5.4、numba 0.68.1、pillow 13.0.0，纯 CPU。

## 单元测试

命令：`.venv/bin/python -m unittest tests.test_renderer -v`

结果：8/8 通过，覆盖

- 程序化密度场确定性（同种子逐元素相等、异种子不同）与 [0,1] 归一化；
- 外部 `.npy` 导入归一化与 NaN 校验；
- 串行/并行三通道逐像素一致（容差 1e-6）；
- single/multiple 透射率与深度完全相同、颜色存在差异且 multiple 不小于 single；
- 透射率取值范围、深度命中/未命中规则；
- PNG 模式（RGB / L / 16 位灰度 uint16）与原始 `.npy` 落盘；
- 六类非法 CLI 输入的单行 JSON 错误与退出码 1；缺失密度文件错误。

## 端到端验收

命令：`.venv/bin/python scripts/acceptance.py`（默认 128x128、64^3 体素）

实测输出：

```
== serial vs parallel max abs diff (tolerance 1e-6) ==
[PASS] single   color_linear   diff=0.000e+00
[PASS] single   transmittance  diff=0.000e+00
[PASS] single   depth          diff=0.000e+00
[PASS] multiple color_linear   diff=0.000e+00
[PASS] multiple transmittance  diff=0.000e+00
[PASS] multiple depth          diff=0.000e+00

== mode comparison ==
color max delta        : 0.540968 (expected > 1e-6)
transmittance max delta: 0.000e+00 (expected 0)
depth max delta        : 0.000e+00 (expected 0)
```

串并行结果逐位相等（Numba 内核每像素独立、无规约顺序差异），优于 1e-6 要求。

非法输入（均为退出码 1、stderr 单行 JSON、`error_code=E_INVALID_INPUT`）：

```
--focal-length -1   -> param=focal_length
--focal-length nan  -> param=focal_length
--step-size inf     -> param=step_size
--width 0           -> param=width
--voxel-res 1       -> param=voxel_res
--light-color 1 1 nan -> param=light_color
```

另手工验证：外部密度场含 NaN / 负值 / 非 3D、`--threads -1`、非法 `--mode` 均返回
同样结构的 JSON 与退出码 1。

## 性能（默认配置，128x128，编译后重复计时）

- 串行：约 288 ms/帧
- prange 4 线程：约 101 ms/帧，加速比约 2.84x

## 输出文件

每次渲染在输出目录生成（`{prefix}_{mode}_` 前缀，模式写入文件名）：

- `*_color.png`：sRGB 8 位 RGB（线性积分 → 扩展 Reinhard → sRGB → 量化）
- `*_transmittance.png`：8 位灰度透射率
- `*_depth.png`：16 位灰度深度，归一化基准 `depth/(step_size*max_samples)`
- `*_color_linear.npy`、`*_transmittance.npy`、`*_depth.npy`：浮点原始数据
- `*_meta.json`：含 `max_distance` 与深度归一化说明
