# 实际验证记录

验证日期：2026-11-16  
工作目录：`/home/liang/feedback/uid121`

## 环境

| 项目 | 实测值 |
|---|---|
| Python | 3.11.18，64 位 |
| NumPy | 2.4.6 |
| Pillow | 13.0.0 |
| 逻辑核心 | 24 |
| 渲染后端 | 纯 CPU，无 GPU/显示依赖 |

Python 3.11 由本机 `uv` 工具链安装，项目虚拟环境为 `.venv`，`pip` 已验证可用。

## 复现命令

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify.py
```

测试结果：

```text
Ran 18 tests in 1.740s
OK
```

## 验收结果

| 验收项 | 实测 | 要求 | 结果 |
|---|---:|---:|---|
| ACES/sRGB 独立参考最大误差 | 0 量化单位 | ≤ 2 | 通过 |
| 1/2/4 线程最大逐像素差 | 0.0 | 完全一致 | 通过 |
| 同参数重复运行最大差 | 0.0 | < 1e-6 | 通过 |
| 多线程报告一致性 | 完全相等 | 无重复、无遗漏、单线程一致 | 通过 |
| SSAO 开/关线性图像 MAD | 0.0244164925 | > 0.02 | 通过 |
| 报告相对 HDR 掩码统计误差 | 0.0 | < 1% | 通过 |
| 报告 HSV 色相误差 | 0.0° | 与 HDR 采样一致 | 通过 |
| 非法场景退出码 | 0 | 0 | 通过 |
| 非法场景 stderr | 含 `E_INVALID_SCENE` | 含业务错误码 | 通过 |
| 非法场景输出目录 | 不存在 | 不渲染图像 | 通过 |
| 降级 stderr | 含 `E_RESOURCE_LIMIT` | 含资源警告 | 通过 |
| 降级后进程退出码 | 0 | 继续完成 | 通过 |

机器可读结果：`verification/results.json`。

## 关键实测数据

### 色彩

独立标量参考覆盖 39 个 HDR 通道值，包括 0、极小值、sRGB 分段点附近、1、100 和 10000。生产向量实现与参考实现的最大 8 位量化差为 0。

### 多线程与重复性

验证分辨率为 96x72、2 spp、tile 16：

```text
max |2 threads - 1 thread| = 0.0
max |4 threads - 1 thread| = 0.0
max |repeat run - reference| = 0.0
reports_equal = true
```

### SSAO

正式对比使用 384x256、2 spp：

```text
mean absolute linear RGB difference = 0.024416492466163692
required minimum                    = 0.02
```

对比场景故意使用较高的环境照明，以确保全图线性平均绝对差稳定超过阈值。ACES 预览中高光区域接近白色，但线性 HDR 中的遮蔽差异和统计值仍然完整保留。

### 材质阵列

正式材质阵列：

- 分辨率：640x480
- 采样：8 spp
- 线程：4
- 球体：12（4 个粗糙度级别 × 3 个金属度级别）
- 输出 HDR 全部有限
- HDR 全局范围：`0.0080000004` 到 `609.0475464`
- 12 个区域和 12 个掩码均有对应条目

预览文件：`verification/material_grid/preview_aces_srgb.png`。

### 资源降级

命令使用 0.5 MiB 内存上限：

```text
requested: 96x72, 8 spp
estimated: 2,139,136 B
effective: 54x40, 1 spp
estimated: 350,656 B
degraded:  true
exit code: 0
```

日志逐步记录了 8→4→2→1 spp，以及 96x72→72x54→54x40 的决策。降级输出仍包含 HDR、PNG、日志、12 项材质报告和掩码。

## 工件

```text
verification/results.json
verification/material_grid/
verification/ssao_on/
verification/ssao_off/
verification/resource_fallback/
```

每个渲染目录均包含：

```text
linear_hdr.npy
preview_aces_srgb.png
render_log.json
material_report.json
sphere_masks.npz
```

