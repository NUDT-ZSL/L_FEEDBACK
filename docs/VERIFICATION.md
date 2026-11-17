# 实际验证记录

验证日期：2026-11-17

## 环境

- 运行环境：WSL2 Ubuntu 26.04（同一套可移植 C++17 核心）
- 编译器：GCC 15.2.0
- CMake：3.30.5
- 构建类型：Release
- GPU：未使用
- 网络：未使用
- 额外 Python 包：无（验收脚本只用标准库）

本机 CMake 分发包未包含 `ctest` 可执行文件，因此采用与 CTest 注册项完全相同的命令直接运行验收脚本。

当前执行环境没有 Windows SDK、MSVC 或 Wine，因此本记录中的实际运行验证使用 GCC 构建的可移植核心。Windows 专用的原子替换（`MoveFileExW`）和 Ctrl+C 控制台处理代码已做条件编译审查，但未在本机执行；Windows 构建与验收入口已提供为 `scripts/build_windows.ps1` 和 `scripts/run_acceptance.ps1`。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

结果：`terrain_renderer` 构建成功，GCC `-Wall -Wextra -Wpedantic` 无警告。

额外执行了 ASan + UBSan Debug 构建，分别运行单帧软阴影渲染和完整增量序列，未报告内存越界、未定义行为或泄漏。

## 验收命令

```bash
python3 tests/verify_acceptance.py \
  --exe build/bin/terrain_renderer \
  --work-dir outputs/acceptance_test_final
```

结果：

```json
{
  "checks": 77,
  "incremental_final_max_channel_diff": 0,
  "incremental_full_equivalent_blocks": 1344,
  "incremental_total_redrawn_blocks": 587,
  "ok": true
}
```

额外的最终图组位于 `outputs/validation/angles/` 和 `outputs/validation/incremental/`。

## 三组相机与硬/软阴影

实际耗时（本机 Release）：

- `angles` 完整图组：1.61 秒，峰值 RSS 约 34 MB。
- `incremental` 七帧加最终全量参考：1.56 秒，峰值 RSS 约 59 MB。

各角度均输出 1024×768 RGB PNG。硬阴影和软阴影图逐字节不同；软阴影输出中高对比相邻像素边数低于硬阴影：

| yaw | hard 高对比边数 | soft 高对比边数 |
|---:|---:|---:|
| 0° | 2123 | 1908 |
| 45° | 2034 | 1814 |
| 90° | 2344 | 2064 |

对应图像：

- `outputs/validation/angles/yaw0_hard.png`、`yaw0_soft.png`
- `outputs/validation/angles/yaw45_hard.png`、`yaw45_soft.png`
- `outputs/validation/angles/yaw90_hard.png`、`yaw90_soft.png`

图像检查未发现地形内部背景漏出或三角形断裂。三张图的三角形数均为 130,050。

## 阴影偏置

固定方向光、yaw=45° 的硬阴影结果：

| 偏置 | shadow samples | 背光像素 | 被遮挡采样 |
|---:|---:|---:|---:|
| 2 纹素 | 178,226 | 29,684 | 7,748 |
| 4 纹素 | 178,226 | 29,684 | 5,559 |

4 纹素结果进一步减少自阴影采样，硬阴影主投影保留，未观察到沿屏幕形成的大面积自阴影条纹。背光区域仍保留环境光和冷色补光，不会变成全黑。

文件：

- `outputs/validation/angles/bias2_hard.png`
- `outputs/validation/angles/bias4_hard.png`
- `outputs/validation/angles/bias2_soft.png`
- `outputs/validation/angles/bias4_soft.png`

## 增量重绘

序列：初始帧 → 5 次 0.65 世界单位水平平移 → 1 次大幅回退到 pan=0。

| 帧 | pan_x | 重绘块 | 复用块 |
|---:|---:|---:|---:|
| 0 | 0.00 | 192 | 0 |
| 1 | 0.65 | 66 | 126 |
| 2 | 1.30 | 66 | 126 |
| 3 | 1.95 | 65 | 127 |
| 4 | 2.60 | 65 | 127 |
| 5 | 3.25 | 66 | 126 |
| 6 | 0.00 | 67 | 125 |

- 总重绘块数：587
- 七次全量渲染等价块数：1344
- 节省：56.324%
- 最终修正块数：0
- 最终逐像素最大通道差：0
- 最终不同像素数：0
- 最终与同参数全量渲染完全一致

最终增量图与参考全量图 SHA-256 相同：

```text
20bfe1528e6f864d3155dca09aa1060b7e09323fbfe7611910bb4a74bb45e588  frame_06.png
20bfe1528e6f864d3155dca09aa1060b7e09323fbfe7611910bb4a74bb45e588  reference_full.png
```

## 确定性与报告

同一命令连续运行两次，PNG 与 JSON 均逐字节相同：

```text
41655d0c0808855f293191bb7af6a4443a8cc144e041ee83659cb3b25e83037b  image.png
246f56e5d6fefbf6267b737c58651862aab0f158736f4cfca3b5343fbd737337  report.json
```

报告包含 `render_time_ms`、`triangle_count`、`redrawn_blocks`、`shadow_samples`、`material_warnings`、相机/光源参数和数据哈希。默认 `render_time_ms` 是 `fixed_work_v1` 确定性工作量时间模型；实测墙钟时间通过 `--include-wall-time` 可选输出。

`MATW_ZERO` 实际出现在报告中：

```json
{"code":"MATW_ZERO","zero_texels":36,"sampled_fallbacks":61,"first_uv":[133,120]}
```

## 错误与取消

已实际验证：

| 场景 | 退出码 | JSON 业务码 |
|---|---:|---|
| 未知参数 `--bogus` | 2 | `INVALID_ARGUMENT` |
| `--out` 缺少值 | 2 | `INVALID_ARGUMENT` |
| `--seed nope` | 2 | `INVALID_ARGUMENT` |
| 128×128 输入用于固定 256×256 管线 | 1 | `DIM_MISMATCH` |
| 高度/权重头尺寸不一致 | 1 | `DIM_MISMATCH` |
| `--fov-y 0` | 1 | `DEGENERATE_PROJ` |
| 注入退化投影 | 1 | `DEGENERATE_PROJ` |
| 注入阴影贴图分配失败 | 1 | `SHADOW_ALLOC` |
| `--test-fault cancel` | 1 | `CANCELED` |
| `--cancel-after-triangles 5000` | 1 | `CANCELED` |

取消测试先放置已有的 `preserved.png`，取消前后 SHA-256 不变；取消路径没有写出报告或半成品 PNG。

## 快照

已验证：

- yaw=45°、pan=1.25 的完整渲染写入快照；
- 下一次不传 yaw/pan 启动，恢复后 PNG 与首次渲染逐字节相同；
- 将快照版本改为 99 后启动，进程退出 0，忽略旧快照并使用默认相机；
- 写入损坏 JSON 后启动，进程退出 0，不崩溃，并使用默认相机重新初始化。

## 结论

所有验收要求中的主路径、确定性、增量一致性、快照容错、错误码和取消保护均通过本机实际运行验证。

