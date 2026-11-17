# 离线软光栅地形渲染器

这是一个可在 Windows 10/11 本地离线运行的 C++17 软件光栅地形渲染器。程序不依赖 GPU、网络或外部美术资源，确定性生成 256×256 高度场与草地/岩石/雪三层材质权重图，输出 1024×768 PNG，并附带 JSON 报告、快照恢复、硬/软阴影对比与增量重绘脚本。

## 交付内容

- `src/`：确定性数据生成、4×4 透视投影、齐次裁剪、重心坐标光栅化、深度缓冲、2048² 阴影贴图、3×3 PCF、三层材质混合、64×64 像素块增量重绘、快照和 CLI。
- `third_party/stb_image.h`、`third_party/stb_image_write.h`：本地 STB 单头文件，用于 PNG 编解码。
- `tests/verify_acceptance.py`：不依赖 Pillow/NumPy 的端到端验收脚本，内置 PNG 解码器。
- `scripts/build_windows.ps1`、`scripts/run_acceptance.ps1`：Windows PowerShell 构建与验收入口。
- `docs/DESIGN.md`：渲染管线、偏置语义、增量失效规则和确定性设计。
- `docs/VERIFICATION.md`：本机实际构建与验收结果。

## Windows 构建

环境要求：

- Windows 10/11
- Visual Studio 2019 或 2022，安装“使用 C++ 的桌面开发”工作负载
- CMake 3.20 或以上
- Python 3.8 或以上（仅运行自动化验收脚本时需要）

Visual Studio 2022：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\bin\Release\terrain_renderer.exe --version
```

Visual Studio 2019 将生成器改为 `"Visual Studio 16 2019"`。也可以直接运行：

```powershell
.\scripts\build_windows.ps1
```

多配置生成器的可执行文件位于 `build\bin\Release\terrain_renderer.exe`。

本机 Linux/WSL 验证使用同样的 C++17 代码：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 快速运行

以下命令会确定性生成输入数据并输出完整验收图组：

```powershell
$exe = ".\build\bin\Release\terrain_renderer.exe"

& $exe generate-data --data-dir outputs\data --seed 1337

& $exe angles `
  --data-dir outputs\data `
  --out-dir outputs\angles `
  --snapshot outputs\angles\snapshot.json

& $exe incremental `
  --data-dir outputs\data `
  --out-dir outputs\incremental `
  --yaw 0 `
  --step 0.65 `
  --shadow soft `
  --bias-texels 2 `
  --snapshot outputs\incremental\snapshot.json
```

单帧渲染：

```powershell
& $exe render `
  --data-dir outputs\data `
  --out outputs\single.png `
  --report outputs\single.json `
  --yaw 45 `
  --shadow soft `
  --bias-texels 2
```

按 `Ctrl+C` 可取消渲染。渲染结果先写入同目录临时文件，成功后再原子替换目标 PNG；取消或失败不会留下半张 PNG，也不会覆盖已有结果。

## 主要输出

`angles` 命令输出：

- `yaw0_hard.png`、`yaw0_soft.png`
- `yaw45_hard.png`、`yaw45_soft.png`
- `yaw90_hard.png`、`yaw90_soft.png`
- `bias2_hard.png`、`bias2_soft.png`
- `bias4_hard.png`、`bias4_soft.png`
- 每张图对应的 JSON，以及汇总 `angles_report.json`

`incremental` 命令输出：

- `frame_00.png` 到 `frame_06.png`
- `reference_full.png`：最终相机参数下的独立全量渲染
- 每帧 JSON 和汇总 `incremental_report.json`

## 命令行约定

常用通用参数：

| 参数 | 说明 |
|---|---|
| `--data-dir DIR` | 输入数据目录；文件均不存在时按种子确定性生成 |
| `--seed N` | 数据生成种子，默认 `1337` |
| `--snapshot PATH` | 渲染快照路径，默认 `render_snapshot.json` |
| `--no-restore` | 本次不读取旧快照 |
| `--no-save-snapshot` | 本次不写快照 |
| `--light-x/--light-y/--light-z F` | 覆盖固定方向光的传播方向 |
| `--bias-texels F` | 自阴影偏移，单位为光源阴影贴图纹素 |
| `--shadow-size 2048` | 验收管线固定为 2048×2048 |
| `--test-fault NAME` | 验证用故障注入：`dim_mismatch`、`degenerate_proj`、`shadow_alloc`、`cancel` |
| `--cancel-after-triangles N` | 确定性取消触发点 |
| `--include-wall-time` | 在报告中额外加入实测墙钟时间；此选项会破坏报告逐字节确定性 |

`render` 额外参数：`--out`、`--report`、`--yaw`、`--pan-x`、`--fov-y`、`--near`、`--far`、`--shadow hard|soft`、`--pcf-radius-texels`、`--dump-shadow`。

退出码：

- `0`：成功
- `1`：业务/渲染错误，包括 `DIM_MISMATCH`、`DEGENERATE_PROJ`、`SHADOW_ALLOC`、`CANCELED`
- `2`：命令行参数缺失或非法

错误和结果均输出 JSON。错误对象示例：

```json
{
  "ok": false,
  "error": {
    "code": "DIM_MISMATCH",
    "message": "Expected 256x256 data, received 128x128"
  },
  "exit_code": 1
}
```

## 输入格式

- `heightfield.hf32`：`HF32` magic、uint32 版本、uint32 宽、uint32 高，随后为小端 float32 高度。
- `material_weights.wgt8`：`WGHT` magic、uint32 版本、uint32 宽、uint32 高，随后为逐像素 RGB8 权重，分别对应草地、岩石、雪。

两个文件尺寸必须一致且为 256×256；否则返回 `DIM_MISMATCH`。默认权重数据中包含一块零权重区域，用于验证 `MATW_ZERO` 回退路径。

## 验证

构建后运行：

```bash
ctest --test-dir build --output-on-failure
```

或直接运行：

```bash
python3 tests/verify_acceptance.py \
  --exe build/bin/terrain_renderer \
  --work-dir outputs/acceptance_test
```

验收脚本覆盖 77 项检查，包括三组相机角度、硬/软阴影差异、2/4 纹素偏置、PNG/JSON 确定性、增量重绘总量与最终 ±1 一致性、快照恢复、快照版本/损坏回退、业务错误码、真实尺寸不匹配和取消时的输出保护。实际结果见 `docs/VERIFICATION.md`。

## 确定性说明

PNG 由固定版本的 STB 和固定压缩参数写出。规范 JSON 中的 `render_time_ms` 是按三角形、着色像素、阴影采样和重绘块数量计算的确定性工作量时间模型，字段 `render_time_model` 固定为 `fixed_work_v1`。这样同一命令的 PNG 与 JSON 可以逐字节一致。真实墙钟时间可显式通过 `--include-wall-time` 写入诊断字段，或观察标准错误输出。

