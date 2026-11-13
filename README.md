# ruin-bake

面向 Windows / Python 3.11 的离线光照探针烘焙工具。场景、光源、探针依赖和
快照写入均由项目代码确定性构造，不依赖 GPU、渲染 SDK、网络或外部数据集。

## 环境

已验证环境：

- Python 3.11.18
- NumPy 1.26.4
- Windows 目标代码路径使用 `spawn` 多进程；本次实际运行在 WSL2 Linux 上

Windows PowerShell 从零安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e . --no-build-isolation
```

Linux/macOS：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-build-isolation
```

未安装包时也可以从源码目录运行：

```bash
PYTHONPATH=src .venv/bin/python -m ruin_bake --help
```

## 快速运行

全量烘焙：

```bash
.venv/bin/ruin-bake full --config configs/demo.json --out artifacts/demo.snapshot
```

局部增量烘焙：

```bash
.venv/bin/ruin-bake incremental \
  --config configs/demo_patch.json \
  --base artifacts/demo.snapshot \
  --out artifacts/demo_incremental.snapshot
```

查看快照摘要：

```bash
.venv/bin/ruin-bake summary --snapshot artifacts/demo_incremental.snapshot
```

三个命令均向 stdout 输出单行 JSON。成功时 `ok` 为 `true`；参数错误、损坏快照
或 I/O 错误时返回非零退出码，并向 stderr 输出包含 `ok: false` 与 `error` 的
JSON。

## 命令行参数

常用覆盖参数：

- `--seed`：程序化场景种子
- `--patch`：JSON 场景补丁
- `--spacing`：探针网格间距
- `--cull-distance`：探针到几何体表面的最小距离
- `--samples`：每探针余弦加权半球采样数
- `--ray-epsilon` / `--integration-tolerance`：射线自交与数值容差
- `--workers`：工作进程数；`0` 表示最多 4 个
- `--chunk-size`：每个并行工作块的探针数

例如：

```bash
.venv/bin/ruin-bake full --seed 20261114 --samples 64 --spacing 2 --workers 4 \
  --out artifacts/seed.snapshot
```

## 场景补丁

补丁是相对确定性地基场景的 JSON 对象：

```json
{
  "geometry_updates": {
    "3": {"albedo": [0.88, 0.16, 0.08]}
  },
  "geometry_remove": [7],
  "geometry_add": [
    {
      "center": [0.0, 0.0, 1.0],
      "half_size": [0.5, 0.5, 1.0],
      "rotation_deg": [0.0, 0.0, 25.0],
      "albedo": [0.4, 0.3, 0.2],
      "emission": [0.0, 0.0, 0.0]
    }
  ],
  "directional_update": {"radiance": [2.2, 2.0, 1.7]},
  "point_light_updates": {
    "point-0": {"position": [1.0, -1.0, 3.0]}
  }
}
```

## 实现摘要

- 场景生成器使用固定 `random.Random(seed)` 操作序列；JSON 采用排序键与可
  逐位回读的浮点表示。
- 探针在包围盒内按 `z -> y -> x` 的稳定顺序枚举，剔除内部或近表面探针后
  赋予连续 ID；并行度不影响 ID。
- 直达项由方向光与点光解析计算；间接项使用固定数量的
  Hammersley/Cranley-Patterson 余弦半球样本和一次漫反射弹射，自发光几何
  作为命中表面的辐亮度来源。
- 每个探针是纯函数；工作块可以任意切分、乱序完成，父进程按稳定探针 ID
  散播结果后按升序归约，不做跨探针浮点累加。
- 增量模式比较几何/光源内容哈希。仅材质变化只重算采样射线实际命中该物体
  的探针；形状变化额外检查遮蔽关系；光源变化逐灯比较解析贡献；几何与光源
  同时变化时保守地全量重算。合并值直接使用新布局中相同位置/法线探针的
  float64 位模式。
- 快照采用版本化二进制容器、JSON 头、固定 80 字节探针记录和 SHA-256
  完整性校验。写入先落到同目录临时文件，`fsync` 后 `os.replace` 原子替换，
  再同步目录。

## 测试

```bash
TMPDIR="$PWD/.tmp" PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -v
```

测试包含真实子进程 SIGKILL 场景：子进程在临时快照只写了一部分时被杀，
父进程验证旧快照仍可读取，随后重试并验证增量结果与全量结果逐位一致。

更详细的设计、格式说明和实际验证记录见 [REPORT.md](REPORT.md)。
