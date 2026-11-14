# 体素化 SH-9 辐照度缓存烘焙器

这是一个只依赖 Python 标准库与 NumPy 的确定性离线烘焙管线。输入是由代码生成的
三角形场景 JSON；输出是带版本、哈希、float32 球谐系数和逐探针残差的二进制缓存。

## 已实现能力

- 自适应包围盒的三角形体素化，使用三角形/AABB SAT 相交测试。
- 稀疏空间哈希索引、3D DDA 射线遍历、首次三角形命中和遮挡射线。
- 指定间距的探针布局，以及低差异、单位半球、均匀立体角方向采样。
- 每个探针 9 个 SH 系数 × RGB 三通道；系数以 float32 存储，另存相对 RMS 残差。
- `ThreadPoolExecutor` 动态工作队列；单线程与多线程结果逐位一致。
- 面积光解析权重采样、点光支持、朗伯材质、环境光漏射贡献。
- 逐探针能量上界钳制，上界为光源总辐射功率 × 场景反照率上界。
- 几何/光照/布局/场景哈希校验，版本拒绝，严格载荷长度与 SHA-256 校验。
- 原子写入缓存和报告；失败时不会破坏已有缓存。
- `--validate` 仅读取并校验已有缓存，不执行重新烘焙。

## 环境

- Python 3.9+
- NumPy 1.22+
- 无 GPU、无外部资产、无 Docker

Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

如果只想直接运行，也可仅安装 NumPy，然后在仓库根目录执行模块命令。

## 快速开始

生成确定性样本场景：

```powershell
python -m voxel_gi.sample_scene artifacts\sample_scene.json
```

烘焙并生成报告：

```powershell
python -m voxel_gi.cli artifacts\sample_scene.json artifacts\sample_cache.vgic --threads 4 --report artifacts\bake_report.json
```

只校验已有缓存：

```powershell
python -m voxel_gi.cli artifacts\sample_scene.json artifacts\sample_cache.vgic --validate --report artifacts\validate_report.json
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

运行完整验收并生成 `artifacts\verification.json`：

```powershell
python tools\run_acceptance.py
```

## 输出约定

- SH 系数形状为 `(probe_count, 9, 3)`：9 个三阶实球谐基函数，每个基函数有 RGB 三个通道。
- 存储精度为 little-endian float32。
- 每个探针附带一个 float32 相对 RMS 残差。
- 方向采样是在固定 `up` 轴周围均匀立体角的半球采样；每个探针的方位相位由固定种子确定。
- 缓存不含时间戳、线程数或运行耗时，因此相同输入和算法版本的缓存字节可重复。

## 结构化错误

CLI 的参数/输入错误以 JSON 写入 stderr，例如：

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_MESH_DEGENERATE",
    "message": "Mesh contains a degenerate triangle",
    "details": {"mesh": "floor", "triangle": 0}
  }
}
```

退出码：输入错误 `2`，缓存校验错误 `3`，I/O 错误 `4`，内部错误 `5`。

## 实际验证结果

完整验收通过 Windows PowerShell 调用同一台机器的 Windows Python 3.10.9、NumPy
2.2.6 执行；11 项 `unittest` 全部通过。随后又从 WSL/Linux 对同一场景独立烘焙，
生成的缓存与 Windows 缓存逐字节一致。代码仅使用 `pathlib`、`os.replace` 等跨平台
API，并按 Python 3.9 语法约束编写。

本次样本结果：

- 16 个探针，每个探针 9×3 个 float32 SH 系数和一个 float32 残差。
- 平均残差 `0.7645672969520092`，最大残差 `0.8899112939834595`。
- 逐探针 RGB 通道总辐照度最大值 `0.11186858785573867`，上界 `8.2`。
- 缓存 SHA-256 `bd80205f4d08621a0f8a4e5bd2ac34f6d307029c4016b2cb5e67f04268d4fe53`。
- Windows 4 线程与单线程载荷逐位一致；`--validate` 未修改缓存。

实际结果见 `artifacts/verification.json`、`artifacts/bake_report.json` 和
`artifacts/validate_report.json`。

更完整的算法、能量语义、缓存格式和确定性说明见 [TECHNICAL.md](TECHNICAL.md)。
