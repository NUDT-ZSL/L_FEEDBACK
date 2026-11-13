# L_FEEDBACK — 各向异性光照探针离线烘焙管线

为程序化高度场离线烘焙受地形遮蔽影响的半球余弦加权辐照度探针，
支持地形编辑后的局部探针增量重建与可恢复的断点续烘。

## 环境

- Python 3.11+（开发验证于 3.14）+ NumPy（1.26+/2.x 均可），无其他依赖、无网络需求。
- 线程数由环境变量 `PROBE_BAKE_THREADS` 配置（缺省用 `--threads`，默认 4）。

## 命令

```bash
# 1. 生成确定性高度场样本（自定义头 + float32，版本哈希 = blake2b(数据)）
python3 main.py gen-sample --out base.hf --width 128 --height 128 --seed 11
#    附带编辑：--edit-rect x,y,w,h --edit-delta 0.4

# 2. 全量 / 断点续烘（状态目录逐块原子持久化，版本变化自动整体失效重烘）
python3 main.py bake --heightfield base.hf --state-dir state \
    --out probes.rgba --log bake.log.json --spacing 2 --block-size 8 \
    --light-dir 0.4,0.3,0.866 --max-cached-blocks 64 [--verify-determinism]

# 3. 预演：只规划不积分、不写状态
python3 main.py bake --heightfield base.hf --state-dir s --out o --log l --dry-run

# 4. 编辑后的局部重建（只重算与编辑矩形相交的探针块，并对全量烘焙逐块校验）
python3 main.py reconstruct --heightfield edited.hf --state-dir state \
    --out probes.rgba --log recon.log.json --edit-rect 40,48,24,16 --tolerance 1e-5

# 5. 日志比较
python3 main.py compare-logs bake_a.log.json bake_b.log.json
```

## 输出

- `probes.rgba`：RGBA float32 小端原始贴图（高×宽×4）。RGB = 天空余弦加权
  辐照度（含遮蔽）+ 定向光贡献；A = 余弦加权天空可见度。旁车 `probes.rgba.json`
  记录尺寸/版本/参数哈希。
- 日志 JSON：版本哈希、总块数、完成块列表、按处理探针数 × 固定系数
  （1e-3）估算的耗时（不读系统时钟）。

## 关键语义

- **退出码**：成功 / Ctrl+C 优雅中断 = 0；业务错误 = 2。`E_xxx` 只出现在
  stderr 的 JSON 或结果文件中，绝不作为进程退出码。
- **错误码**：`E_HEADER_MISMATCH`（魔数/格式版本/尺寸/长度/版本哈希不符）、
  `E_HEIGHT_INVALID`（NaN/Inf 或超出 [0,1]）、`E_MEMORY_LIMIT`（缓存块预算超限）、
  `E_RECONSTRUCT_MISMATCH`（局部重建与全量烘焙逐块比对超容限）、
  `E_NONDETERMINISTIC`（同一输入两次烘焙字节不一致）。
- **持久化**：每块先写临时文件再 `os.replace` 原子落盘，manifest 记录版本哈希、
  参数哈希与完成块列表；版本或参数变化即清空旧状态全量重烘，绝不沿用旧块偏移。
- **确定性**：float64 定序累加、固定半球方向集与固定射线步进序列、无 BLAS 归约；
  任意线程数下输出按位一致（`--verify-determinism` 可自检）。

## 验证

```bash
python3 run_acceptance.py   # 29 项验收检查，详见 VERIFICATION.md
```
