# 验证记录（本地确定性样本，2026-11-14）

环境：WSL Ubuntu（Python 3.14.4 + NumPy 2.5.4，用户态安装；代码兼容 3.11+/NumPy 1.26+）。
样本：`gen-sample --width 128 --height 128 --seed 11`（spacing=2 → 64×64 探针，8×8 探针/块 → 64 块）。
执行：`python3 run_acceptance.py` → **29/29 通过，退出码 0**（原始输出见 `.work/acceptance_last_run.txt`）。

## 覆盖项与实测结果

- 全量烘焙：64/64 块完成，日志 `estimated_cost = probes_processed × 1e-3`（不读时钟）。
- 多线程按位一致：`PROBE_BAKE_THREADS=1` 与 `=8` 的 `.rgba` 输出字节完全相同，并与参考烘焙一致。
- 强制终止恢复：烘焙中 `SIGKILL`（模拟 `taskkill /F`），已持久化 14 块；重启后仅补算 50 块，结果与未中断烘焙按位一致。
- Ctrl+C 语义：`SIGINT` 后当前块写盘、进程退出码 0、日志 `status=interrupted`，续烘结果按位一致。
- 局部重建：编辑矩形 `40,48,24,16` → 重算 49 块、沿用 15 块；与编辑后高度场全量烘焙逐块一致（max|Δ|=0，按位相同）。
- 重建校验负例：篡改被沿用块的持久化数据 → `E_RECONSTRUCT_MISMATCH`，退出码 2。
- 版本切换：同版本重烘 0 块（纯跳过）；换版本后旧状态清空、64/64 块全量重烘。
- 预演模式：`--dry-run` 输出总块数 64、块网格 [8,8]、预计内存，未创建状态目录/结果/日志。
- 内存预算：`--max-cached-blocks 2`（需求 5 块）→ `E_MEMORY_LIMIT`，退出码 2。
- 非法输入：坏魔数 / 数据长度不符 / 哈希不符 → `E_HEADER_MISMATCH`；NaN 样本 → `E_HEIGHT_INVALID`；均退出码 2，错误码仅以 JSON 出现在 stderr。
- 日志比较：版本不同 → `full_rebake_required` 含原因；版本同、完成块不同 → 列出差异块坐标。
