# 离线快照库采集流水线

纯 Python 3.11 标准库实现的页批量归档采集器，输出一个 SQLite 快照库和按
内容哈希寻址的原始响应体文件。不依赖第三方包、数据库服务、Docker 或外部
网络；测试站点由 `http.server` 在回环动态端口本地模拟，内容由固定种子生成。
同时兼容 Windows 10/11 与 Linux。

## 目录结构

- `snapshot_crawler/`
  - `config.py`：JSON 配置与命令行覆盖参数。
  - `crawler.py`：多源 BFS 调度、双层限速、重试退避、源级熔断、状态机、`os._exit` 中断点。
  - `fetcher.py`：基于 `http.client` 的流式抓取、同源重定向、304、最大字节截断与错误分类。
  - `normalize.py`：charset 解码、LF/行尾空白规范化、流式 SHA-256，解码失败回退原始字节。
  - `blobstore.py`：临时文件 + `os.replace` 原子改名的内容寻址 Blob 存储与启动清理。
  - `db.py`：SQLite 表结构、URL 状态机、内容/归属/版本/指针/事件/条件缓存。
  - `urlutils.py`：URL 规范化、同源判定（协议+主机+端口）、HTML 链接提取。
  - `report.py`：可审计 JSON 报告及按数据库同口径重建的对账断言。
  - `cli.py` / `__main__.py`：命令行入口。
- `tests/`：确定性本地测试服务与 25 个测试（单元 + 端到端）。
- `scripts/run_verification.py`：一键双跑确定性与对账验证。
- `config/sample.json`：配置样例。

## 运行方式

```bash
# 单元 + 端到端测试（自动启停本地测试服务，不访问外网）
python -m unittest discover -s tests -v

# 双跑确定性 / 对账验证
python scripts/run_verification.py

# 真实采集（先用 tests/test_server.py 起本地服务，或替换为自有站点）
python -m snapshot_crawler --config config/sample.json --output-dir out

# 中断恢复演练：第 N 个请求完成后 os._exit，再次执行同命令即恢复
python -m snapshot_crawler --config config/sample.json --exit-after 2
python -m snapshot_crawler --config config/sample.json
```

Windows PowerShell 下命令相同（`python` 替换为实际解释器即可），路径均由
`pathlib` 处理，原子改名使用跨平台的 `os.replace`。

## 配置项

`sources[].root_url`、每源并发 `per_source_concurrency`、每源起始间隔
`start_interval`、`global_concurrency`、`max_depth`、`max_pages`，以及
`timeout`、`max_retries`、`backoff_base/cap`、`circuit_threshold`、
`circuit_cooldown`、`max_bytes`。命令行参数可覆盖超时、重试、熔断、字节上限
和输出目录，见 `python -m snapshot_crawler --help`。

## 输出布局

```
<output_dir>/
  snapshot.db            # SQLite 快照库（WAL 模式）
  report.json            # 可审计报告
  blobs/ab/cd/<sha256>   # 内容寻址的原始响应体（含分片目录）
```

## 八项需求落实位置

1. 多源调度与双层限速：`crawler.py` 的门闩 `acquire_send_slot`、每源/全局信号量。
2. 跨源去重：`db.py` 的 `content` 表仅以哈希为主键（无来源列），`attribution` 记录归属。
3. 版本与原子替换：Blob 先写临时文件再 `os.replace`，`versions` + `current_pointer` 仅在哈希变化时新增。
4. 可中断恢复：`pending/inflight/done/failed` 状态机，启动 `reset_inflight` 与 `cleanup_temp`，`--exit-after`。
5. 重试与熔断：5xx/重置/超时指数退避，404/400 立即失败，源级连续失败熔断与冷却自恢复。
6. 规范化再哈希：`normalize.py` 流式处理，仅换行/行尾空白差异不改变哈希。
7. 可审计报告：`report.py` 全部计数由 SQL 同口径重建并断言相等，双跑除时间外逐字段相等。
8. 范围与安全：同源判定拒绝外部域名/非 http(s)/超深度链接，`max_bytes` 流式截断标 `oversized`，片段去除、查询保留。
