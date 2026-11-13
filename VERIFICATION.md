# 实际验证结果

验证环境：Linux 回环地址 + Python 标准库（代码目标为 Windows 10/11 / Python 3.11，
仅使用跨平台标准库 API：`os.replace`、`pathlib`、`http.client`、`sqlite3`、
`ThreadingHTTPServer`）。测试服务监听 `127.0.0.1` 动态端口，固定种子内容，
全程无外部网络访问、无第三方包、无容器。

## 测试执行

命令：`python -m unittest discover -s tests -v`

结果：**25 个测试全部通过（OK），耗时约 43 秒**。完整日志见
`artifacts/unittest_output.txt`。

覆盖映射：

| 需求 | 对应测试 |
| --- | --- |
| 1 多源调度与双层限速 | `RateLimitTests.test_per_source_interval_and_inflight_caps`（重建到达/完成时间窗，断言每源间隔与每源/全局在途峰值） |
| 2 跨源内容去重 | `FullPipelineTests.test_02_cross_source_dedup`、`BlobAndDbTests.test_content_dedup_and_attribution_uniqueness` |
| 3 版本识别与整体替换 | `VersioningTests.test_hash_change_creates_version_and_keeps_old_blob`、`BlobAndDbTests.test_version_pointer_updates_only_on_hash_change` |
| 4 中断恢复状态机 | `RecoveryTests.test_hard_exit_then_resume_and_temp_cleanup`、`BlobAndDbTests.test_inflight_reset_on_restart`、`test_atomic_blob_write_and_temp_cleanup` |
| 5 重试与源级熔断 | `FullPipelineTests.test_05_retry_then_success_and_4xx_no_retry`、`test_06_circuit_breaker_trips_and_recovers` |
| 6 规范化再哈希 | `NormalizeTests` 全部 9 项（含 1 字节分块、CRLF 跨块、非法/未知字符集回退、latin-1、非文本、超大文本释放缓存） |
| 7 可审计报告/确定性 | `FullPipelineTests.test_01_full_run_and_report_reconciliation`、`test_08_deterministic_two_runs` |
| 8 范围与安全/超大响应 | `FullPipelineTests.test_03_same_origin_and_depth_limits`、`test_04_oversized_streaming_and_flag` |
| 304 不新增版本 | `FullPipelineTests.test_07_304_creates_no_version_or_content`、`NotModifiedCrawlerTests.test_worker_304_keeps_single_version` |

## 双跑确定性与对账

命令：`python scripts/run_verification.py`（结果落盘
`artifacts/verification_result.json`）。

- `report_reconciliation`: **PASS**（报告计数与数据库 SQL 同口径重建相等，内容哈希集合与磁盘 Blob 集合相等）。
- `deterministic_excluding_timestamps`: **true**（两次独立运行去除起止时间后逐字段相等）。
- `content_hash_sets_equal`: **true**，`blob_count = 12`。
- 一次运行的总量：`fetched=48, done=24, failed=8, dedup_hits=24, retries=16`。
- 失败 URL 最终错误类型示例：`http_500`、`http_404`、`connection_reset`、`cross_origin_redirect`；其中 404/跨域重定向 `attempts=1`（不重试），500/重置 `attempts=4`（达到重试上限）。

## 限速实测

配置 `start_interval=0.2s`、每源并发 2、全局并发 3，基于服务器端
`time.monotonic()` 到达戳测得每源相邻请求最小间隔约 0.19–0.20s；时间窗重叠
峰值不超过每源 2、全局 3。

## 复现步骤

```bash
python -m unittest discover -s tests -v
python scripts/run_verification.py
```
