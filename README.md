# 离线依赖混淆靶场

这是一个完全离线、固定种子的 Python 依赖混淆实验环境。它只用 Python 标准库，同时模拟公共索引和私有索引，演示“公共优先”解析如何被同名高版本投毒，并提供三种修复与复核策略。

## 环境

- Windows 10/11 64 位
- Python 3.11+（验收命令使用 `python`；开发机 WSL 使用 `python3`，宿主 Windows 实测使用现成的 3.12）
- `curl`（Windows PowerShell 中建议写 `curl.exe`，避免与内置别名混淆）
- 无第三方 Web 框架、无外部网络、无账号、无容器

## 快速开始

在仓库根目录运行完整演示：

```text
python demo.py
```

默认使用 `127.0.0.1:5000`。端口被占用时可指定空闲端口：

```text
python demo.py --port 5080
```

演示会以子进程启动真实服务，轮询 `/status`，依次完成注入、检测、`pin`/`hash`/`private-first` 修复与验证，最后调用关闭接口并通过 `finally` 清理子进程。成功输出为：

```text
索引服务已启动
注入攻击完成
检测到风险: internal-utils
修复完成: pin
验证通过: pin
修复完成: hash
验证通过: hash
修复完成: private-first
验证通过: private-first
索引服务已关闭
```

## 文件结构

```text
depconfusion/
  versions.py    简化 PEP 440 版本比较与约束判断
  manifest.py    清单解析、哈希指令和策略标记
  state.py       固定种子生成、JSON 持久化、快照哈希、注入校验
  service.py     索引服务
  client.py      标准库 HTTP 客户端、3 次/1 秒重试
  resolver.py    公共优先分解析器
  injector.py    公共仓库投毒 CLI
  detector.py    risks/latent 与 stale 检测
  repair.py      pin/hash/private-first 修复
  validator.py   按策略重跑解析并校验哈希
service.py       服务入口
resolver.py      解析入口
injector.py      注入入口
detector.py      检测入口
repair.py        修复入口
validator.py     验证入口
demo.py          端到端演示
tests/           unittest 验收套件
artifacts/       实际运行结果
```

## 启动索引服务

```text
python service.py --host 127.0.0.1 --port 5000 --state data/index_state.json --seed 20240517
```

服务先绑定端口，再加载或生成状态。端口被占用时会在绑定阶段快速以退出码 `2` 结束，不会修改已有服务的状态。状态文件采用同目录临时文件加原子替换写入。

接口：

```text
GET  /status
GET  /<repo>/<name>
GET  /<repo>/<name>/<version>
GET  /<repo>/_snapshot
POST /<repo>/_snapshot
POST /public/<name>
POST /shutdown
```

`<repo>` 只能是 `public` 或 `private`。未知仓库返回 `400`，未知包返回 `404`。包接口除要求的 `name`、`version`、`sha256`、`uploaded_at` 外，还返回 `versions` 列表供解析器求约束。`sha256` 固定为 64 位小写十六进制。

默认数据包含 10 个私有包、13 个公共包，每个私有包至少 3 个版本。`internal-utils` 同时存在于两个仓库，私有最高版为 `3.8.2`，公共初始最高版为 `1.5.0`。

`GET /<repo>/_snapshot` 返回当前快照摘要、包数量、每包版本数和仓库快照哈希。`POST` 或 `?rebuild=true` 会重建并持久化摘要，不删除已注入数据。

## curl 验收

Windows CMD 示例（PowerShell 请把 `curl` 写作 `curl.exe`）：

```text
curl http://127.0.0.1:5000/status
curl http://127.0.0.1:5000/public/internal-utils
curl http://127.0.0.1:5000/private/_snapshot
curl -X POST http://127.0.0.1:5000/public/_snapshot
curl -X POST http://127.0.0.1:5000/public/internal-utils -H "Content-Type: application/json" -d "{\"version\":\"9.9.9\",\"payload\":\"malicious\"}"
curl http://127.0.0.1:5000/nope/package
curl http://127.0.0.1:5000/public/no-such-package
curl -X POST http://127.0.0.1:5000/shutdown
```

PowerShell 中注入请求可先构造正文，避免多层引号：

```powershell
$body = '{"version":"9.9.9","payload":"malicious"}'
curl.exe -X POST http://127.0.0.1:5000/public/internal-utils -H "Content-Type: application/json" -d $body
```

## 清单格式

```text
# 注释和空行会被忽略
internal-utils>=1.0.0
internal-auth==1.2.0  # 精确版本
internal-utils==3.8.2 --hash=sha256:<64位小写摘要>
```

只支持 `name`、`==` 和 `>=`。无法解析的非注释行会跳过并在 stderr 输出警告，最终退出码仍为 `0`。

默认来源顺序为 `[public, private]`：公共仓库有满足约束的最高版则选公共，否则选私有最高满足版，否则标记 `unresolved`。`private-first` 策略把顺序改为 `[private, public]`。

## 解析、注入、检测

解析：

```text
python resolver.py --manifest requirements.txt --base-url http://127.0.0.1:5000
python resolver.py --manifest requirements.txt --policy private-first --base-url http://127.0.0.1:5000
```

JSON 输出同时含 `resolved` 和 `unresolved`，每项记录来源、版本、摘要、哈希校验状态及私有/公共最高版本。

注入：

```text
python injector.py --package internal-utils --version 9.9.9 --payload evil --base-url http://127.0.0.1:5000 --log inject.log
```

注入版本必须严格高于私有最高版，否则服务返回 `409`。成功或失败都会向 `inject.log` 追加带 UTC 时间、`SUCCESS`/`FAILURE` 和原因的记录。制品摘要为 `sha256(payload UTF-8)`。

检测：

```text
python detector.py --dir . --output report.json --base-url http://127.0.0.1:5000
```

扫描目录下所有 `*.txt` 清单。报告字段：

- `risks`：实际解析结果来自公共仓库，且私有仓库存在同名包。
- `latent`：实际未选公共、选择了私有同名包，但公共仓库存在注入版本且私有最高版高于公共原始基线。
- `snapshot_sha256` / `final_snapshot_sha256`：扫描开始/结束时的全局快照哈希。
- `stale`：两个哈希不同时为 `true`，表示扫描期间索引发生变化。
- `summary`：`packages`、`requirements`、`resolved`、`unresolved`、`risks/risk_count`、`latent/latent_count`。

测试时可加 `--scan-delay 1.2` 或环境变量 `DEPCONFUSION_SCAN_DELAY=1.2`，在初始快照之后留出并发注入窗口。

## 修复与验证

`pin`：改写为私有仓库满足原约束的最高精确版本。私有包存在但没有满足版本时拒绝。

```text
python repair.py pin --manifest requirements.txt --base-url http://127.0.0.1:5000
python validator.py pin --manifest requirements.txt --base-url http://127.0.0.1:5000
```

`hash`：向原清单行追加私有最高满足版的 `--hash=sha256:<摘要>`；解析时哈希作为候选过滤条件，公共同名投毒制品会失配。私有包存在但没有满足版本时拒绝。

```text
python repair.py hash --manifest requirements.txt --base-url http://127.0.0.1:5000
python validator.py hash --manifest requirements.txt --base-url http://127.0.0.1:5000
```

`private-first`：在清单首部写入 `# depconfusion-policy: private-first`，并写同目录 `.depconfusion-policy.json`。解析器读取清单时会自动识别，也可显式传 `--policy private-first`。

```text
python repair.py private-first --manifest requirements.txt
python validator.py private-first --manifest requirements.txt --base-url http://127.0.0.1:5000
```

也可使用 `--strategy pin|hash|private-first` 形式。验证成功输出 `PASS: <策略> (resolved=N)` 并退出 `0`；失败退出非 `0`，stderr 以 `FAIL:` 开头并包含具体原因。`pin` 还要求精确约束并解析到私有制品，`hash` 必须存在哈希且实际制品摘要匹配，`private-first` 必须优先选到存在的私有制品。

## 退出码与重试

| 情况 | 退出码 |
| --- | --- |
| 成功，或清单中仅含被跳过的非法行 | `0` |
| 清单/目录不存在、修复拒绝、验证 FAIL、非法注入 | `1` |
| 端口占用/监听失败、索引不可达 | `2` |

所有需要索引的 CLI 先查询 `/status`。连接失败固定重试 3 次，每次间隔 1 秒；全部失败时 stderr 包含 `索引服务不可达`，退出码为 `2`。

关闭接口先持久化状态，再返回“索引状态已落盘，服务正在关闭”，延迟退出 `serve_forever`，确保客户端先读完响应。demo 的 `finally` 在任何失败路径都会终止子进程。

## 测试

```text
python -m unittest discover -s tests -v
```

当前实际结果：

- WSL/Linux Python：15 项测试全部通过，约 12.8 秒；demo 在默认 `5000` 端口连续运行两次均退出 `0`。
- 宿主 Windows Python 3.12：15 项测试全部通过，约 19.2 秒；demo 在默认 `5000` 端口连续运行两次均退出 `0`。
- Windows `curl.exe`：`/status` 为 `ready=true`，未知仓库 `400`、未知包 `404`、注入成功 `200`、重复注入 `409`，检测为 `risks=1`，关闭接口正常落盘退出。

覆盖接口/快照、种子确定性、公共/私有解析、注入日志、risks/latent、stale、三策略 PASS/FAIL、错误码、3 次重试、端口冲突、状态持久化、demo 端口冲突保护和同端口 demo 复跑。

实际运行输出保存在 `artifacts/test-output.txt`、`artifacts/demo-output.txt`、`artifacts/demo-rerun-output.txt`、`artifacts/curl-verification.txt`、`artifacts/edge-case-verification.txt`、`artifacts/windows-test-output.txt`、`artifacts/windows-demo-output.txt`、`artifacts/windows-demo-rerun-output.txt` 和 `artifacts/windows-curl-verification.txt`。
