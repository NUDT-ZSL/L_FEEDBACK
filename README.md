# audit_workbench

离线、确定性的 C/C++ 源码安全准入审计工作台。项目只使用 Python 标准库，
不访问网络，不依赖 MSVC、Docker、外部数据集或真实凭据。所有源码样本、配置、
故障夹具和基线都由工具自身按固定种子生成。

## 环境

- 目标运行环境：Windows 10/11 + Python 3.11；入口为
  `py -3.11 -m audit_workbench`。
- 当前验证环境：Linux/WSL、Python 3.14.4（本机没有 Windows `py` 启动器）。
  实现刻意限制在 Python 3.11 语法和标准库范围内，因此同一代码可直接在
  Python 3.11 下运行。验证时用 `python3` 替代 `py -3.11`。
- 项目本身就是一个包目录，无需安装第三方依赖。从项目根目录执行模块即可。

## 快速开始

```text
py -3.11 -m audit_workbench gen --seed 20240501 --out samples

py -3.11 -m audit_workbench scan ^
  --root samples --configs cfgA,cfgB --jobs 4 ^
  --json report.json --text report.txt

py -3.11 -m audit_workbench scan ^
  --root samples --configs cfgA,cfgB,cfgC --jobs 4 --timeout 2 ^
  --json timeout-report.json

py -3.11 -m audit_workbench scan ^
  --root samples --configs cfgA ^
  --baseline <snapshot-id> --fail-on "new-high>0" ^
  --json gate-report.json
```

PowerShell 下续行符应改为反引号。`--fail-on "new-high>0"` 需要加引号，避免
shell 将 `>` 解释为重定向。

`gen` 输出的 `roots.json` 同时是样本清单、配置清单和验收期望值清单。`scan`
检测到 `roots.json` 时会展开其中的 3 个源码根，并从同目录的 `configs/` 解析
`cfgA/cfgB/cfgC`。

## 确定性样本

每个源码根都有 3 层以上源码目录、`.c/.h` 文件、include guard、`#pragma once`、
对象式/函数式宏、字符串化、`#ifdef/#ifndef/#if defined`、跨文件声明与定义、
函数指针表，以及至少 6 处危险调用。固定种子相同则目录清单和全部文件
SHA-256 完全一致；换种子会改变根目录、内部目录、文件名和内容。

三个配置的关键差异：

- `cfgA`：`ENABLE_UNSAFE=1`，因此条件块内的 `strcpy/getenv` 与
  `system(argv[1])` 存在并可达。
- `cfgB`：`ENABLE_UNSAFE` 显式未定义，同一调用点消失；
  `FEATURE_B_ONLY=1` 则使另一个 `sprintf/getenv` 调用点只在 B 中存在。
- `cfgC`：启用 `FEATURE_ALT`，并对 `unit_c` 启用生成的
  `@audit-hang` 夹具，用于验证真实子进程超时。

样本中的稳定调用点 ID 写在源码标记中，例如
`/* @call:unit_a.orphan_strcpy */`。没有标记的调用点使用
`file + function + callee + normalized arguments + ordinal` 生成身份键，
不包含行号或列号。

## 配置模型

配置 JSON 示例：

```json
{
  "name": "cfgA",
  "macros": [
    {"name": "ENABLE_UNSAFE", "defined": true, "value": "1"},
    {"name": "FEATURE_ALT", "defined": false}
  ],
  "include_paths": ["include_x", "src_x"],
  "language_standard": "c11",
  "entries": ["entry_unit_a", "entry_unit_b", "entry_unit_c"]
}
```

也支持 `macros` 为普通对象（`"FEATURE": "1"`、`"FEATURE": null`）。
宏重复定义且值不同会得到 `E_MACRO_CONFLICT`。配置等价比较忽略显示名，
但严格比较宏集合、include 路径顺序、语言标识和入口函数集合。

## 分析与判定

- 预处理器支持限制深度的 include、include guard、`#pragma once`、条件编译、
  对象式/函数式/可变参数宏、嵌套展开、字符串化和简单的 `##`。
- 解析器在有效行上重建函数边界、直接调用、跨文件调用、函数指针表调用和
  外部结构体字段调用。
- 可达性为 `reachable/unreachable/unknown`。外部结构体字段函数指针等无法静态
  确定的调用为 `unknown`，并带非空原因。
- 污点来源包括 `read_input`、`recv_from_socket`、`getenv`、命令行的
  `argv`/`argc`、`getopt` 等。简单赋值和函数参数传递会传播污点；递归宏、
  未解析标识符或不支持的语法得到 `unknown`。
- 只有 `reachability=reachable && taint=tainted` 才进入 `high_risk` 集合。
  顶层三态汇总由逐点明细重新计算，不另行维护第二套计数。
- 每条报告包含文件/行/列、函数、规则、配置、路径、`truncated` 与
  `path_truncated`。路径按边存储，最多保留 `--max-path-hops` 跳；超限时
  显式标记 `truncated=true`，不会静默裁剪。

## 错误与退出码

稳定业务错误码包括：

```text
E_INCLUDE_CYCLE       include 栈中再次遇到同一文件
E_MACRO_CONFLICT      同名宏先后定义为不同值
E_SNAPSHOT_CORRUPT    JSON 截断、校验和不匹配、ID 不匹配或读到临时文件
E_TIMEOUT             子进程组合超时并被真实终止
E_PARSE_LIMIT         include 深度、有效行数、展开次数或展开深度超限
E_BASELINE_MISMATCH   门禁基线与当前源码/配置不兼容
E_GUARD_FAILED        门禁阈值或迁移规则被触发
E_ARGS/E_IO/E_CONFIG/E_WORKER 等辅助码
```

每个错误对象严格包含 `code/message/context` 三字段，`message` 非空，
`context` 至少包含文件、行、组合或哈希之一。

退出码：

```text
audit_clean=0                 审计完成且无错误对象；门禁通过
guard_pass=0                  基线门禁通过（与 audit_clean 同值）
audit_complete_with_errors=1  审计完成但结论含错误对象（默认）
tool_failure=2                参数非法、输入缺失、输出不可写等工具失败
strict_failure=3              --strict 下审计完成但含错误对象
guard_failure=4               基线门禁被触发
```

同一组值同时出现在 `--help` 的 `EXIT CONTRACT` 和
`metadata --json` 的 `exit_codes` 字段中。

## 快照、差异和门禁

- 快照 ID 是规范化 payload 的 SHA-256，payload 包含源码根内容哈希、配置内容
  哈希、规则版本、逐调用点结论、错误和汇总。
- 写入先落到同目录临时文件，`fsync` 后 `os.replace` 原子提交；读取只接受
  已提交且校验和正确的 `<64位SHA>.json`。
- 基线差异分为 `added`、`disappeared`、`transitions`。同身份但行号移动不会
  产生差异；调用语义指纹改变时不会复用旧条目，而归入新增/消失。
- 门禁只读取预加载的已提交快照，不做全量高危计数。
- 默认门禁允许“稳定调用身份仍重叠”的增量源码演进，以便在修改样本后检测
  新增高危；若源码根集合、配置内容、规则版本或调用身份整体不重叠，则拒绝
  判定并输出 `E_BASELINE_MISMATCH`，不执行阈值比较。需要任何字节级内容变化
  都拒绝时使用 `--strict-content-lock`。
- 截断/篡改快照会在审计和当前快照写入之前被拒绝，因此不会产生部分差异，
  也不会用当前审计覆盖损坏的基线文件。

## 并发与超时

每个“源码根 × 配置”组合在独立子进程中运行。POSIX 使用新 session/进程组，
超时后先 `SIGTERM` 再 `SIGKILL`；Windows 使用新进程组和
`taskkill /F /T`。父进程在超时后等待子进程退出并清理临时文件。

超时组合记录 `status=timed_out` 和 `E_TIMEOUT`，其余组合及部分结论完整保留。
`--jobs 1` 与 `--jobs 4` 的逐点结论和规范化 JSON 均为字节级一致。

## 解析限制

为防止恶意或畸形样本拖垮批处理，当前硬上限声明在 `metadata --json`：

```text
include depth      12
virtual lines      30000
macro expansions   20000
expansion depth    48
```

单个组合仍有子进程超时作为最终隔离边界。解析器不是完整 C/C++ 编译器，
不支持模板、完整 C++ 语义、复杂宏运算符和跨过程路径敏感别名分析；这些
情况应保守落入 `unknown` 或解析失败，而不是被静默当作安全。

## 验收脚本

```text
py -3.11 tests/acceptance.py --work-dir artifacts/acceptance-work
```

当前实际运行结果：在 Linux/WSL Python 3.14.4 上，72 项检查全部通过，耗时约
7 秒。覆盖两次生成逐文件 SHA 一致性、配置等价、双向条件编译、三态汇总、
污点/高危互证、五类业务错误、快照损坏/临时文件拒绝、状态迁移、门禁通过/
新增高危/失败迁移、9 组合并发超时隔离、1/4 并发等价、无残留子进程、
JSON 字节复现和文本/JSON 一致性。证据保存在
`artifacts/acceptance/results.json`。

## 代码地图

```text
audit_workbench/generator.py    确定性样本和故障夹具
audit_workbench/config.py       配置模型、JSON、等价比较、宏冲突
audit_workbench/preprocessor.py 受限预处理、include/条件/宏、限制
audit_workbench/cparse.py       词法、函数、调用点、函数指针表
audit_workbench/analysis.py     可达性、路径、污点、高危
audit_workbench/scheduler.py    子进程隔离、并发、真实超时
audit_workbench/snapshot.py     原子快照、校验、基线差异
audit_workbench/engine.py       编排、报告、退出契约和门禁
audit_workbench/cli.py          CLI 与 JSON/文本输出
tests/acceptance.py             端到端验收
```

