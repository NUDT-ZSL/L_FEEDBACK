# PE 补丁审计工具

纯标准库、离线、可在 Windows 10/11 + Python 3.11 上运行的确定性 PE 补丁审计项目。工具先生成补丁前后两个 PE32 风格样本，再按函数名建立稳定身份映射，在身份之上解析调用和计算调用图可达性，最终输出可复核的 JSON 审计报告。

> 样本是 **PE32 风格固件审计夹具**，包含标准 PE 头、节表、真实导入描述符和自定义定宽函数字节码；它不是 Windows 可直接执行的程序。

## 快速开始

PowerShell：

```powershell
python generate_samples.py --seed 0 --out artifacts/acceptance/fixture
python audit_cli.py `
  --before artifacts/acceptance/fixture/patch_before.bin `
  --after artifacts/acceptance/fixture/patch_after.bin `
  --before-manifest artifacts/acceptance/fixture/before_manifest.json `
  --after-manifest artifacts/acceptance/fixture/after_manifest.json `
  --defects artifacts/acceptance/fixture/defects.json `
  --entries artifacts/acceptance/fixture/entries.json `
  --output artifacts/acceptance/audit_report.json
```

cmd.exe：

```bat
python generate_samples.py --seed 0 --out artifacts\acceptance\fixture
python audit_cli.py --before artifacts\acceptance\fixture\patch_before.bin --after artifacts\acceptance\fixture\patch_after.bin --before-manifest artifacts\acceptance\fixture\before_manifest.json --after-manifest artifacts\acceptance\fixture\after_manifest.json --defects artifacts\acceptance\fixture\defects.json --entries artifacts\acceptance\fixture\entries.json --output artifacts\acceptance\audit_report.json
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

也可直接运行 `run_demo.bat`，或在 Windows setup 中用 `py -3.11` 替换 `python`。

## 生成器

```text
python generate_samples.py --seed <int> --out <dir>
```

输出：

- `patch_before.bin`、`patch_after.bin`
- `before_manifest.json`、`after_manifest.json`
- `entries.json`
- `defects.json`

两版样本的函数集合和函数名完全一致，但函数排列顺序、函数基址、对齐填充和无关指令不同。字节码中的本地调用只保存 **函数名表索引**，导入调用只保存 **导入表索引**，不保存函数 RVA 或文件偏移。

同一 `seed` 的全部输出逐字节复现；不同 `seed` 的两版样本哈希不同。时间戳固定为 0，不使用 `hash()`、系统随机数或当前时间。

## 输入格式

### 函数边界清单

```json
{
  "schema_version": 1,
  "version": "before",
  "functions": [
    {"name": "entry_main", "rva": 5456, "length": 48}
  ]
}
```

每版清单独立校验：`name` 必须存在且不重复，`rva` 为非负整数，`length` 为正整数，区间必须完全落在映射的节数据内。同一版本内区间重叠、字段缺失、重名或越界都会失败。两版之间允许函数新增/删除；存在函数按名字对齐，因此地址变化不会影响配对。

### 入口点

```json
{"schema_version": 1, "entries": ["entry_main"]}
```

命令行可用 `--entries` 提供。若省略，依次使用缺陷 JSON 的 `entry_points`、清单中的 `entry_points`，最后才从 PE 头入口 RVA 反查函数名。

### 缺陷规则

```json
{
  "schema_version": 1,
  "entry_points": ["entry_main"],
  "rules": [
    {
      "id": "D1",
      "target": "unsafe_copy",
      "dangerous_imports": [{"module": "msvcrt.dll", "name": "strcpy"}],
      "rule": {"type": "guarded_call", "branch_number": 0}
    },
    {
      "id": "D2",
      "target": "orphan_copy",
      "dangerous_imports": [{"module": "msvcrt.dll", "name": "strcpy"}],
      "rule": {"type": "forbid_call"}
    }
  ]
}
```

`dangerous_imports` 也接受 `"msvcrt.dll!strcpy"` 或 `"strcpy"` 字符串形式。字符串不能为空；对象形式必须同时包含非空 `module` 和 `name`。规则类型：

- `forbid_call`：目标函数出现任意危险导入调用即 `VULNERABLE`。
- `guarded_call`：每个危险导入调用都带有与 `branch_number` 相同的最近条件分支才为 `CLEAN`；任一调用缺少或分支号不符为 `VULNERABLE`。

规则 ID 必须唯一，目标必须至少出现在一版清单中。字段缺失、类型错误、未知规则类型或未知目标均返回 `RULE_INVALID`，且不写报告。

## 审计算法

1. 解析两版 PE 头、节表、导入表和本地函数名表。
2. 校验两版函数边界，并按 `name` 建立身份映射；禁止用 RVA/文件偏移直接配对函数。
3. 在每个函数内解码定宽指令。非法四字节字和截断尾部按字节计数，随后继续解码；显式填充指令不进入有效指令序列。
4. 本地调用按名字表索引解析为函数身份，导入调用按导入索引解析为 `module!name`。调用证据只记录函数内相对偏移和分支号。
5. 按函数身份比较有效指令签名。签名包含立即数、本地目标名、导入名和分支号，不包含 RVA、文件偏移或分支绝对目标。填充变化和整体挪位不会误报。
6. 从入口集在函数名调用图上做确定性 BFS，生成无重复节点最短路径；用强连通分量报告环和自递归。
7. 对每条规则分别评定补丁前、补丁后状态，再结合可达性和证据完整性给出结论。

## 结论映射

| 条件 | 结论 | `basis` |
|---|---|---|
| 目标函数在补丁后清单中被删除 | 已封堵 | `target_deleted` |
| 补丁前有漏洞，补丁后 `CLEAN` 且证据完整 | 已封堵 | `patched_clean` |
| 补丁前后都 `CLEAN` | 已封堵 | `already_clean` |
| 补丁后仍 `VULNERABLE` 且可从入口到达 | 仍可达 | `vulnerable_and_reachable` |
| 补丁后漏洞函数不可达，图完整 | 证据不足 | `vulnerable_unreachable` |
| 目标函数解码损坏/调用断裂 | 证据不足 | `evidence_incomplete` |
| 不可达判断受缺失入口或断裂调用影响 | 证据不足 | `graph_incomplete` |

“仍可达”目标的 `path` 给出函数序列，`path_edges` 和 `call_evidence` 给出调用相对偏移；`call_evidence` 还包含分支相对偏移、实际分支号和期望分支号。

## 退出码与错误格式

| 退出码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | I/O 或二进制结构错误 |
| 3 | 清单/入口点格式错误 |
| 4 | `BOUNDS_OVERFLOW`（函数区间越界） |
| 5 | `RULE_INVALID`（缺陷规则无效） |

错误只写 stderr，格式固定：

```json
{"schema_version": 1, "error": {"code": "BOUNDS_OVERFLOW", "message": "..."}}
```

所有输入校验完成后才会原子写报告；输入错误不会产生半成品报告。

## 确定性保证

- 所有函数、导入、调用、边、目标、规则和结论集合均稳定排序。
- 报告不含时间戳、绝对路径、临时文件名或随机 ID。
- 同一输入重复运行逐字节一致。
- 清单数组乱序不改变报告。
- 生成器同 seed 逐字节一致，异 seed 样本哈希不同。

## 项目结构

```text
pe_audit/
  binary.py        PE 夹具构建、解析、字节码解码
  generator.py     确定性样本与默认元数据生成
  manifests.py     函数边界校验和名字对齐
  callgraph.py     调用解析与图证据
  diffing.py       有效指令序列差异
  rules.py         规则校验和 VULNERABLE/CLEAN 判定
  reachability.py  路径、环、孤立节点
  audit.py         三态结论和确定性报告
  cli.py           审计命令行
tests/             61 个 unittest，覆盖全部验收点
artifacts/acceptance/  实际运行样本、报告和验证结果
```

## 实际运行结果

默认样本 seed 为 0。`artifacts/acceptance/audit_report.json` 得到：

- `D1 unsafe_copy`：已封堵（`patched_clean`，补丁后危险 `strcpy` 消失）
- `D2 orphan_copy`：证据不足（`vulnerable_unreachable`，仍含危险调用但无入口路径）
- `D3 guarded_copy`：仍可达（`vulnerable_and_reachable`，附 `entry_main -> dispatch -> guarded_copy` 路径和偏移）

`artifacts/acceptance/verification.json` 记录测试、重复运行、清单乱序、错误退出码和 Python 3.11 语法检查结果。
