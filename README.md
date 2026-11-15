# 确定性分层角色渲染状态同步核心

这是一个可离线运行的 Python 3.10+ 渲染状态同步实现，不依赖 NumPy、
PyOpenGL、PyQt5 或网络。核心用不可变快照、整数定点时钟、逐字段补丁和
显式恢复链路解决动画时钟与 GPU 提交时钟不一致导致的材质闪烁、发光残留
和版本错配问题。

## 快速运行

```bash
# 带三类故障注入与恢复的确定性演示
python demo.py

# 机器可读输出
python demo.py --format json

# 仅正常链路
python demo.py --no-faults

# 21 项离线验收测试
python run_tests.py
# 或
python -m unittest discover -s tests -v
```

演示和测试均不需要安装第三方包。GUI 不是同步核心的前置条件；本实现演示
的是无窗口、可重复的提交包与校验值，不创建真实 OpenGL 上下文。

## 固定状态契约

动画层顺序固定为：

```text
locomotion(override) -> upper_body(override) -> breath(additive)
```

材质层顺序固定为：

```text
base_color(opaque) -> normal(normal_map) -> emissive(additive) -> transparent(alpha)
```

材质参数量固定为：

| 层类型 | 向量长度 | 含义 |
| --- | ---: | --- |
| `base_color` | 4 | RGBA 定点值 |
| `normal` | 3 | 编码法线 XYZ |
| `emissive` | 4 | RGB + 强度 |
| `transparent` | 4 | RGB + 透明度 |

默认 `exclusive` 规则禁止 `emissive` 与 `transparent` 同时启用；可通过
`make_default_state("stacked")` 构造允许两者叠加的状态，此时仍按固定层
顺序提交。材质槽固定存在，`remove_material()` 表示禁用并清空该槽的历史
参数与缓存，而不是改变层顺序。

## 双时钟与确定性

- 时间基为 `600_000` tick/秒，逻辑步长为 `10_000` tick，严格等于 60 Hz。
- `advance_animation(n)` 只推进逻辑帧和动画层时间，不读取墙钟。
- `submit_render(render_time_ticks)` 接收外部确定性时间戳，按
  `render_delta = render_time_ticks - logic_time_ticks` 采样关键帧。
- 所有姿态、权重、材质参数与光照值均为整数或整数定点值，关键帧线性插值
  使用显式整数除法，重复输入产生逐位一致的姿态、补丁和校验值。
- 每次渲染携带 `RenderTicket`，同时校验收到的状态、动画、材质和光源版本。

## 版本、脏标记与补丁

每次有效变更只提交一次新快照，并递增全局 `state_version`：

- 动画推进同时递增 `animation_version`。
- 材质启用/禁用、参数更新或移除同时递增 `material_version` 和目标层版本。
- 光源增删同时递增 `light_revision`。
- 任意版本变化都会重建当前版本的光照快照包装，使其
  `light_snapshot.state_version == state_version`。

事务先完整构造并验证候选快照，再一次性替换当前状态。非法材质组合、参数
长度错误或重复光源只返回结构化错误，不修改状态。

`pending_patch()` 仅遍历脏路径并生成字段级操作，例如：

```text
set material_layers/emissive/params/2
set animation_layers/locomotion/current_time_ticks
set light_snapshot/state_version
```

补丁包含基线版本/摘要与目标版本/摘要。`replay_delta_patch()` 会严格校验
字段白名单、版本、基线摘要和最终状态摘要；未知字段或缺失字段不会被静默
忽略，而是判定回放失败。

## 发光/透明残留治理

- 禁用 `emissive` 或 `transparent` 时立刻把层参数置零。
- 对应 `render_cache` 的 `params` 和 `value_fp` 同步清零，并递增
  `generation` 使旧缓存代次失效。
- 从禁用态重新启用时仍从全零参数和零缓存开始，不恢复历史值。
- 渲染提交包同时携带材质参数和缓存，校验值会覆盖两者，因此旧高亮或旧
  透明度若泄漏会改变后续校验值。

## 光照快照与拒绝策略

光源集合按 `light_id` 稳定排序。每次渲染提交前依次检查：

1. `light_snapshot.state_version == state_version`
2. `light_snapshot.light_revision == light_revision`
3. `light_snapshot.lights == current lights`

任一不匹配都会返回可恢复错误、拒绝本次提交，并写入带序号和版本信息的
`diagnostics()`。匹配时渲染只读取该光照快照，不直接读取可变光源集合。

## 异常恢复

以下场景进入 `recoverable_error`，不会覆盖最近一次有效快照：

| 场景 | 业务错误码 |
| --- | --- |
| 逻辑步进不在 `1..600` | `E_LOGIC_STEP_RANGE` |
| 补丁回放/摘要校验失败 | `E_PATCH_REPLAY_FAILED` |
| 光照快照状态版本不匹配 | `E_LIGHT_SNAPSHOT_VERSION_MISMATCH` |
| 光照快照修订号不匹配 | `E_LIGHT_SNAPSHOT_REVISION_MISMATCH` |
| 光照快照内容不匹配 | `E_LIGHT_SNAPSHOT_CONTENT_MISMATCH` |
| 动画/材质/光源渲染票据过期 | `E_COUPLING_VERSION_MISMATCH` |

调用 `rollback()` 恢复最近有效快照、版本、逻辑帧和一致的光照快照，清除
脏集合与错误状态，并递增 `recovery_epoch`。随后可继续推进动画和提交。
非法材质校验属于普通业务拒绝，不进入故障态，但仍可在 `diagnostics()`
查询。

## 代码结构

- `render_sync/model.py`：不可变状态、固定层契约、关键帧采样基础与默认样本。
- `render_sync/patch.py`：字段级补丁、严格反序列化和摘要回放。
- `render_sync/core.py`：事务、版本、脏标记、渲染门票、光照校验和回滚。
- `render_sync/demo.py`：确定性事件序列、故障注入、逐帧摘要与校验值。
- `tests/test_render_sync.py`：标准库 `unittest` 验收测试。

## 实际验证结果

验证环境为本地 Python 3.14.4（代码保持 Python 3.10+ 兼容），未安装第三方
依赖。执行结果：

```text
$ python run_tests.py
Ran 21 tests
OK

$ python demo.py
frame=00 label=boot ... patch_ops=0 checksum=db17c857...
...
fault=logic_step_range code=E_LOGIC_STEP_RANGE recoverable=true rolled_back_to=16
fault=patch_replay_failure code=E_PATCH_REPLAY_FAILED recoverable=true rolled_back_to=17
fault=light_snapshot_version_mismatch code=E_LIGHT_SNAPSHOT_VERSION_MISMATCH recoverable=true rolled_back_to=18
final status=ready state_version=19 logic_frame=16 frames=14
final_render_checksum=652a50ce7af7434c5390bef99d1f454c14427f22575733aec76341955768d6df
exit_code=0
```

连续两次 JSON 演示输出经 `diff` 比较完全一致。完整摘要记录见
`artifacts/verification_report.json`。
