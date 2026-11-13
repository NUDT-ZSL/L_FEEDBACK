# 离线对白剪辑工作站 — 技术说明

## 运行环境

- Python 3.11（本仓库验证版本 3.11.18）
- 依赖：`numpy==1.26.4`、`soundfile==0.12.1`（见 `requirements.txt`）
- 无网络/GPU/外部账号要求；所有 WAV 由 `workstation/fixtures.py` 本地确定性生成
- 所有文件路径均通过 `pathlib.Path` 处理，可在 Windows 上以
  `python -m workstation <command>` 运行

## 模块结构

- `workstation/fingerprint.py`：素材内容 SHA-256 指纹（只哈希文件字节，
  不依赖修改时间）与入参规范化 JSON 指纹
- `workstation/errors.py`：稳定错误码枚举 `E_MISSING / E_FORMAT / E_SHORT`
- `workstation/resample.py`：采样级线性插值重采样，
  输出帧数 `ceil(n * dst_rate / src_rate)`
- `workstation/sources.py`：WAV 解码、错误登记（单素材失败不中断整体流程）、
  项目采样率重采样缓存
- `workstation/timeline.py`：台词表解析与时间轴映射，
  采样偏移采用四舍五入 `floor(x + 0.5)`，原始浮点秒数原样保留
- `workstation/fades.py`：片段淡入/淡出与同轨重叠片段线性交叉淡化，
  包络只依赖绝对采样帧，天然与块大小无关
- `workstation/mixer.py`：多轨混音（增益/声像/静音/独奏）、分块流式输出、
  实时预览与快照增量更新
- `workstation/exporter.py`：分块写入多声道 WAV，进度状态持久化到 JSON，
  支持崩溃续导与入参指纹校验
- `workstation/project.py`：工程状态保存与基于内容指纹的恢复校验
- `workstation/fixtures.py`：确定性素材/台词表/混音配置生成
- `workstation/cli.py`、`workstation/__main__.py`：命令行入口
- `tests/test_workstation.py`：13 项验收测试

## 关键设计

### 1. 时间轴对齐

- `start_frame = floor(start_time * project_rate + 0.5)`（半值向上）
- 报告同时输出 `start_time`（原始浮点）、`start_frame`（整数偏移）与
  `rounding_delta_frames`
- 不同采样率素材在加载时一次性重采样到项目采样率，片段只保存项目率下帧数

### 2. 块大小无关性

- 淡化包络与交叉淡化全部按绝对帧号计算，渲染函数 `render_range` 无跨块状态
- 已验证块大小 128 / 512 / 1024 / 2003 产出的 WAV 字节级相同

### 3. 实时预览

- `PreviewEngine.preview(start, count)` 通过 `clips_for_range` 只筛选与请求
  区间相交的片段，不扫描整条时间轴
- 分段请求拼接与一次性请求在采样级一致（测试 `test_05`）

### 4. 可恢复导出

- 每写一块即把 `version`、`input_fingerprint`、`frames_written` 等原子写入
  JSON 状态文件（先写 `.tmp` 再 `replace`）
- 续导以 `r+b` 打开已存在 WAV，`seek` 到已写帧继续追加；量化由
  libsndfile 统一完成，因此续导文件与一次性导出字节级一致
- 入参指纹变化时删除旧文件并从头导出，且标记 `resumed=false`

### 5. 异常素材

- 缺失：`E_MISSING`；soundfile 无法识别/损坏：`E_FORMAT`；
  合法容器但零帧：`E_SHORT`
- 错误记录含素材路径与错误码，其他有效素材继续处理

### 6. 淡化与交叉淡化

- 淡入/淡出时长按项目率换算为帧数，半开区间内线性变化、采样级连续
- 重叠区间 `outgoing = 1 - t`、`incoming = t`，两片段增益之和恒为 1
- 混音末端统一 `clip(-1, 1)`，保证不超过 0 dBFS

### 7. 快照与增量

- 快照整体替换时比较版本号；版本变化即调用 `remap_timeline` 按原始浮点
  时间重新计算全部采样偏移，不沿用旧偏移
- 单轨参数修改输出差异报告（受影响帧区间与变化帧数）

### 8. 工程恢复校验

- 状态内保存每个素材的内容指纹；恢复时重新哈希比对
- 指纹不一致（`E_CHANGED`）或缺失（`E_MISSING`）的素材，其对应片段被
  拒绝并从恢复后的时间轴剔除，其余片段正常恢复

## 命令行用法

```bash
python -m workstation init <workspace>          # 生成确定性素材与配置
python -m workstation build <workspace>         # 对齐并生成工程状态/报告
python -m workstation preview <workspace> \
    --start 100 --frames 300 [--out pcm.bin]    # 实时混音预览
python -m workstation export <workspace> \
    --block-size 512                            # 分块导出（可断点续导）
python -m workstation demo <workspace>          # 一键跑通完整流程
```

## 实际验证结果

在 Python 3.11.18 + numpy 1.26.4 + soundfile 0.12.1 下：

- `python -m unittest discover -s tests`：13 项测试全部通过
- 对齐取整：`0.12345s @48k -> 5926` 帧（`floor(5925.6+0.5)`）
- 块大小 128/512/1024/2003 导出 WAV 的 SHA-256 完全相同
- 模拟崩溃后续导文件与一次性导出 SHA-256 相同（字节级一致）
- 错误码报告实际输出 `['E_MISSING', 'E_FORMAT', 'E_SHORT']`
- 恢复校验对素材篡改（含还原修改时间）稳定识别并拒绝对应片段

