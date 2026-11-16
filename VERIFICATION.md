# 验证记录

验证环境（本地，无 Docker，无外部素材）：

- Python 3.14.4、NumPy 2.5.5
- FFmpeg 8.0.1（`ffmpeg`/`ffprobe`）
- 样本由 `tools/generate_samples.py` 使用 lavfi 的 `testsrc` 与 `sine` 确定性生成

## 自动化测试

命令：`python -m unittest discover -s tests -v`

结果：**30 个用例全部通过**，覆盖：

- 工程校验：缺失文件、封装格式、轨道数量、偏移范围、包络范围/长度、
  标记数量/偏差范围、非法 JSON 的退出码 2 与 stderr JSON 错误对象
- 单元逻辑：整数帧时间戳（含 29.97 型有理数帧率）、标记颜色确定性、
  2px/ms 偏移与边界夹取、正负偏移的静音/裁剪、尾部截断、包络线性插值、
  峰值归一化（>1.0 整体缩放到 0.99，否则不缩放）
- 端到端：H.264 + AAC 192k、48 kHz 立体声、时长误差、中间视频与状态文件
  落盘、标记帧渲染、增量复用（中间视频字节不变并报告受影响轨道）、
  包络变化走增量、视频内容变化与中间文件被篡改后强制全量重建、
  残留 `*.tmp` 状态文件清理、输出目录自动创建、峰值归一化日志、
  运行期错误退出码 1

## 手工验收演示

全量运行（4 秒、640x360、30 fps、3 条音频、2 个标记）：

- `ffprobe` 输出 `duration=4.000000`，视频 `h264` 120 帧，音频 `aac`
  采样率 48000、码率约 192 kb/s，两条流 `start_time=0.000000`
- 同目录生成 `output.intermediate.mp4`、`output.mp4.state.json`、
  `output.audiocache/`

增量运行（仅修改一条轨道包络）：

- stdout 报告 `incremental_reuse` 与 `changed_tracks`
- 中间视频 SHA-256 与修改前完全一致，输出时长仍为 4.000 秒

错误码：

- 工程校验失败：stderr `{"code":"VALIDATION_ERROR",...}`，退出码 2
- `FFMPEG_PATH` 指向无效目录：stderr `{"code":"FFMPEG_ERROR",...}`，退出码 1

补充边界验证：

- MKV 封装、25 fps、1 秒视频搭配 5 秒音频：输出时长 1.000 秒、25 帧，
  超长音频被截断，无标记时不绘制矩形，中间视频正常持久化
- AAC 编码固有 priming/padding 仅带来少量额外音频字节，容器时长与
  视频严格一致，满足 ±50 ms 误差要求
