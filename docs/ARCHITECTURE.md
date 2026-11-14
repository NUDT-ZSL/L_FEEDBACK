# Architecture

## 数据流

每个逻辑帧按以下顺序执行：

1. 到达恢复帧时，从保存的完整光源集合恢复场景，并重新发布严格更大的版本号。
2. 动画池对已到开始时间的任务执行按帧线性插值；每个光源更新单独递增一次场景版本。
3. 固定种子生成器执行增、删、动画排队；第 0 帧只排队不修改光源，用于验证版本未变时跳过上传。
4. 保存快照（若配置）。
5. 场景层捕获双缓冲快照。
6. 同步器比较版本：立即上传、延迟排队或跳过上传。
7. 渲染器使用 GPU resident 快照渲染。
8. 调试模式回读 SSBO 并与 resident 快照逐字节比较。

## 场景双缓冲

`SceneManager` 有两个 `SceneSnapshot` 槽：

- 写操作持有独占共享互斥锁，修改 `working_`。
- 每次修改递增 `version_`，将完整光源集合复制到非活动槽，再切换活动槽。
- 渲染端 `snapshot()` 持有共享锁并复制活动槽。

因此读取者只能看到发布前或发布后的完整集合，不会看到“数量已增加但数据尚未写完”的中间状态。版本与光源集合在同一发布动作中可见。

## 版本与上传

`LightBufferSynchronizer` 的 resident 状态包含完整 `SceneSnapshot`、GPU 布局镜像和容量：

- 版本相同：`uploaded=false`，直接复用。
- 版本变化且数量不超过容量：`glBufferSubData`，`buffer_grew=false`。
- 数量超过容量：容量翻倍，仅此时使用 `glBufferData`。
- 延迟模式：每个版本记录源帧和 `source_frame + N` 就绪帧；选择已就绪的最新快照，消除旧快照。

SSBO 头为 8 个 `uint32`：

```text
[0] total count
[1] point count
[2] directional count
[3] reserved
[4] version low
[5] version high
[6] reserved
[7] reserved
```

其后是 48 字节光源记录：位置+类型、方向+范围、颜色+强度。C++ 布局与 GLSL `std430` 布局一致。

## 动画池

`AnimationPool` 在构造时分配固定数量的槽位和空闲索引。排队时捕获当前位置/光强和目标值：

- 开始帧：`current_frame`
- 完成帧：`current_frame + duration_frames`
- 后续帧先按 `(frame - start) / duration` 插值，再写回场景。
- 池满返回 `ANIM_POOL_FULL`；光源已删除返回 `ANIM_LIGHT_NOT_FOUND`。

动画更新仍走 `SceneManager::updateLight`，因此版本号和双缓冲语义与普通修改完全一致。

## 延迟模式的显示一致性

延迟模式不混合“最新 CPU 光源集合”和“旧 GPU 缓冲”。渲染只读取同步器 resident 快照；日志同时记录：

- `scene_version`：最新前端版本。
- `gpu_version`：当前 GPU 缓冲版本。
- `gpu_source_frame`：当前 GPU 快照的有效来源帧；版本已相等且无需上传时向前对齐到当前帧。
- `snapshot_age_frames`：当前帧减来源帧。

在持续修改期间，resident 快照的每个元素都来自同一个历史版本，因此不会出现个别光源位置/强度错位。

## 调试回读

GPU 上传后，`glGetBufferSubData` 读取实际使用的 SSBO 字节。校验顺序：

1. 缓冲长度。
2. `scene_version`。
3. 总光源数。
4. 点光源和方向光计数。
5. 每个光源的 48 字节记录。

不一致时输出 `debug_error` JSON 并返回 `75`。`--inject-debug-fault` 只在 GPU SSBO 头写入错误计数，用于验证真实回读路径能够检测并退出。
