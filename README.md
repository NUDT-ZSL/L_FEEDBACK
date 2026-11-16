# 第一人称空间音频离线渲染器

这是一个可在 Windows 命令行运行的确定性离线渲染工具。它读取场景 JSON 和
WAV 多轨素材，按听者头部四元数轨迹与声源位置曲线生成双耳立体声，并提供
响度归一化、动态范围压缩、循环、多普勒和 AABB 遮挡处理。

项目不依赖网络或外部音频素材。`spatial_audio_renderer.demo` 会确定性生成
完整的测试场景、WAV 素材和 JSON，可直接用于验收。

## 环境与安装

- Python 3.9+
- NumPy
- SoundFile（libsndfile 由 SoundFile wheel 提供）

Windows PowerShell 或 CMD：

```bat
py -3 -m pip install -r requirements.txt
py -3 -m pip install setuptools
py -3 -m pip install -e .
```

`setuptools` 仅用于构建/可编辑安装。不安装包也可以在当前目录直接运行：

```bat
py -3 -m spatial_audio_renderer --help
spatial_render.cmd --help
```

`spatial_render.cmd` 会优先使用 Windows `py` launcher，找不到时回退到
`python`。

## 快速运行内置场景

生成确定性的 8 秒场景并同时渲染：

```bat
py -3 -m spatial_audio_renderer.demo demo_scene ^
  --output demo_scene\mix.wav --normalize-lufs -16 --compress
```

只生成素材：

```bat
py -3 -m spatial_audio_renderer.demo demo_scene
spatial_render.cmd demo_scene\scene.json demo_scene\mix.wav
```

演示场景包含对白、循环环境声和移动声源，声源会穿过 AABB 遮挡体；头部在
8 秒内完成 90 度偏航旋转。

## 命令行

```text
spatial-render SCENE OUTPUT
  [--normalize] [--normalize-lufs LUFS]
  [--compress] [--threshold-db DB] [--ratio R] [--attack-ms MS]
  [--release-ms MS] [--knee-db DB] [--makeup-db DB]
```

- `SCENE`：场景 JSON 路径。
- `OUTPUT`：输出 `.wav` 路径，必须为 WAV 扩展名。
- `--normalize`：启用响度归一化，默认目标 `-16 LUFS`。
- `--normalize-lufs` / `--target-lufs`：指定目标积分响度并启用归一化。
- `--compress`：启用主总线压缩，使用默认参数。
- 任一压缩参数单独出现时也会自动启用压缩。
- 压缩默认值：阈值 `-18 dBFS`、比率 `4:1`、启动 `10 ms`、释放
  `120 ms`、软拐点 `6 dB`。

退出码：

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功；缺失或损坏的声源轨道只警告并跳过 |
| `1` | 参数或场景 JSON 错误；在打开输出文件前返回 |
| `2` | 运行时 I/O、磁盘空间或渲染错误；清理临时文件后返回 |

渲染进度以“每秒音频输出一行”的方式写到 `stderr`：

```text
progress: 25.00%
...
finished: duration=8.000 s, peak=-3.60 dBFS, integrated=-16.00 LUFS
```

警告也写入 `stderr`，例如：

```text
warning: skipping track 'dialogue': cannot decode WAV ...
```

## 场景 JSON

时间戳均为整数采样点。数组中的时间轴必须严格单调递增；头部四元数序列和
每个声源位置曲线分别校验。四元数采用标量在前格式 `[w, x, y, z]`，解析时
自动归一化，零四元数会被拒绝。

```json
{
  "format": "spatial-scene-v1",
  "sample_rate": 48000,
  "frame_length": 256,
  "duration_samples": 384000,
  "speed_of_sound": 343.0,
  "reference_distance": 1.0,
  "head_radius": 0.0875,
  "occlusion_base_cutoff_hz": 18000.0,
  "head": {
    "position": [0.0, 0.0, 0.0],
    "quaternion_keys": [
      {"time": 0, "quaternion": [1.0, 0.0, 0.0, 0.0]},
      {"time": 384000, "quaternion": [0.9238795, 0.0, 0.3826834, 0.0]}
    ]
  },
  "sources": [
    {
      "name": "dialogue",
      "file": "assets/dialogue.wav",
      "gain": 0.9,
      "loop": false,
      "start_sample": 0,
      "position_keys": [
        {"time": 0, "position": [-1.2, 0.0, 2.0]},
        {"time": 384000, "position": [-1.2, 0.0, 2.0]}
      ]
    },
    {
      "name": "ambience",
      "file": "assets/ambience_loop.wav",
      "gain_db": -6.0,
      "loop": true,
      "start_sample": 0,
      "position_keys": [
        {"time": 0, "position": [3.0, 0.4, -2.0]}
      ]
    }
  ],
  "occluders": [
    {
      "name": "screen",
      "min": [-4.0, -2.0, 3.6],
      "max": [4.0, 2.0, 4.2],
      "attenuation": 0.55,
      "cutoff_hz": 5200.0
    }
  ]
}
```

字段说明：

- 坐标轴：`+X` 右、`+Y` 上、`+Z` 前。默认听者位于原点，面向 `+Z`。
- `duration_samples` 可省略；此时取头部/声源轨迹终点与已加载素材结束点的
  最大值。循环素材应显式提供场景时长。
- `head.position` 可替换为 `head.position_keys`，支持移动听者。
- 声源 `file` 相对场景 JSON 所在目录解析；Windows 反斜杠路径在非 Windows
  测试环境下也会转换。
- `gain` 是线性增益；也可用 `gain_db`，两者不能同时出现。
- `start_sample` 是该轨道在场景时间轴上的起点。
- `loop=true` 时，轨道素材独立循环，其他轨道不受影响。
- 遮挡体 `min`/`max` 为 AABB 两角。`attenuation` 是单层线性增益，
  `cutoff_hz` 是单层低通截止频率。
- 常用别名也被接受：`tracks`、`positions`、`time`/`t`、
  `position`/`p`、`quaternion`/`q`。若四元数是裸数组列表，则按
  `frame_length` 隐含时间间隔解析。

输入 WAV 可以是 SoundFile 支持的多声道文件，渲染前会下混为单声道；若素材
采样率与场景不同，会先做带限重采样。

## 处理链路

1. 每个输出采样点对位置曲线做分段线性插值，对四元数做最短路径 SLERP；
   两者使用同一个场景时间戳。
2. 将世界坐标中的声源方向旋转到头部坐标，生成侧向分量、ITD、ILD 和
   单层头影参数。ITD 由 `2r/c * lateral` 计算，使用三次插值分数延迟线。
3. HRTF 近似由耳侧一阶低通（头影）与最小相位陷波器（耳廓）组成。两者
   所有极点/零点均在单位圆内，滤波器参数逐帧平滑并跨帧保留状态。
4. 距离采用 inverse 模型：`reference_distance / max(distance,
   reference_distance)`。
5. 多普勒速率为 `c / (c + v_radial)`，其中外向径向速度为正；速率限制在
   `[0.25, 4.0]`，并经过 4 ms 一阶平滑以柔化关键帧速度突变。
6. 每轨使用 32 抽头 Kaiser 窗带限 sinc 重采样。播放速率大于 1 时截止频率
   随 `1/rate` 下降以抑制混叠。分数相位时钟跨块连续，所有轨道在相同输出
   采样索引求和。
7. AABB 使用 slab 法逐采样求交。每增加一层遮挡，截止频率乘以该遮挡体的
   相对衰减因子，增益乘以 `attenuation`；一阶低通状态和 12 ms 增益平滑
   状态跨帧延续。
8. 压缩器使用立体声联动峰值检测、软拐点、启动/释放指数包络，无前视和
   额外采样延迟。
9. 简化 K 加权积分响度按 100 ms 能量段组成 400 ms、75% 重叠块，并执行
   `-70 LUFS` 绝对门限与相对 `-10 LU` 门限。
10. 输出采用两遍流式链路：第一遍写临时 float32 原始文件并统计响度，
    第二遍施加归一化增益并分块写 16 位 PCM WAV。

## 输出与可靠性

- 输出固定为双声道、16 位 PCM WAV。
- 场景音频逐帧处理，最终文件逐块写出；不会把整段渲染结果保存在内存中。
- 临时文件位于输出目录，成功时原子替换，失败时删除。
- 开始渲染前按原始 float32 临时文件和最终 PCM 文件总量检查磁盘空间。
- 16 位转换发生削波时输出 `warning: output clipped at 0 dBFS`。
- 参数和场景结构在创建输出临时文件前完成校验。

## 测试

```bat
py -3 -m unittest discover -s tests -v
```

测试覆盖：

- 双耳 ILD 对称性、右声源右耳更响与 ITD 到达顺序。
- 多普勒闭式速率、跨块相位连续性、重采样频移。
- AABB 求交、层数导致截止/增益下降、跨帧滤波等价性。
- 压缩软拐点、启动/释放包络、零采样延迟。
- K 加权响度随电平变化。
- 缺失/损坏轨道跳过与命名警告。
- 非单调时间轴退出码 `1`、写入失败退出码 `2`、临时文件清理。
- 归一化后响度落点与最终 WAV 格式。
- 内置 WAV/JSON 与重复渲染输出的逐字节确定性。

实际运行结果见 [VERIFICATION.md](VERIFICATION.md)。

## 已知边界

- HRTF 是可复现的参数化最小相位近似，不是测量 SOFA 数据集。
- 只处理直达声、距离、多普勒、头影和 AABB 遮挡，不包含房间反射或混响。
- 场景为单听者；遮挡体为静态 AABB。
- 输出场景流式处理，但每个输入 WAV 轨道会完整解码到内存。
- BS.1770 实现使用简化滤波器设计，不声称替代正式合规响度表。
- 当前仓库验证环境为 WSL2/Linux；Windows 入口使用 `pathlib` 与 `.cmd`
  启动器，未在真实 Windows 主机上执行。
