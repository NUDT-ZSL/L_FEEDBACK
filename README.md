# 离线分镜混流系统

本项目按玩家选择路径，把对白、环境音效和音乐三类分镜音轨合成为一条连续
多声道 WAV，并输出采样点级对齐报告。所有音频读取、重采样、包络、总线混合
和限幅均在本地完成，不需要网络或外部音频素材。

## 环境与安装

```bash
python -m pip install "numpy>=1.24" "soundfile>=0.12"
```

Python 3.10+、64 位环境。项目根目录直接运行，无需额外安装本包；也可执行
`python -m pip install -e .`。

## 快速运行

```bash
python examples/generate_demo.py --run
```

或手动使用题目要求的入口：

```bash
python -m mixer \
  --config examples/demo/shots.json \
  --path examples/demo/choices.json \
  --out examples/demo/mix.wav \
  --report examples/demo/report.json
```

`examples/generate_demo.py` 用确定性正弦生成器创建 44.1kHz/48kHz、单声道/
立体声素材。三个分镜为 `0-1000ms`、`900-1800ms`、`1800-2600ms`，第二段
声明 100ms 等功率交叉淡变，最终固定为 124800 个采样点、4 声道、48kHz
PCM16 WAV。

## 输入格式

分镜描述采用毫秒。`start_ms` 必须小于 `end_ms`，时间统一为毫秒；可通过
`time_unit: "ms"` 显式声明。

```json
{
  "schema_version": 1,
  "time_unit": "ms",
  "target_sample_rate": 48000,
  "output_channels": 4,
  "output_subtype": "PCM_16",
  "bus_weights": {
    "dialogue": 1.0,
    "ambience": 0.7,
    "music": 0.5
  },
  "shots": [
    {
      "id": "s2",
      "start_ms": 900,
      "end_ms": 1800,
      "allow_overlap": true,
      "crossfade_ms": 100,
      "tracks": [
        {
          "type": "dialogue",
          "path": "media/s2_dialogue.wav",
          "gain_db": 0.0,
          "fade_in_ms": 30,
          "fade_out_ms": 30
        },
        {
          "type": "ambience",
          "path": "media/s2_ambience.wav",
          "gain_db": 0.0,
          "fade_in_ms": 30,
          "fade_out_ms": 30
        },
        {
          "type": "music",
          "path": "media/s2_music.wav",
          "gain_db": 0.0,
          "fade_in_ms": 30,
          "fade_out_ms": 30
        }
      ]
    }
  ]
}
```

`tracks` 也可写成 `{"dialogue": {"path": "...", ...}, ...}`。素材路径相对
分镜描述文件所在目录解析；输入采样率只接受 44.1kHz 和 48kHz，多声道按
算术平均降混为单声道，再线性重采样到目标采样率。`gain_db` 以 dB 表示，
也接受非负线性 `gain`/`volume` 字段。`fade_in_ms + fade_out_ms` 不得超过
分镜时长。

选择文件支持 `{"path": ["s1", "s2", ...]}`、`{"choices": [...]}`，或直接
使用字符串数组。未知分镜、重复分镜和空路径均为 `E_PATH_INVALID`。

## 混音语义

- 毫秒到采样点使用十进制半入（half-up）舍入，全部时间轴均以整数采样点计算。
- 输入素材按分镜时长截断或尾部补零；素材短于分镜时不会阻止其他分镜混音。
- 每个音轨先应用 dB 增益与等功率淡入/淡出，再加入对应分类总线。
- 合法的分镜重叠使用等功率曲线：后到分镜 `sin(a)`、先前分镜 `cos(a)`，
  因此交叉区满足 `sin²(a)+cos²(a)=1`，能量连续。
- 三条总线默认按 `dialogue=1.0`、`ambience=0.7`、`music=0.5` 合成为单声道
  主输出。默认 WAV 为 4 声道，依次是未加权对白总线、环境总线、音乐总线
  和主输出；`output_channels` 为 1/2 时只写主输出（2 声道为双声道副本），
  为 3 时只写三条分类总线，大于 4 时在第四声道之后重复主输出。
- 主输出峰值超过 0dBFS 时使用统一峰值限幅增益并硬性钳位，报告记录
  `limiter_active`、`limiter_gain` 和最终 `master_peak_dbfs`。
- 最终总时长取所选分镜的最大结束采样点，包含合法时间空隙中的静音。

## 报告

报告固定包含 `schema_version/status/target_sample_rate/total_samples/shots/
bus_weights/bus_peak_dbfs/master_peak_dbfs/limiter_active/limiter_gain`。每个
分镜包含整数 `start_sample`、`end_sample`、`duration_samples`，以及最终混音
对应分镜自身音轨的 `raw_peak` 和 `peak_dbfs`：原始峰值在音轨增益与淡入
淡出之后、交叉淡变和总线权重之前取得。峰值不高于 `1e-12` 时用 `-120.0`
表示静音，以保持 JSON 可由标准解析器读取。JSON 使用排序键、固定缩进与
固定换行写出。`bus_peak_dbfs` 是应用总线权重前的总线峰值；
`master_peak_dbfs` 取限幅后的最终主输出。

## 错误与部分报告

所有业务错误退出码为 `2`，错误码同时写入 stderr 和报告的 `error_code`：

| 错误码 | 触发条件 |
| --- | --- |
| `E_SHOT_DESC` | JSON/字段/时间/采样率/声道等分镜描述非法 |
| `E_MEDIA_LOAD` | 素材不存在、不可读、格式或输入采样率不支持 |
| `E_TIMELINE_CONFLICT` | 所选分镜重叠但未声明 `allow_overlap`/交叉淡变 |
| `E_FADE_OVERFLOW` | 淡入淡出为负或时长超过分镜 |
| `E_PATH_INVALID` | 选择路径为空、引用未知或重复分镜 |

解析、时间轴或素材阶段失败时仍会写出 `status: "error"` 的部分报告；失败前
已经通过校验的分镜会保留在 `shots` 中，其峰值字段为 `null`。

## 确定性、并发与测试

WAV 与 JSON 均通过同目录临时文件加 `os.replace` 原子发布。并发进程即使指向
同一输出路径，也不会读到半写文件；相同输入生成逐字节相同的 WAV 和报告。
本地验收命令：

```bash
python -m unittest discover -s tests -v
```

当前 8 个用例覆盖 44.1kHz 立体声重采样/降混、采样点时长、交叉淡变能量、
总线限幅、五类错误码及部分报告、逐字节确定性，以及 4 进程同路径并发写入。
