# Spatial Audio Mixer

一个 C++17 编写的本地实时三维空间音频混音器。程序在运行时生成单声道
44.1 kHz / 16-bit PCM WAV 样本，通过 libsndfile 加载，完成距离衰减、基于
听者朝向的等功率声像、AABB 遮挡检测和低通滤波，再以 256 帧块混音并通过
PortAudio 输出 44.1 kHz 立体声 16-bit PCM。

项目不依赖外部音频素材，不使用容器；确定性模式输出可通过哈希复现的 WAV。

## 已实现能力

- 严格加载单声道、44.1 kHz、16-bit PCM WAV；不符合约束时返回非零错误码并向
  `stderr` 输出具体原因。
- 每个声源独立配置循环、初始位置、基础增益、最大可听距离和遮挡截止频率。
- 线性距离衰减：`d=0 -> gain=1`，`d>=maxDistance -> gain=0`。
- 听者朝向投影到右轴，按 `sqrt((1-pan)/2)` / `sqrt((1+pan)/2)` 计算等功率
  左右增益。
- 每个音频块读取最新位置；增益、pan 和滤波参数在块内逐样本线性过渡，块边界
  保持连续。
- AABB 线段相交遮挡检测；遮挡时对声源应用一阶低通，截止频率和目标滤波混合量
  均渐变，避免状态切换咔嗒声。
- 最多 32 个声源同时混音，测试覆盖 32 个满幅声源叠加。
- 浮点混音后使用软膝限幅器，量化峰值不超过 `32112`，不会发生整数溢出或
  `[-32768,32767]` 边界硬削波。
- PortAudio 回调只做预分配缓冲区上的 DSP 和采样，不分配内存、不加锁、不做
  文件 I/O 或日志。
- 声源增删通过固定容量 SPSC 无锁队列传递；移除在下一个块边界启动淡出，淡出
  完成后才允许槽位复用。
- 确定性测试模式使用固定种子的 `mt19937_64` 生成场景参数和移动轨迹，输出 WAV，
  打印每声源平均距离增益、平均 pan、左右增益、遮挡比例和平均截止频率。

## 目录

```text
include/spatial_audio/audio_engine.h   公共引擎 API
src/audio_engine.cpp                   DSP、混音、原子状态和无锁生命周期
src/wav_io.cpp                         libsndfile 严格 WAV 读写
src/portaudio_output.cpp               PortAudio 实时输出与回调
src/scenario.cpp                       确定性场景、运动轨迹和样本生成
src/main.cpp                           CLI、WAV 输出和结果校验
tests/audio_engine_tests.cpp           功能、并发生命周期和零分配测试
CMakeLists.txt                          Windows/Linux CMake 构建
```

## Windows 构建

依赖：

- CMake 3.20+
- MSVC 2019+ 或 MinGW-w64
- PortAudio 19.7.0+ 预编译库
- libsndfile 1.2.0+ 预编译库

目录布局可使用：

```text
C:\deps\portaudio\
  include\portaudio.h
  lib\portaudio_x64.lib

C:\deps\libsndfile\
  include\sndfile.h
  lib\sndfile.lib
```

设置环境变量并构建：

```bat
set PORTAUDIO_DIR=C:\deps\portaudio
set LIBSNDFILE_DIR=C:\deps\libsndfile

cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

MinGW 示例：

```bat
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

若使用 DLL 版依赖，需要把 `portaudio_x64.dll`、`libsndfile-1.dll` 等运行时
DLL 放到可执行文件同目录或 `PATH` 中。CMake 同时支持依赖自带的
`SndFileConfig.cmake` / `portaudioConfig.cmake`，以及上述常见预编译目录布局。

Linux/WSL 若已安装开发包，可直接设置 `CMAKE_PREFIX_PATH`；不构建实时部分时也可
使用 `-DSPATIAL_AUDIO_BUILD_REALTIME=OFF` 仅编译核心与确定性测试。

## 运行

确定性混音，默认生成 8 个持续声源，并在 1.0 s 添加、2.5 s 移除第 9 个声源：

```bat
build\Release\spatial_audio_mixer.exe --deterministic --duration 4 ^
  --out artifacts\deterministic_mix.wav ^
  --sample-dir artifacts\generated_samples
```

实时 PortAudio 播放：

```bat
build\Release\spatial_audio_mixer.exe --realtime --duration 10
```

单独验证 WAV 错误路径：

```bat
build\Release\spatial_audio_mixer.exe --probe-wav missing.wav
echo %ERRORLEVEL%
```

主要参数：

```text
--deterministic     离线确定性渲染
--realtime          PortAudio 实时输出
--probe-wav PATH    仅加载并校验指定 WAV
--duration SEC      时长，默认 4.0
--seed N            64 位确定性场景种子
--out PATH          确定性输出 WAV
--sample-dir DIR    运行时生成输入 WAV 的目录
```

## DSP 与线程模型

### 空间计算

坐标约定为 `x=右, y=上, z=前`。听者提供位置、前向和上向；右轴为
`normalize(cross(up, forward))`。声源偏移为 `p_s - p_l`：

```text
distanceGain = clamp(1 - |offset| / maxDistance, 0, 1)
pan          = clamp(dot(normalize(offset), right), -1, 1)
leftPan      = sqrt((1 - pan) / 2)
rightPan     = sqrt((1 + pan) / 2)
```

每个 256 帧块更新一次目标和位置，块内对 `distanceGain` 与 `pan` 进行逐样本
线性插值，并在每个样本重新计算等功率 pan，因此过渡期间左右声道功率和也保持
恒定。声源加入时从零增益淡入；移除时向零增益淡出。

### 遮挡

场景可配置最多 16 个 AABB 遮挡体。引擎对听者到声源的线段做 slab 相交测试。
遮挡目标为 `1`、无遮挡为 `0`；一阶低通系数
`a = 1 - exp(-2*pi*fc/fs)` 和干湿混合量都在块内渐变。

### 混音与限幅

所有声源先累加到浮点立体声缓冲。输出前使用连续软膝限幅：

```text
|x| <= 0.90: y = x
|x| >  0.90: y = sign(x) * (0.90 + 0.08 * tanh((|x|-0.90)/0.08))
```

函数连续且上界为 `0.98`，量化到 16-bit 后最大绝对值为 `32112`，因此不会出现
整数溢出或硬削波平台。

### 声源生命周期

- 主线程预加载全部样本，并预分配 32 个声源槽和 128 项命令队列。
- `addSource()` / `removeSource()` 通过原子槽状态和 SPSC 队列通知音频线程。
- 音频线程只在块起始处消费命令；移除请求先触发一个块长度的淡出，之后释放
  槽位并发布 `Removed` 状态。
- 主线程只会在看到 `Removed` 后复用槽位；旧指针不会在音频线程仍可能访问时
  被释放。样本由引擎统一持有，槽位销毁和样本释放均为 RAII。
- 位置、朝向、增益、距离、截止频率使用无锁原子变量传入回调。
- 回调的 DSP、统计和队列路径均不执行堆分配；测试中重载全局 `new` 验证了单块
  处理分配次数为 0。

## 确定性场景

默认场景包含：

- 8 个永久声源：220/247/262/294/330/349/392/440 Hz。
- 1 个生命周期声源：494 Hz，在 1.0 s 创建，在 2.5 s 请求移除。
- 所有输入 WAV 都是运行时生成的 1 秒无缝循环单声道 PCM16 样本。
- 听者做 Lissajous 移动并左右转向；声源绕场景中心运动。
- 位于 `(0,0,4.5)` 的墙面 AABB 会间隔遮挡多个声源。

确定性输出会校验：输出非静音、峰值不超过软限幅上界、所有永久声源处理完整
时长、统计值范围合法、至少发生遮挡、pan 同时覆盖左右方向、临时声源完成
创建/移除，并打印输出样本的 FNV-1a 哈希。

## 实测验证

本机环境：

```text
Ubuntu 26.04 / GCC 15.2.0 / CMake 3.30.5
PortAudio 19.7.0 (源码本地构建)
libsndfile 1.2.2 (源码本地构建)
```

`ctest --test-dir build --output-on-failure`：

```text
1/2 Test #1: audio_engine_tests ............... Passed
2/2 Test #2: deterministic_mix ................ Passed
100% tests passed, 0 tests failed out of 2
```

4 秒确定性混音实测：

```text
blocks=690 rendered_frames=176400
source[0]/220Hz avg_gain=0.4660 avg_pan=-0.8752 occluded=14.3%
source[1]/247Hz avg_gain=0.4740 avg_pan=+0.1918 occluded=0.0%
source[2]/262Hz avg_gain=0.5557 avg_pan=-0.9619 occluded=0.0%
source[3]/294Hz avg_gain=0.6431 avg_pan=-0.1093 occluded=42.8%
source[4]/330Hz avg_gain=0.5129 avg_pan=+0.0274 occluded=0.0%
source[5]/349Hz avg_gain=0.5144 avg_pan=-0.1729 occluded=69.1%
source[6]/392Hz avg_gain=0.5186 avg_pan=-0.9411 occluded=0.0%
source[7]/440Hz avg_gain=0.5012 avg_pan=-0.9226 occluded=0.0%
transient/494Hz blocks=259 occluded=44.4%
peak_abs=6153
fnv1a_wav16=0x8752b762e23a4e0e
validation=PASS
```

连续两次独立运行生成的 WAV SHA-256 均为
`fdafdd3891dd19858fb922681e0c4aa02ba1911524ad014e47b7e3af9664a9a1`，
确认输出可复现。

核心测试结果：

```text
distance 0 -> gain 1.0, distance 5/max 10 -> 0.5, distance 10/max 10 -> 0.0
center pan L=R=0.7071, right pan L=0, R=1
pan transition spread across one 256-frame block
8 kHz occlusion RMS 0.0900 -> 0.0059 with 500 Hz cutoff
32-source mix peak 32112, no integer hard clip
deferred remove, slot reuse and ID invalidation passed
audio callback allocations = 0
12 concurrent add/remove swaps passed while audio thread ran
```

ASan + UBSan + LeakSanitizer 运行全部测试和 4 秒确定性混音均为 `exit=0`，未报告
越界、未定义行为或泄漏。ThreadSanitizer 因当前 GCC 工具链未安装 `libtsan` 而
无法链接；并发路径由独立线程压力测试覆盖。

错误路径实测：

```text
--probe-wav missing.wav  -> stderr 明确说明 WAV 打开失败，exit=2
--realtime (无音频设备)   -> "no default output device"，exit=21
```

本执行环境没有声卡，因此实时模式只能验证 PortAudio 初始化失败的退出码和错误
信息；在 Windows 有声卡环境中，`--realtime` 会打开默认输出设备并运行同一场景。

## 错误码

| 值 | 含义 |
|---:|---|
| 1 | 参数无效 |
| 2 | WAV 打开失败 |
| 3 | WAV 格式不受支持 |
| 4 | WAV 读取失败 |
| 5 | 无空闲声源槽 |
| 6 | 命令队列已满 |
| 7 | 样本 ID 未加载 |
| 20 | PortAudio 初始化失败 |
| 21 | PortAudio 流打开失败 |
| 22 | PortAudio 流启动失败 |
