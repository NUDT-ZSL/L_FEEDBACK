# 实际验证记录

日期：2026-11-17  
环境：WSL2 / Ubuntu 26.04，GCC 15.2，CMake 3.30.5，Release  
Windows/MSVC 状态：当前容器没有 MSVC；代码使用 C++20 标准库与跨平台 CMake，Windows 构建命令见 README。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOFTRASTER_BUILD_TESTS=ON
cmake --build build -j4
cmake -S . -B build-werror -DCMAKE_BUILD_TYPE=Release -DSOFTRASTER_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS=-Werror
cmake --build build-werror -j4
```

两项构建均通过。GCC 使用 `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`；`-Werror` 构建无告警。

另外使用 AddressSanitizer + UndefinedBehaviorSanitizer 的 Debug 构建运行了全部单元与端到端测试，无越界、未定义行为或泄漏报告：

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DSOFTRASTER_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-asan -j4
./build-asan/softraster_unit_tests
python3 tests/integration.py ./build-asan/softraster build-asan/integration_work
```

当前镜像没有 `ctest` 可执行文件，因此测试直接运行：

```bash
./build/softraster_unit_tests
python3 tests/integration.py ./build/softraster build/integration_work
```

## 确定性样本

样本由 `--generate-sample` 生成，包含：

- 14 边躯干、低多边形头部、16x5 环半透明角色护罩、地面。
- 3 个点光源，1 个带 512x512 阴影贴图的定向光。
- 16x16 内嵌切线空间法线贴图。
- 128 个半透明三角形，透明度 0.34，透明深度写阈值 0.5。

两次独立进程运行结果：

```text
PPM SHA-256: 986b0b44e64825e3b4888949d0fc25abd4d49fa9432dbc1d161c19a160eeb538
rasterizedTriangles: 187
clippedTriangles: 2
shadowSampleCount: 544761
transparentSortSwaps: 4039
transparentTriangles: 128
transparentFragments: 10792
transparentDepthWrites: 10792
opaqueFragments: 49737
peakMemoryEstimatedBytes: 6785712
shadowMapSize: 512
ppmBytes: 307215
```

`sha256sum` 对两次 PPM 输出一致；统计 JSON 两次逐字节一致。

## 错误路径

`tests/integration.py` 实际覆盖并全部通过：

| 场景 | 期望退出码/JSON code |
| --- | ---: |
| 无参数调用 | 5 / `INVALID_VALUE` |
| 场景文件不存在 | 1 / `IO_ERROR` |
| 删除 `camera.target` | 3 / `MISSING_FIELD` |
| mesh 索引设为 999999 | 7 / `INDEX_OUT_OF_RANGE` |
| `fovYDegrees` 写为 `1e999` | 6 / `NON_FINITE_FLOAT` |
| 阴影贴图设为 8193 | 8 / `SHADOW_MAP_TOO_LARGE` |
| 复制透明对象造成相同排序深度 | 9 / `DUPLICATE_TRANSPARENT_DEPTH` |
| 内存上限设为 1024 字节 | 10 / `MEMORY_LIMIT_EXCEEDED` |
| 输出目录不存在 | 11 / `OUTPUT_WRITE_ERROR` |

每个错误用例都断言：

- stderr 是合法 JSON，且 `error.code`/`error.name` 匹配。
- stdout 为空。
- 进程退出码等于错误码。
- 目标 PPM 与统计 JSON 均不存在。

另有一个回退用例把头部法线放大 3 倍并把所有 UV 固定为 `(0.25, 0.75)`；渲染成功，说明非单位法线归一化与 UV 退化切线回退均可工作。

还有一个视口/阴影分辨率一致性用例：把视口改为 `(16,12)-(256,212)`、阴影贴图改为 256。输出仍为 320x320，统计中的 `viewport` 与 `shadowMapSize` 均与场景修改一致，阴影采样仍非零。

## 画面检查

将 PPM 临时转换为 PNG 后目视检查，确认：

- 角色头部/躯干轮廓与背面剔除正常。
- 半透明护罩可见，并在躯干前后产生混合。
- 法线贴图产生高频明暗变化。
- 青色、洋红、橙色点光在地面与角色上形成不同方向照明。
- 定向光在地面产生阴影。
