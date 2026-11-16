# 实际验证记录

验证日期：2026-11-16（Asia/Shanghai）

## 环境

- Python：3.14.4
- NumPy：2.5.5
- PyOpenGL：3.1.10
- PyGLM：2.8.3
- Pillow：13.0.0
- GL 上下文：`4.5 (Core Profile) Mesa 26.0.8-1ubuntu0.3`
- GLSL：`4.50`
- 渲染器：`llvmpipe (LLVM 21.1.8, 256 bits)`，Mesa 软件后备
- 平台：Linux，EGL surfaceless，无窗口/X server 依赖

## 端到端基线

场景：`examples/default_scene.json`

- 400x300 输出，SSAA 2x，内部 800x600
- 1 个方向光、1 个点光、1 个聚光灯，均开启阴影
- 方向光/聚光阴影 512x512，点光立方体阴影 6 x 256x256
- 5 个网格节点、5 个材质、32x32 内联切线空间法线贴图
- ACES 色调映射，曝光 1.05
- 输出 9 张 PNG 与 `report.json`

命令：

```bash
.venv/bin/deferred-validation generate -o examples/default_scene.json
.venv/bin/deferred-validation render \
  --scene examples/default_scene.json -o artifacts/baseline
```

报告自检结果：

| 检查 | 实测 | 限制 |
|---|---:|---:|
| 直接光 + 间接光 vs 合成 HDR 最大绝对误差 | `1.9073e-06` | `<= 0.02` |
| CPU 色调映射重建 vs 最终 PNG 最大绝对误差 | `0.0019609` | `<= 0.0098039` |
| 可见几何覆盖率 | `0.558792` | `>= 0.01` |
| 报告状态 | `pass` | `pass` |

额外指标：

- `direct + indirect` 与合成通道平均绝对误差：`8.38e-09`
- 直接光/间接光不同像素占比（`>0.01`）：`0.647778`
- 最终图像亮度均值/方差：`0.404567 / 0.150178`
- 直接光 HDR 亮度均值/最大值：`0.195146 / 28.207183`
- 间接光 HDR 亮度均值/最大值：`0.049671 / 0.110069`

场景校验和：

```text
geometry        4d1bf7b6cd76b33f1c5859c98b88b68f72781e791d57ebaaadf1bfcf3ff01ee9
materials       420e9c9a8a16f78dbd55d34e55a78a639b3aba9f0ca9e7824cd425ffe80fd7e2
lighting        031f69219ed296b60f7c4d92579f3c01ac48955ab0dc870159f5b871050f2fad
render_settings 385dcfaba5d34b3cb7c2ecb488b3d11fdc687c8f8e803be86ea733a9db201b9e
scene           76b51d353cd6d0a606c2897fb79ce6f269f356df76e5a4d5af29ec53a8aabe26
```

## 验证模式

同场景复核：

```bash
.venv/bin/deferred-validation verify \
  --scene examples/default_scene.json \
  --report artifacts/baseline/report.json \
  -o artifacts/verify_same \
  --json-out artifacts/verify_same_summary.json
```

- 进程退出码：`0`
- 分类：`identical`
- 比较 PNG：9
- 最大像素差：`0.0`

修改方向光强度为原来的 1.2 倍后复核：

```bash
.venv/bin/deferred-validation verify \
  --scene /tmp/changed_scene_final.json \
  --report artifacts/baseline/report.json \
  -o artifacts/verify_changed \
  --json-out artifacts/verify_changed_summary.json
```

- 进程退出码：`3`
- 分类：`substantive`
- `lighting` 与 `scene` 校验和变化
- `direct.png`、`final.png` 超过像素阈值
- 最大像素差：`0.054902`
- stderr 输出具体校验和、变化通道及阈值

缺失输入：

```bash
.venv/bin/deferred-validation render \
  --scene /tmp/definitely_missing.json -o artifacts/nope
```

- 进程退出码：`2`
- stderr：`configuration error: /tmp/definitely_missing.json: scene file does not exist`

## 自动化测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：`Ran 11 tests ... OK`

覆盖内容：

- 合法场景加载、层级变换、5 种材质、3 类光源、内联法线贴图
- 未知材质、越界粗糙度、缺失相机、非法 JSON 的带路径错误
- 完整 G-buffer/光照/合成/色调映射渲染及 9 个报告产物
- 报告像素统计、校验和、G-buffer 布局契约
- 同一渲染器从 SSAA 1 切换到 SSAA 2，资源代数 `1 -> 2`，内部宽度 `128 -> 256`
- 验证模式同场景退出码 `0`
- 光照参数变化退出码 `3`
- 单像素 1 LSB 变化、低于阈值时归类为 `floating_point_noise`

## 产物

- `artifacts/baseline/`：最终基线 9 张 PNG 和 `report.json`
- `artifacts/verify_same_summary.json`：同场景验证汇总
- `artifacts/verify_changed_summary.json`：参数变更验证汇总
- `artifacts/verify_same_stderr.txt`、`artifacts/verify_changed_stderr.txt`：实际 stderr
- `artifacts/invalid_stderr.txt`：缺失输入错误信息
