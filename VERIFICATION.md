# 实际验证记录

验证日期：2026-11-16  
工作区：`/home/liang/feedback/uid119`  
运行环境：WSL Ubuntu 26.04、Python 3.14.4、NumPy 2.5.4、ModernGL 5.13.0、Mesa llvmpipe OpenGL 4.5（EGL 无窗口）

目标平台是 Windows 10+ 的 WGL/EGL 无窗口上下文。当前执行环境不是原生 Windows，因此这里实际验证的是
Linux/EGL 路径；代码的上下文创建顺序包含 EGL -> WGL -> 自动回退。Windows 上应首先运行
`python -m material_validator check-gl` 确认驱动和 `glcontext` 可用。

## 环境检查

```text
$ python -m material_validator check-gl
{
  "backend": "egl",
  "renderer": "llvmpipe (LLVM 21.1.8, 256 bits)",
  "vendor": "Mesa",
  "version": "4.5 (Core Profile) Mesa 26.0.8-1ubuntu0.3"
}
```

## 自动化测试

```text
$ pytest -q
.............                                                    [100%]
13 passed in 1.00s
```

覆盖内容：

- 缺失版本时的 `1.0` 默认值、未知字段保留与警告。
- 混合开启但深度写入开启、G-Buffer 槽位顺序错误的拒绝原因。
- 同一材质跨通道 uniform 冲突的全部来源路径。
- 顶点和片元两个阶段的编译错误同时收集，单个材质失败不中断其他材质。
- 清单声明但未使用 uniform 的 `warn/error` 策略。
- 差异阈值、最大偏差通道/位置及四类互斥分类。
- G-Buffer 零法线回退、非单位法线和金属粗糙度越界统计。
- 有效样本端到端、并行度 1/4 的缓冲哈希一致性、重复报告一致性。
- 通道声明顺序反转后输出缓冲不变。
- 向量、矩阵、标量、布尔 uniform 实际进入着色器并改变 G-Buffer 数据。
- 有效、无效、透明混合三个样本的报告和退出码。
- CLI 失败时返回整数 `1` 并写出 `report.json`、`summary.txt`。

## 有效样本

```text
$ python -m material_validator --manifest examples/valid_manifest.json \
    --out outputs/valid --parallel 2 --render-parallel 2
PASSED: 2/2 materials compiled, 0 pixels over threshold
```

关键结果：

- 状态：`passed`
- 材质：2/2 编译成功，2/2 可渲染
- 场景：1/1 对比通过
- 对比像素：19,200
- 超阈值像素：0
- 全局最大绝对偏差：`3.5762786865234375e-7`，通道 `B`
- 零长度法线回退：54 个覆盖像素
- 非单位法线：0
- 深度预通道与 G-Buffer 深度不一致：0
- 金属度/粗糙度越界像素：0
- 唯一警告：`normal_fallback_applied`

对照 `--parallel 1 --render-parallel 1` 与 `--parallel 4 --render-parallel 4`，以下缓冲
SHA-256 全部一致：

```text
world_position       722a378e48912bdeadda19ba4f3bae1f206f1ecd97536cef5b84a1b1eef22fc8
world_normal         ce96ab72528ca86de62185b6f76f41adb73054f7bc70468371d8800293999767
albedo               7cfac75049b68393a47752c267fb01b34b14040064c9a2c63ac88957f994f014
metallic_roughness   362cfc99158cca7e3a77aaff845ff9b4ba10035cfef8b477894cbc1cea52779a
depth_prepass        1aeed084a1aaffdde2ab6594412b67d1ea1ac4fc846cbe654013734acd4030a0
depth_gbuffer        1aeed084a1aaffdde2ab6594412b67d1ea1ac4fc846cbe654013734acd4030a0
deferred             3a8c737e59f6208be2b9bf6a8fa80bb1b13d1824ed2c0a704cc896242b34fe52
forward              63ef811db4fe75e578437add894d1156300830a6171c4daf6b612d04e21759cb
difference           e913c23abcc45616b11426d9f0ae7cb5edc99a56229094c34b0398f2271d70ab
```

清单 SHA-256：`4ce499761aa81a22d55137c0cbf273accb09b4d8d5b19c23a71077a2999bb79e`

## 无效样本

```text
$ python -m material_validator --manifest examples/invalid_manifest.json \
    --out outputs/invalid --parallel 3 --render-parallel 2
FAILED: 4/6 materials compiled, 0 pixels over threshold
```

结果：

- 退出码：`1`
- 6 个材质中 4 个编译成功、2 个通过装配且可渲染
- 9 个错误，1 个警告（未知字段）
- 汇总中的错误代码包括：
  `glsl_compile_error`、`active_uniform_not_declared`、`blend_with_depth_write`、
  `channel_uniform_conflict`、`metallic_out_of_range`、`roughness_out_of_range`
- 编译错误同时包含 `vertex` 与 `fragment` 阶段；错误行号、列号、原始驱动文本均在报告中
- `conflicted.u_base_color` 冲突列出基础材质与三个通道引用共四个来源路径
- 未知字段 `studio_note` 保留在 `unknown_fields`
- 越界材质仍以夹取后的数据完成对比，因此可以同时看到参数越界错误和渲染统计

## 透明混合样本

```text
$ python -m material_validator --manifest examples/blend_manifest.json \
    --out outputs/blend --parallel 2 --render-parallel 2
FAILED: 2/2 materials compiled, 497 pixels over threshold
```

结果：

- 退出码：`1`
- 19,200 个比较像素中 497 个超阈值
- 最大绝对偏差：`1.2810242772102356`，通道 `B`，位置 `x=76, y=58`
- `floating_blend_order`：461 个像素
- `unclassified`：36 个像素，最大偏差约 `2.38e-7`（极严相对阈值下的浮点噪声）
- 说明正向阶段确实应用了材质混合，而 G-Buffer 路径没有静默合成透明颜色

## 复现

```text
python -m material_validator generate-sample --out examples --kind valid
python -m material_validator --manifest examples/valid_manifest.json \
    --out outputs/valid --parallel 4 --render-parallel 4
pytest -q
```
