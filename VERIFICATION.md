# 实际验证记录

验证日期：2026-11-16  
环境：Python 3.14.4，Linux；依赖安装在 `.venv`  
样例输出：`artifacts/terrain`

## 默认命令

```bash
.venv/bin/python bake.py \
  --size 65 \
  --seed 20261116 \
  --lods 3 \
  --normal-resolution 512 \
  --preview-width 800 \
  --preview-height 600 \
  --output artifacts/terrain \
  --preview-backend auto
```

实际输出摘要：

```text
[samples] 65x65 seed=20261116 sha256=8386b4449072
[lod] L0=4225v/8192f, L1=2324v/4500f, L2=634v/1204f
[seam] L0->L1 102 tris, L1->L2 51 tris | T-junctions=0
[normal] 512x512 step=0.00195695
[preview] backend=cpu 800x600
[result] status=ok lods=3 artifacts=11
```

当前环境没有有效的 OpenGL 3.3 上下文。清单记录：

```json
{
  "requested": "auto",
  "attempted": ["opengl", "cpu"],
  "used": "cpu",
  "fallback_reason": "Attempt to retrieve context when no valid context"
}
```

## LOD 与接缝

| 级别 | 顶点 | 面 | 实际相对误差 | 阈值 | 流形 |
| --- | ---: | ---: | ---: | ---: | --- |
| LOD0 | 4225 | 8192 | 0 | 0 | 是 |
| LOD1 | 2324 | 4500 | 0.005887529272203268 | 0.0105 | 是 |
| LOD2 | 634 | 1204 | 0.02479048388793782 | 0.03 | 是 |

| 过渡 | 高侧边界顶点 | 低侧边界顶点 | 缝合三角形 | 高侧坐标复用 | T 型接缝 | 组合流形 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| LOD0 -> LOD1 | 65 | 39 | 102 | 逐位相等 | 0 | 是 |
| LOD1 -> LOD2 | 36 | 17 | 51 | 逐位相等 | 0 | 是 |

## 校验和与 schema

对 `bake_manifest.json` 中全部 11 个产物重新读取并计算 SHA-256，结果全部与清单一致。使用 `jsonschema` Draft 2020-12 校验成功清单和失败清单，均为 0 个 schema 错误；成功/失败清单的顶层字段集合一致。

法线贴图：

- 分辨率 `512x512`
- 像素步长 `1/511 = 0.0019569471624266144`
- 世界坐标约定 `X=cols, Z=rows, Y=up`
- 切线基 `T=+X, B=-Z, N=+Y`

## 测试

```text
Ran 12 tests in 0.960s
OK
```

测试实际覆盖并通过：

- 固定种子高度场/权重逐位确定性。
- 平坦高度场权重和有限且归一。
- 三级 LOD 严格减面、误差受限。
- 接缝高细节坐标复用、零 T 型接缝、组合网格流形。
- 法线图平坦基准、X/Z 梯度方向和分辨率缩放步长。
- 强制模拟 OpenGL 上下文缺失时的 CPU 回退记录。
- CLI 非法尺寸、NaN/Inf 高度场、权重和不匹配的业务退出码。
- 失败目录只包含 `bake_manifest.json`，无 OBJ/PNG/NPY 半成品。
- 成功清单产物、路径和 SHA-256 全量一致。

## 确定性抽检

使用相同 `--size 17 --seed 424242 --preview-backend cpu` 参数运行两次到不同目录，比较清单中全部 11 个产物的校验和：全部逐位一致，`source` 中的高度场与权重哈希也一致。

