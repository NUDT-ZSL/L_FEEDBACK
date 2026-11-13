# L_FEEDBACK

# 离线地形瓦片渲染管线原型

完全离线、确定性可复现的地形瓦片渲染管线。程序化高度场来自自实现的
`splitmix64` 整数散列，固定世界范围四叉树划分，CPU 保守视锥裁剪、屏幕空间
误差 LOD、T 形接缝消除、邻域差分法线、Lambert 光照、gamma 校正，提供
ModernGL 与 NumPy 软件光栅化两个后端，并输出 PNG 与契约 JSON。

## 环境与依赖

目标环境（需求规定）：

```
Windows 10/11 64 位, Python 3.11
python -m pip install numpy==1.26.4 pillow==10.2.0 moderngl==0.11.0
```

本仓库实际验证环境为 Linux + Python 3.14，使用该解释器可用的兼容版本
（numpy 2.x / pillow 13.x / moderngl 5.x）。代码只使用稳定的公开 API，
在 Python 3.11 与上述锁定版本上同样可运行；不访问网络、不读取外部数据、
不使用容器，所有样本与相机配置均由代码生成。

## 目录结构

```
terrain/
  config.py        类型化配置与 JSON 读写、参数校验（业务拒绝）
  errors.py        业务码与 PipelineError
  gen_config.py    代码生成确定性样本配置 configs/*.json
  heightfield.py   splitmix64、值噪声高度场、任意点/向量化采样
  quadtree.py      瓦片键、AABB、四邻域/相邻叶集合查找
  camera.py        float64 视图/投影矩阵、六平面视锥与保守 AABB 测试
  lod.py           SSE LOD、每级裁剪计数、受限四叉树、高度界缓存
  mesh.py          网格构建、共享边接缝顶点插入、扇形三角化、校验/回滚
  normals.py       一环外扩邻域差分法线、Lambert、gamma
  software_backend.py  NumPy 软件光栅化（近裁剪、背面剔除、深度、超采样）
  gl_backend.py    ModernGL 光栅化，失败映射 E_NO_GL 且释放全部 GL 资源
  pipeline.py      端到端编排、契约 JSON、PNG 输出
  main.py          CLI 入口与退出码约定
verify_all.py      五类契约回归（自包含）
configs/           生成的默认配置与关闭接缝配置
artifacts/         PNG 与契约 JSON 输出
```

## 运行

```
python -m terrain.gen_config
python -m terrain.main --config configs/default.json
python verify_all.py
```

切换后端只改配置中的 `backend`（`software` 或 `moderngl`）；关闭接缝只改
`enable_stitching`，无需改源码。

## 关键设计

- 确定性：散列全程 `numpy.uint64` 回绕运算，固定遍历顺序，单线程，无随机数；
  几何与法线用 float64，仅光栅化帧缓冲用 float32。
- 保守裁剪：六平面 Gribb-Hartmann 提取并归一化，逐平面取 p-顶点，任一平面
  严格在外才剔除，跨边界保留，杜绝假阴性。
- SSE LOD：`几何误差 * 焦距 * 屏高 / 距离`，以阈值的尺度相关相对容差比较；
  受限四叉树保证相邻叶层级差不超过 1。
- 接缝消除：粗瓦片把细瓦片共享边上的顶点逐点复制进边界单元并做凸扇形三角化，
  两侧遍历同一组世界坐标点；修补后校验索引、退化三角形、悬挂顶点，异常时
  回滚为无修补规则网格并在契约中标记，禁止半修补网格。
- 双后端共享全部网格/LOD/接缝逻辑，仅替换光栅化；软件端含近平面裁剪、
  背面剔除、线性深度测试与 2x2 超采样。

## 退出码与业务码

- 成功：退出码 0。
- 业务拒绝：退出码 0，业务码写 stderr，包括
  `E_RES_TOO_HIGH`（分辨率 > 2048，分配前拒绝）、
  `E_BAD_FRUSTUM`（近平面 <= 0 等）、
  `E_NO_GL`（GL 不可用，自动回退软件后端）、
  `E_MESH_PATCH`（网格修补异常，内部回滚）。
- 未预期异常：非零退出码并输出回溯。

## 实际验证结果

在本机 Linux/Python 3.14 环境实测：

- `python verify_all.py` 五类检查全部 `[PASS]`，退出码 0。
- `python verify_all.py configs/no_stitch.json` 中 `stitch_consistency`
  按预期 `[FAIL]` 并打印 `FAILED_CHECKS=stitch_consistency`，退出码 1。
- 软件与 ModernGL 后端像素一致率约 99%（容差 <=6 灰度），三角形数相同。
- 重复运行 PNG 字节哈希与契约（除计时字段）完全一致。
- GL 上下文/着色器失败路径上下文释放后无存活对象，回退软件后端。

详见 `artifacts/contract.json`。
