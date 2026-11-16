# 技术说明

## 1. 模块边界

项目分成三层：

- `lcr_core`：不包含 OpenGL、不读取系统随机源，负责所有需要确定性复现的数据和状态。
- `character_headless`：只链接核心库，用于 CI、回归测试和状态快照。
- `character_demo`：Windows OpenGL 3.3 前端。GPU 资源只在启动时创建，运行时只更新 uniform、纹理绑定、骨骼矩阵和披风索引缓冲。

可选 `character_offscreen` 使用 OSMesa 创建真实 OpenGL 上下文，用于没有窗口系统时验证 GLSL 和像素输出。

## 2. 确定性数据契约

### 2.1 随机数

`DeterministicRng` 使用显式 SplitMix64，不使用 `std::uniform_*_distribution`。整数取模使用拒绝采样，浮点随机值固定取 24 位尾数。这样不会因标准库实现差异产生不同随机流。

### 2.2 规范化序列化

`CanonicalWriter` 固定为：

- 小端 `u32/u64`；
- IEEE-754 float 的 32 位位模式；
- 字符串前写 32 位长度；
- 所有数组先写元素数量。

骨骼哈希覆盖骨骼名、父索引、绑定局部变换和逆绑定矩阵。网格哈希覆盖每个 part 的名称、材质槽、透明度、全部顶点字段和索引。同一 seed 在同一构建下重复启动会生成完全相同的字节。

测试同时比较：

- `skeleton.canonicalBytes()`；
- `meshCanonicalBytes(parts)`；
- 三个公开数据哈希；
- 不同 seed 必须产生不同骨骼和网格哈希。

### 2.3 骨骼

骨骼索引用固定枚举，共 22 根：

```text
Root
  Pelvis
    Spine
      Chest
        Neck
          Head
        LeftUpperArm -> LeftLowerArm -> LeftHand
        RightUpperArm -> RightLowerArm -> RightHand -> WeaponMount
        CapeRoot -> CapeMid -> CapeTip
    LeftThigh -> LeftShin -> LeftFoot
    RightThigh -> RightShin -> RightFoot
```

种子会给每根非根骨骼的局部平移和 yaw 添加小幅确定性扰动。逆绑定矩阵在绑定姿态世界矩阵上求刚性逆。

## 3. 网格与蒙皮

- 躯干：椭圆截面分层管。
- 四肢：两段锥形管，分别跨大臂/小臂和手，或大腿/小腿/脚。
- 头部：管状颈部 + UV 球。
- 武器：4 个盒体组成的剑刃、护手和手柄，100% 绑定 `WeaponMount`。
- 披风：24×16 网格，权重在 `CapeRoot/CapeMid/CapeTip` 间按距离混合。

每个权重集合由确定性骨骼选择、参数混合和最多 0.025 的确定性扰动生成，随后归一化。启动验证保证：

- 每个 joint 在 `[0, 22)`；
- 每个权重有限且非负；
- 四权重和与 1 的误差不超过 `1e-3`；
- 所有索引小于顶点数；
- 顶点位置/法线有限。

任一检查失败都会抛出 `E_BONE_INDEX` 或 `E_INTERNAL_INVARIANT`，不会跳过坏顶点继续渲染。

## 4. 固定步进与姿态插值

逻辑频率固定为 `kFixedLogicHz = 60`。`Simulation::advanceBy(dt)` 只通过固定步消耗累计时间；单帧追赶上限为 0.25 秒，防止断点或窗口拖动造成无限追帧。

渲染不直接使用任意时间生成的“游戏状态”。动画局部姿态按逻辑 tick 采样：

```text
tick   = floor(renderTime * 60)
alpha  = renderTime * 60 - tick
pose   = slerp/lerp(sampleTick(tick), sampleTick(tick + 1), alpha)
```

平移线性插值，旋转单位四元数 slerp，缩放线性插值。装备状态与动画姿态完全分离，因此切换装备不会参与姿态计算。

验证时以连续的解析动画 `animator.sample(t)` 作为参考：

- `without_interpolation` 使用 `floor(t * 60)` 的保持姿态；
- `with_interpolation` 使用相邻逻辑样本插值；
- 误差为所有关节点世界位置的最大欧氏距离与 RMS。

## 5. 装备与 GPU 管线

两套装备均在启动时生成 CPU 贴图，并上传到两个独立 GL 纹理对象。装备切换执行：

1. 切换 `EquipmentCatalog` 的活动索引；
2. 下一帧绑定该装备的 body/weapon 纹理；
3. 更新基础色、金属度、粗糙度、自发光颜色/强度、透明度 uniform。

不执行 `glLinkProgram`、不创建 VAO/VBO/EBO、不修改顶点格式。因此：

- `shader_program_builds == 1`
- `vao_builds == 8`
- `texture_uploads == 4`
- `equipment_pipeline_rebuilds == 0`

`SwitchEvent` 在切换前后分别计算插值姿态哈希和所有关节点最大误差。正常结果为哈希相同、误差恰好为 `0`。

## 6. 光照

单个 GLSL 330 程序同时用于不透明角色与透明披风：

- 主光：暖色方向光，Blinn-Phong 漫反射与高光；
- 补光：低强度冷色方向光；
- 自发光：装备材质颜色 × 强度，并按贴图 alpha 保存的程序化发光遮罩调制；
- 轻微环境项只用于避免背光面完全丢失。

片元输出做 Reinhard 压缩和 gamma 2.2 编码。装备自发光改变只涉及 uniform，不改变 shader 源。

## 7. 半透明披风

渲染顺序固定：

1. 所有不透明 part 写入颜色和深度；
2. CPU 用当前插值姿态计算披风每个顶点的蒙皮后世界位置；
3. 计算每个披风三角形质心到相机的距离；
4. 按距离从远到近稳定排序，相同距离用三角形索引决定次序；
5. 展开为新的索引流，`glBufferSubData` 更新披风动态 EBO；
6. 开启 blending，关闭深度写入，保留深度测试，绘制披风；
7. 恢复深度写入。

不透明躯干和武器已经在深度缓冲中，所以披风与它们的交叉由逐片元深度测试处理；披风自身三角形则由 CPU 排序消除叠层次序抖动。着色器对双面披风翻转背面法线，并在 UV 四周使用：

```text
feather = smoothstep(0.018, 0.123, min(u, 1-u, v, 1-v))
```

修饰 alpha，形成柔化边缘。

验证器用射线与蒙皮后的躯干/武器三角形求交，分别统计同一样本中被躯干、被武器以及被任一不透明网格遮挡的披风三角形。最终样本同时存在前景三角形、躯干遮挡和武器遮挡，且 768 个三角形无排序违例。

## 8. 错误策略

所有致命错误使用 `AppError`，`main` 最外层统一输出：

```json
{"error":{"code":"E_...","exit_code":20,"stage":"...","message":"..."}}
```

异常路径在进入渲染循环前终止，不会用空纹理、无效 shader 或越界骨骼继续运行。`--fault` 用于可重复触发：

- `texture`：贴图生成函数立即失败；
- `shader`：窗口/离屏程序使用真实 GLSL 编译器编译带 `#error` 的源，无头程序走同一错误上报通道；
- `bone`：故意把一个网格顶点的 joint 改为 22，再执行完整验证。

## 9. 已知边界

- 规范化字节一致性保证同一 seed 在同一平台/编译器构建下逐字节一致；不同 C 数学库的三角函数末位可能不同，因此不承诺跨编译器浮点逐位一致。
- 披风逐三角形排序只能精确处理单一透明层；本项目只有一层披风。更多透明层需扩展为逐层排序。
- Windows 图形窗口需在目标 Windows 机器上运行；本工作区使用 OSMesa 4.5 core / llvmpipe 对同一渲染器完成真实 GL 3.3 着色器与像素验证。
