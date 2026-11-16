# 验证记录

验证日期：2026-11-16  
当前环境：Ubuntu 26.04 WSL，GCC 15.2.0，CMake 3.30.5  
离屏图形：Mesa 25.1.7，OpenGL 4.5 Core Profile，llvmpipe (LLVM 20.1.8)

Windows 构建入口为 `build_windows.bat` / `CMakePresets.json`。当前执行环境不是 Windows，因此窗口程序未在 Windows 桌面会话启动；核心、GPU 上传、GLSL 编译和实际像素渲染统一通过 `character_offscreen` 在同一套 `GLRenderer` 代码上验证。

## 1. 核心测试

命令：

```bash
bash scripts/validate_local.sh
```

结果：

```text
determinism skeleton=add2fbad4bb6bfe1 mesh=32a200b4ecf70a56
texture catalog hash=87bcdd599f8f54f3
switch deltas=0,0
interpolation fps=45 max_hold=0.0102053 max_interp=3.15402e-05 ratio=0.00309056
interpolation fps=144 max_hold=0.0140304 max_interp=9.25816e-05 ratio=0.00659864
variable-rate state=a438d78bda29be92
cape crossing camera=4.5,1.8,-1.2 time=0
cape torso_occluded=121 weapon_occluded=195 order=valid
ALL CORE TESTS PASSED
fault texture -> rc=20 E_TEXTURE_GENERATION
fault shader -> rc=30 E_SHADER_COMPILE
fault bone -> rc=40 E_BONE_INDEX
```

确定性运行以 120 FPS 和 45 FPS 分别执行同一 360 个逻辑步，两个最终快照字节完全一致：

```text
a9b1f83cae2645ba8117344e22652b46f99cde6b0abafd0c69000bbc110b29dc  snapshot-a.json
a9b1f83cae2645ba8117344e22652b46f99cde6b0abafd0c69000bbc110b29dc  snapshot-b.json
```

## 2. 确定性数据

seed `1337`：

| 数据 | 哈希 |
| --- | --- |
| skeleton | `0b6bd8ea0b650af1` |
| mesh | `6495698dc1fc5c8e` |
| character | `743d69a03bfed51b` |
| equipment catalog | `a96958b02f68f2c6` |
| final state | `56830c5da803dee3` |

Iron texture：body `bdd04b19710b20f4`，weapon `73dc91b85db0ed9c`。  
Ember texture：body `f23ea94f9cf3ca73`，weapon `608ad018e439db99`。

完整结果见 `validation/validation-report.json` 和 `validation/state-snapshot.json`。

## 3. 插值误差

参考值为连续解析动画的所有关节点世界坐标。误差单位为场景米。

| 渲染 FPS | 无插值最大误差 | 插值后最大误差 | 最大误差比 |
| ---: | ---: | ---: | ---: |
| 45 | 0.00772096403 | 0.000346708111 | 0.0449047685 |
| 144 | 0.0106138391 | 0.000433905952 | 0.0408811495 |

RMS 误差：

| 渲染 FPS | 无插值 | 插值后 | 比值 |
| ---: | ---: | ---: | ---: |
| 45 | 0.00126944389 | 0.00000894130608 | 0.00704348274 |
| 144 | 0.00160193746 | 0.00000787493354 | 0.00491588097 |

## 4. 装备切换连续性

切换时刻 `1.25 s`，tick 75：

```text
iron -> ember
pose_hash_before = 258d224406ccb7e0
pose_hash_after  = 258d224406ccb7e0
max_joint_position_delta = 0
```

另有两个自动切换测试（0.5 s、1.5 s）同样得到两次 `0` 位置差。

## 5. 披风排序与交叉

一次 360 逻辑步运行包含 23 个披风采样。全部样本：

- 三角形数 768；
- 排序违例 0；
- 边缘 alpha 最小值 0、最大值 1。

最显著交叉样本：

```text
render_time = 3.85
triangles_in_front_of_opaque = 435
triangles_behind_opaque = 333
torso_occluded_triangles = 272
weapon_occluded_triangles = 189
order_violations = 0
```

独立核心测试遍历 seed 1337 的多个相机和姿态，进一步确认同一样本同时存在躯干遮挡 121 个、武器遮挡 195 个三角形。

## 6. 错误链路

无头故障注入：

```text
texture rc=20 {"error":{"code":"E_TEXTURE_GENERATION",...}}
shader  rc=30 {"error":{"code":"E_SHADER_COMPILE",...}}
bone    rc=40 {"error":{"code":"E_BONE_INDEX",...}}
```

真实 Mesa GLSL 编译失败：

```text
rc=30
{"error":{"code":"E_SHADER_COMPILE","exit_code":30,"stage":"shader.compile.character.vertex","message":"shader compilation failed:\n0:2(1): preprocessor error: #error forced shader compile failure\n..."}}
```

原始 stderr 保存在 `validation/shader-compile-fault.stderr`。

## 7. 实际离屏渲染

命令：

```bash
./build/offscreen/character_offscreen \
  --seed 1337 --equipment iron --texture-size 128 \
  --width 960 --height 640 \
  --report build/offscreen/reports/offscreen-report.json
```

结果：

```text
GL version:  4.5 (Core Profile) Mesa 25.1.7
GL renderer: llvmpipe (LLVM 20.1.8, 256 bits)
before hash: db6177eea1a6496b  (iron)
after hash:  0b7caf933ea0a3ff  (ember)
non_background_pixels: 44284 / 614400
changed_pixels: 44643
switch_max_joint_delta: 0
shader_program_builds: 1
vao_builds: 8
texture_uploads: 4
equipment_pipeline_rebuilds: 0
```

截图：

- Iron：`docs/images/offscreen-iron.png`
- Ember：`docs/images/offscreen-ember.png`

完整 JSON 见 `validation/offscreen-report.json`。

## 8. 结论

本地样本覆盖了角色渲染、两套装备贴图/自发光切换、姿态连续、固定步插值、半透明披风排序与交叉遮挡、确定性快照和三种错误退出。`character_demo` 目标也已使用固定本地 GLFW/GLAD/GLM 依赖完成编译和链接；Windows 专用窗口/上下文创建路径未在 Windows 桌面会话启动。实际 Windows 运行仍需在目标 Windows 10/11 + OpenGL 3.3 驱动环境执行 `build_windows.bat` 和 `run_demo.bat`。
