# Technical Design

## Pipeline

1. Load and validate JSON. Global errors stop initialization with exit code `2`; per-material errors are attached to that group and isolated.
2. Create an OpenGL 4.3 core context. Linux uses EGL surfaceless first, then an optional hidden GLFW window; Windows uses hidden GLFW.
3. Compile the two shader programs (`sss-lut`, `surface`) through a program cache.
4. Render the RGB16F SSS LUT into a cached offscreen target.
5. For each valid group:
   - Bind the cached `RGBA8 + DEPTH24` render target and clear it.
   - Set camera/light/environment/tone mapping uniforms once.
   - Draw the primary sphere and reference sphere, changing only material uniforms.
   - Read back RGBA8, drop alpha, flip OpenGL's bottom-up rows into top-down RGB8.
   - Compare an optional baseline, reserve a collision-free filename, and atomically write PPM.
6. On failure, log group/error code, skip that group, and continue.
7. Release every GL object while the context is still current, then emit created/released counters.

## SSS LUT

The LUT is `profileSamples x curvatureSamples`, stored as `GL_RGBA16F`, linear filtered and clamped. Every texel integrates `scatteringSamples` samples of the diffusion kernel.

For normalized x coordinate `u` and y coordinate `v`:

```text
NdotL        = 2u - 1
curvature    = mix(curvatureMin, curvatureMax, v)
for i in [0, scatteringSamples):
  spread        = 2 * ((i + 0.5) / scatteringSamples) - 1
  localNdotL    = clamp(NdotL + spread * curvature * 2, -1, 1)
  wrappedNdotL  = clamp((localNdotL + wrap) / (1 + wrap), epsilon, 1)
  phase         = wrappedNdotL ^ profilePower
  transmittance = exp(-scatterRadius * (1 / wrappedNdotL - 1) * 8)
  weight        = exp(-abs(spread) * 1.5)
  integral     += phase * (0.25 + 0.75 * transmittance) * weight
profile      = scatterColor * integral / weightSum * mix(0.70, 1.0, v)
```

The surface shader uses exactly the same mapping:

```text
u = clamp(NdotL * 0.5 + 0.5, 0, 1)
v = clamp((materialCurvature - curvatureMin)
          / (curvatureMax - curvatureMin), 0, 1)
diffuseProfile = texture(sssLut, vec2(u, v)).rgb
```

Both generation and sampling receive `curvatureMin`/`curvatureMax` from the same `RenderConfig`; the LUT width is the configured `profile_samples`, while `scattering_samples` controls the per-texel integration count. This prevents generation/sampling parameter drift.

## Anisotropic GGX

Tangent basis:

```text
dp1 = dFdx(worldPosition), dp2 = dFdy(worldPosition)
duv1 = dFdx(uv),           duv2 = dFdy(uv)
T = normalize((dp1 * duv2.y - dp2 * duv1.y) / determinant)
T = normalize(T - N * dot(N, T))
B = normalize(cross(N, T))
```

`anisotropy_rotation` rotates `T`/`B` around `N`. The roughness/anisotropy mapping is:

```text
aspect = sqrt(1 - 0.9 * anisotropy)
alphaT = roughness^2 / aspect
alphaB = roughness^2 * aspect
```

The shader evaluates anisotropic GGX `D`, separable Smith masking and Schlick Fresnel for each area-light sample. Direct diffuse adds the area-light radiance to the LUT profile. A hemisphere ambient term is added in the same linear space. ACES (default), Reinhard, or none is applied once, followed by gamma correction.

## Area Light

The rectangular emitter has a CPU-built orthonormal basis. A fixed `samples_x * samples_y` grid is jittered with a deterministic hash of sample index and `seed`. Each sample contributes:

```text
deltaArea = width * height / sampleCount
Li        = lightColor * intensity * emitterCos * deltaArea / distance^2
```

Only emitter-facing samples contribute. No shadow ray is traced; the scene intentionally limits itself to two non-overlapping spheres.

## Resource Reuse and Failure Isolation

- `FramebufferCache` key: `width x height : colorFormat : depthFlag`. The LUT target and final target are each created once.
- Color textures and depth renderbuffers are owned by the framebuffer cache and reused with their FBO.
- `ProgramCache` compiles each shader pipeline once and caches uniform locations.
- Sphere mesh/VAO/buffers and the fullscreen VAO are created once.
- A material switch only changes uniforms, then issues two draw calls.
- Per-group errors throw `GroupFailure`; the outer loop records the group name/code and continues.
- GL errors are drained before each group, then checked after draw/readback. A failed group never reaches PPM finalization.
- Resource counters are logged as DEBUG lines in `render.log`; all normal output is file-based.

## Determinism

- No wall-clock, random device, or unordered iteration affects pixels.
- Area-light jitter and seed are fixed by config.
- FBOs are cleared before each group, smoothing is not requested, dithering and sRGB framebuffer conversion are disabled.
- PPM bytes are formed from the exact RGB8 readback with a fixed header and row order.
- Same executable, driver, device and config produced byte-identical PPMs in validation.

Bit-level identity across different GPUs/vendors is not claimed; the requirement is verified on the local deterministic sample environment.

## Logging

`render.log` lines contain an ISO-8601 local timestamp, level, group name and message:

```text
2026-11-17T05:28:25+0800 [STAT] group="skin_satin" schema=1 baseline="..." ...
2026-11-17T05:28:25+0800 [ERROR] group="bad_roughness" code=111 message=...
2026-11-17T05:28:25+0800 [DEBUG] group="-" resource framebuffers created=2 released=2 balanced=yes
```

`diff_stats.json` uses `stats_schema_version: 1` and records baseline version, per-image FNV-1a fingerprints and comparison statistics.

## Validation Matrix

| Check | Expected | Observed |
|---|---|---|
| Valid demo batch | exit `0`, 2 successes | `success=2 failed=0 exit=0` |
| Invalid roughness | group code `111`, continue | group failed; later groups rendered |
| Invalid anisotropy | group code `112`, continue | group failed; later groups rendered |
| Duplicate names | `name.ppm`, `name_001.ppm` | both valid PPMs created |
| Existing output | never overwritten | sentinel unchanged; suffix file created |
| Missing baseline | warning, group success | warning in log; PPM created |
| Self-baseline comparison | MSE/max = 0 | `mse=0 max_abs=0` |
| Two runs, same config | byte-identical PPMs | SHA-256 matched |
| Default dimensions | `512x512` | header matched |
| Resource release | created == released | all counters balanced |
| Init error | exit `2`, no PPM | `success=0 failed=0 exit=2` |

