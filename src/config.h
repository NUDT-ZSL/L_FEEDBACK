#pragma once

constexpr int kMaxCascades = 3;
constexpr int kMaxPointLights = 2;
constexpr int kMaxDirectionalLights = 1;
constexpr int kPcssBlockerSamples = 16;
constexpr int kPcssFilterSamples = 32;
constexpr int kPcssTotalSamples = kPcssBlockerSamples + kPcssFilterSamples;
constexpr unsigned int kDefaultSeed = 0x5EED2026u;

