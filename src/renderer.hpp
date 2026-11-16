#pragma once

#include <functional>
#include <memory>

#include "cli.hpp"
#include "output.hpp"
#include "scene.hpp"

namespace orv {

struct RenderOptions {
  int width = 640;
  int height = 360;
  int shadowSize = 1024;
  float shadowBias = 0.002f;
  RenderPass pass = RenderPass::Full;
};

struct RenderOutput {
  Image8 image;
  RenderStats stats;
};

class Renderer {
 public:
  using InterruptCheck = std::function<bool()>;

  Renderer(const Scene& scene, const RenderOptions& options,
           InterruptCheck interruptCheck);
  ~Renderer();

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  RenderOutput render();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace orv
