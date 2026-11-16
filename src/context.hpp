#pragma once

#include <string>

#include "errors.hpp"

namespace orv {

class GlContext {
 public:
  GlContext() = default;
  ~GlContext();

  GlContext(const GlContext&) = delete;
  GlContext& operator=(const GlContext&) = delete;

  bool initialize(AppError& error);
  const std::string& description() const { return description_; }

 private:
  void shutdown();
  std::string description_;

#ifdef _WIN32
  void* window_ = nullptr;
  bool glfwInitialized_ = false;
#else
  void* display_ = nullptr;
  void* surface_ = nullptr;
  void* context_ = nullptr;
  void* eglApi_ = nullptr;
  void* libEgl_ = nullptr;
  void* libGl_ = nullptr;
#endif
};

}  // namespace orv
