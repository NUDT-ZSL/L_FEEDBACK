#pragma once

#include <memory>
#include <string>

namespace pbr {

class GLContext {
public:
    GLContext();
    ~GLContext();
    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;

    bool initialize(int width, int height, std::string& error);
    std::string rendererSummary() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pbr
