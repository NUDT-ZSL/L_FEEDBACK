#pragma once

#include <memory>
#include <string>

#include "cli.h"

namespace sr {

class GLContext {
public:
    virtual ~GLContext() = default;

    virtual void makeCurrent() = 0;
    virtual void swapBuffers() {}
    virtual std::string description() const = 0;
};

std::unique_ptr<GLContext> createGLContext(ContextKind kind);

} // namespace sr
