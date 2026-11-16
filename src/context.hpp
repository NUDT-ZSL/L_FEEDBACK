#pragma once

#include "types.hpp"

#include <memory>
#include <string>

namespace mv {

class GlContext {
public:
    static std::unique_ptr<GlContext> create();
    ~GlContext();

    GlContext(const GlContext&) = delete;
    GlContext& operator=(const GlContext&) = delete;

    const ContextInfo& info() const;
    const std::string& signature() const;

    struct Impl;

private:
    explicit GlContext(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace mv
