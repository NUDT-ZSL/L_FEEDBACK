#pragma once

#include "types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mv {

class GlContext;

struct RawAttachment {
    std::string name;
    std::string kind;
    std::string format;
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> raw;
    std::vector<float> rgba;
    std::vector<float> depth;
    std::vector<std::uint8_t> stencil;
    std::string sha256;
    std::string dumpedPath;
};

struct PixelDifference {
    int x = 0;
    int y = 0;
    std::vector<float> expected;
    std::vector<float> actual;
    float delta = 0.0f;
    std::string source;
    std::string target;
};

struct RuleResult {
    std::string id;
    std::string type;
    bool passed = true;
    std::uint64_t testedPixels = 0;
    std::uint64_t passedPixels = 0;
    std::uint64_t failedPixels = 0;
    bool differencesTruncated = false;
    std::vector<PixelDifference> differences;
};

struct VariantResult {
    std::string name;
    std::vector<std::string> macros;
    std::string cacheKey;
    bool cacheHit = false;
    std::string compileLog;
    std::string linkLog;
    std::vector<std::string> warnings;
    std::vector<RawAttachment> attachments;
    std::vector<RuleResult> rules;
    std::uint64_t pixelCount = 0;
    std::uint64_t passedPixels = 0;
    std::uint64_t failedPixels = 0;
    double durationMs = 0.0;
};

class Renderer {
public:
    explicit Renderer(const GlContext& context);
    VariantResult render(const MaterialSpec& material,
                         const VariantRequest& variant,
                         const CommandLine& command,
                         const std::string& pixelRoot);

private:
    const GlContext& context_;
};

}  // namespace mv
