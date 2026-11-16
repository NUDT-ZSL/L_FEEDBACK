#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mv {

enum class ExitCode : int {
    Success = 0,
    Input = 2,
    ShaderCompile = 3,
    ShaderLink = 4,
    Framebuffer = 5,
    Attachment = 6,
    Validation = 7,
    Runtime = 8,
    Cache = 9,
    Output = 10,
};

enum class AttachmentKind {
    Color,
    Depth,
    Stencil,
};

enum class PixelFormat {
    RGBA8,
    RGBA16F,
    Depth24Stencil8AsColor,
};

struct AttachmentSpec {
    std::string name;
    int location = 0;
    std::string format;
    PixelFormat pixelFormat = PixelFormat::RGBA8;
    std::array<float, 4> clear{{0.0f, 0.0f, 0.0f, 0.0f}};
};

struct DepthStencilSpec {
    bool enabled = false;
    std::string format = "D24S8";
    float clearDepth = 1.0f;
    int clearStencil = 0;
    bool depthTest = true;
    bool depthWrite = true;
    std::string depthFunction = "less";
    bool stencilTest = false;
    std::string stencilFunction = "always";
    int stencilReference = 0;
    int stencilReadMask = 0xff;
    int stencilWriteMask = 0xff;
    std::string stencilFail = "keep";
    std::string stencilDepthFail = "keep";
    std::string stencilDepthPass = "keep";
};

struct GeometrySpec {
    std::string mode = "fullscreen_triangle";
    int vertices = 3;
    int instances = 1;
};

struct VariantSpec {
    std::string name;
    std::vector<std::string> macros;
};

struct TextureSpec {
    std::string name;
    std::string uniform;
    int width = 8;
    int height = 8;
    std::uint32_t seed = 1;
    int binding = 0;
};

struct ConstantBufferSpec {
    std::string blockName = "DepConstants";
    int binding = 0;
    std::uint32_t seed = 0;
    std::vector<std::string> layout;
    std::map<std::string, std::array<float, 4>> values;
};

struct RuleSpec {
    std::string id;
    std::string type;
    std::string source;
    std::string target;
    std::string attachment;
    std::string texture;
    std::string constant;
    std::string sourceChannel;
    std::string targetChannel;
    std::string channels;
    std::string attachmentChannel;
    std::string sampling = "texel_fetch_mod";
    int component = 0;
    float expected = 0.0f;
    float tolerance = 0.0f;
};

struct MaterialSpec {
    std::string name;
    std::string metadataPath;
    std::string sourceDir;
    std::string vertexPath;
    std::string fragmentPath;
    std::string vertexSource;
    std::string fragmentSource;
    int width = 64;
    int height = 48;
    GeometrySpec geometry;
    DepthStencilSpec depthStencil;
    std::vector<AttachmentSpec> attachments;
    std::map<std::string, VariantSpec> variants;
    std::vector<std::string> defaultVariants;
    std::vector<TextureSpec> textures;
    std::optional<ConstantBufferSpec> constants;
    std::vector<RuleSpec> rules;
    int maxDiffPixels = 4096;
    std::string metadataSha256;
};

struct VariantRequest {
    std::string name;
    std::vector<std::string> macros;
};

struct CommandLine {
    std::string materialPath;
    std::string variants = "";
    std::string cacheDir;
    std::string outputPath;
    std::string pixelDir;
    std::optional<std::pair<int, int>> textureSize;
    std::map<std::string, std::array<float, 4>> constantOverrides;
    int maxDiffs = -1;
    bool deterministic = false;
    bool showHelp = false;
};

struct MaterialError {
    ExitCode exitCode = ExitCode::Runtime;
    std::string code;
    std::string stage;
    std::string message;
    std::string shader;
    std::string variant;
    std::string log;
    int line = -1;
    std::string detail;
};

struct CliResult {
    std::optional<CommandLine> command;
    std::optional<MaterialError> error;
};

struct ContextInfo {
    std::string api;
    std::string version;
    std::string vendor;
    std::string renderer;
    std::string glslVersion;
};

}  // namespace mv
