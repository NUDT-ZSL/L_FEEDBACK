#include "renderer.hpp"

#include "context.hpp"
#include "util.hpp"
#include "validation.hpp"

#include <glad/gl.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mv {
namespace {

using json = nlohmann::json;

struct ShaderHandle {
    GLuint id = 0;
    ShaderHandle() = default;
    explicit ShaderHandle(GLenum type) : id(glCreateShader(type)) {}
    ~ShaderHandle() {
        if (id != 0) {
            glDeleteShader(id);
        }
    }
    ShaderHandle(const ShaderHandle&) = delete;
    ShaderHandle& operator=(const ShaderHandle&) = delete;
    ShaderHandle(ShaderHandle&& other) noexcept : id(other.id) { other.id = 0; }
    ShaderHandle& operator=(ShaderHandle&& other) noexcept {
        if (this != &other) {
            if (id != 0) {
                glDeleteShader(id);
            }
            id = other.id;
            other.id = 0;
        }
        return *this;
    }
};

struct ProgramHandle {
    GLuint id = 0;
    ProgramHandle() = default;
    ~ProgramHandle() {
        if (id != 0) {
            glDeleteProgram(id);
        }
    }
    ProgramHandle(const ProgramHandle&) = delete;
    ProgramHandle& operator=(const ProgramHandle&) = delete;
    ProgramHandle(ProgramHandle&& other) noexcept : id(other.id) { other.id = 0; }
    ProgramHandle& operator=(ProgramHandle&& other) noexcept {
        if (this != &other) {
            if (id != 0) {
                glDeleteProgram(id);
            }
            id = other.id;
            other.id = 0;
        }
        return *this;
    }
};

struct GlResources {
    std::vector<GLuint> textures;
    std::vector<GLuint> buffers;
    std::vector<GLuint> vaos;
    GLuint framebuffer = 0;

    ~GlResources() {
        if (!textures.empty()) {
            glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
        }
        if (!buffers.empty()) {
            glDeleteBuffers(static_cast<GLsizei>(buffers.size()), buffers.data());
        }
        if (!vaos.empty()) {
            glDeleteVertexArrays(static_cast<GLsizei>(vaos.size()), vaos.data());
        }
        if (framebuffer != 0) {
            glDeleteFramebuffers(1, &framebuffer);
        }
    }
};

struct ColorFormat {
    GLenum internal = GL_RGBA8;
    GLenum external = GL_RGBA;
    GLenum type = GL_UNSIGNED_BYTE;
};

ColorFormat colorFormat(const AttachmentSpec& attachment) {
    switch (attachment.pixelFormat) {
        case PixelFormat::RGBA8:
            return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};
        case PixelFormat::RGBA16F:
            return {GL_RGBA16F, GL_RGBA, GL_FLOAT};
        case PixelFormat::Depth24Stencil8AsColor:
            return {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8};
    }
    return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};
}

std::string glErrorText(GLenum error) {
    switch (error) {
        case GL_INVALID_ENUM:
            return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE:
            return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION:
            return "GL_INVALID_OPERATION";
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_OUT_OF_MEMORY:
            return "GL_OUT_OF_MEMORY";
        case GL_NO_ERROR:
            return "GL_NO_ERROR";
        default:
            return "GL_ERROR_0x" + std::to_string(error);
    }
}

void drainGlErrors() {
    while (glGetError() != GL_NO_ERROR) {
    }
}

void throwGlError(const char* code, const char* stage, const char* context) {
    const auto error = glGetError();
    if (error != GL_NO_ERROR) {
        throw makeError(ExitCode::Attachment, code, stage,
                        std::string(context) + ": " + glErrorText(error));
    }
}

std::string framebufferStatusText(GLenum status) {
    switch (status) {
        case GL_FRAMEBUFFER_COMPLETE:
            return "GL_FRAMEBUFFER_COMPLETE";
        case GL_FRAMEBUFFER_UNDEFINED:
            return "GL_FRAMEBUFFER_UNDEFINED";
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            return "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            return "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
        case GL_FRAMEBUFFER_UNSUPPORTED:
            return "GL_FRAMEBUFFER_UNSUPPORTED";
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            return "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
        default:
            return "GL_FRAMEBUFFER_STATUS_0x" + std::to_string(status);
    }
}

std::string expandSource(const std::string& source, const std::vector<std::string>& macros) {
    std::string result;
    std::size_t remainderOffset = 0;
    int nextOriginalLine = 1;
    const auto firstNewline = source.find('\n');
    const auto firstLine = source.substr(0, firstNewline);
    auto firstTrimmed = trim(firstLine);
    if (firstTrimmed.rfind("#version", 0) == 0) {
        result += firstLine;
        result += '\n';
        remainderOffset = firstNewline == std::string::npos ? source.size() : firstNewline + 1;
        nextOriginalLine = 2;
    }
    for (const auto& macro : macros) {
        result += "#define ";
        result += macro;
        result += '\n';
    }
    result += "#line ";
    result += std::to_string(nextOriginalLine);
    result += " 0\n";
    result += source.substr(remainderOffset);
    return result;
}

int lineFromLog(const std::string& log) {
    static const std::regex mesaPattern(R"((?:^|\s)\d+:(\d+))");
    static const std::regex nvidiaPattern(R"((?:^|\s)\d+\((\d+)\))");
    static const std::regex linePattern(R"(\bline\s+(\d+)\b)", std::regex::icase);
    std::smatch match;
    if (std::regex_search(log, match, mesaPattern)) {
        return std::stoi(match[1].str());
    }
    if (std::regex_search(log, match, nvidiaPattern)) {
        return std::stoi(match[1].str());
    }
    if (std::regex_search(log, match, linePattern)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

std::string sourceLine(const std::string& source, int line) {
    if (line <= 0) {
        return {};
    }
    std::size_t begin = 0;
    for (int current = 1; current < line; ++current) {
        const auto newline = source.find('\n', begin);
        if (newline == std::string::npos) {
            return {};
        }
        begin = newline + 1;
    }
    const auto newline = source.find('\n', begin);
    return source.substr(begin, newline == std::string::npos ? std::string::npos
                                                             : newline - begin);
}

ShaderHandle compileShader(GLenum type,
                           const std::string& source,
                           const std::vector<std::string>& macros,
                           const std::string& path,
                           const std::string& variant) {
    ShaderHandle shader(type);
    if (shader.id == 0) {
        throw makeError(ExitCode::Runtime, "GL_SHADER_ALLOCATION_FAILED", "shader_compile",
                        "glCreateShader failed");
    }
    const auto expanded = expandSource(source, macros);
    const char* text = expanded.c_str();
    const GLint length = static_cast<GLint>(expanded.size());
    glShaderSource(shader.id, 1, &text, &length);
    glCompileShader(shader.id);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader.id, GL_COMPILE_STATUS, &ok);
    if (ok == GL_TRUE) {
        return shader;
    }

    GLint logLength = 0;
    glGetShaderiv(shader.id, GL_INFO_LOG_LENGTH, &logLength);
    std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
    GLsizei written = 0;
    glGetShaderInfoLog(shader.id, logLength, &written, log.data());
    log.resize(static_cast<std::size_t>(std::max<GLsizei>(written, 0)));
    auto error = makeError(ExitCode::ShaderCompile, "SHADER_COMPILE_FAILED",
                           "shader_compile", "shader compilation failed");
    error.shader = path;
    error.variant = variant;
    error.line = lineFromLog(log);
    error.log = log;
    if (error.line > 0) {
        error.detail = sourceLine(source, error.line);
    }
    throw error;
}

void checkOutputContract(GLuint program, const MaterialSpec& material) {
    GLint outputCount = 0;
    glGetProgramInterfaceiv(program, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES, &outputCount);
    std::set<int> declaredLocations;
    for (const auto& attachment : material.attachments) {
        declaredLocations.insert(attachment.location);
    }
    bool foundOutput = false;
    for (GLint i = 0; i < outputCount; ++i) {
        const GLenum property = GL_LOCATION;
        GLint location = -1;
        glGetProgramResourceiv(program, GL_PROGRAM_OUTPUT, static_cast<GLuint>(i), 1,
                               &property, 1, nullptr, &location);
        if (location < 0) {
            continue;
        }
        foundOutput = true;
        if (declaredLocations.count(location) == 0) {
            throw makeError(
                ExitCode::Attachment, "ATTACHMENT_OUT_OF_BOUNDS", "attachment",
                "fragment output location " + std::to_string(location) +
                    " has no attachment declaration");
        }
    }
    if (!foundOutput) {
        throw makeError(ExitCode::Attachment, "ATTACHMENT_OUTPUT_MISSING", "attachment",
                        "linked fragment program has no explicit color outputs");
    }
}

std::string programCacheKey(const MaterialSpec& material,
                            const VariantRequest& variant,
                            const std::string& contextSignature) {
    std::vector<std::string> parts{
        "material-verify-program-cache-v1",
        contextSignature,
        Sha256::hex(material.vertexSource),
        Sha256::hex(material.fragmentSource),
    };
    for (const auto& macro : variant.macros) {
        parts.push_back("macro:" + macro);
    }
    for (const auto& attachment : material.attachments) {
        parts.push_back("attachment:" + std::to_string(attachment.location) + ":" +
                        attachment.format);
    }
    for (const auto& texture : material.textures) {
        parts.push_back("texture:" + texture.name + ":" + texture.uniform + ":" +
                        std::to_string(texture.width) + "x" +
                        std::to_string(texture.height) + ":" +
                        std::to_string(texture.seed) + ":" +
                        std::to_string(texture.binding));
    }
    if (material.constants) {
        parts.push_back("ubo:" + material.constants->blockName + ":" +
                        std::to_string(material.constants->binding) + ":" +
                        std::to_string(material.constants->seed));
        for (const auto& field : material.constants->layout) {
            const auto& value = material.constants->values.at(field);
            parts.push_back("constant:" + field + ":" + formatFloat(value[0]) + "," +
                            formatFloat(value[1]) + "," + formatFloat(value[2]) + "," +
                            formatFloat(value[3]));
        }
    }
    return sha256OfStrings(parts);
}

struct CacheLoadResult {
    ProgramHandle program;
    bool loaded = false;
    std::string warning;
};

CacheLoadResult tryLoadProgram(const MaterialSpec&,
                               const VariantRequest&,
                               const std::string& cacheKey,
                               const std::string& cacheDir,
                               const std::string& contextSignature) {
    CacheLoadResult result;
    if (cacheDir.empty()) {
        return result;
    }
    const auto root = std::filesystem::path(cacheDir) / "program-cache-v1" /
                      cacheKey.substr(0, 2) / cacheKey.substr(2);
    const auto binaryPath = root.string() + ".program";
    const auto metadataPath = root.string() + ".json";
    if (!std::filesystem::exists(binaryPath) || !std::filesystem::exists(metadataPath)) {
        return result;
    }
    try {
        const auto metadata = json::parse(readTextFile(metadataPath));
        if (metadata.value("key", "") != cacheKey ||
            metadata.value("gl_signature", "") != contextSignature) {
            result.warning = "cache metadata key mismatch; recompiling";
            return result;
        }
        const auto binary = readBinaryFile(binaryPath);
        if (binary.empty()) {
            result.warning = "cache program binary is empty; recompiling";
            return result;
        }
        Sha256 binaryDigest;
        binaryDigest.update(binary.data(), binary.size());
        if (metadata.value("binary_sha256", "") != binaryDigest.hexDigest()) {
            result.warning = "cache program binary checksum mismatch; recompiling";
            return result;
        }
        const auto format = metadata.at("binary_format").get<GLenum>();
        result.program.id = glCreateProgram();
        if (result.program.id == 0) {
            result.warning = "glCreateProgram failed while loading cache";
            return result;
        }
        glProgramBinary(result.program.id, format, binary.data(),
                        static_cast<GLsizei>(binary.size()));
        GLint linked = GL_FALSE;
        glGetProgramiv(result.program.id, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE) {
            result.warning = "cached program binary rejected by driver; recompiling";
            result.program = ProgramHandle{};
            return result;
        }
        result.loaded = true;
    } catch (const std::exception& error) {
        result.warning = std::string("cache load failed; recompiling: ") + error.what();
        result.program = ProgramHandle{};
    }
    return result;
}

void saveProgramCache(const MaterialSpec&,
                      const VariantRequest&,
                      const std::string& cacheKey,
                      const std::string& cacheDir,
                      const std::string& contextSignature,
                      GLuint program,
                      std::vector<std::string>& warnings) {
    if (cacheDir.empty()) {
        return;
    }
    try {
        GLint binaryLength = 0;
        glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &binaryLength);
        if (binaryLength <= 0) {
            warnings.push_back("driver did not provide a program binary; cache skipped");
            return;
        }
        GLint formatCount = 0;
        glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &formatCount);
        if (formatCount <= 0) {
            warnings.push_back("driver exposes no program binary formats; cache skipped");
            return;
        }
        std::vector<GLenum> formats(static_cast<std::size_t>(formatCount));
        glGetIntegerv(GL_PROGRAM_BINARY_FORMATS,
                      reinterpret_cast<GLint*>(formats.data()));
        GLenum format = formats.front();

        std::vector<std::uint8_t> binary(static_cast<std::size_t>(binaryLength));
        GLsizei written = 0;
        glGetProgramBinary(program, binaryLength, &written, &format, binary.data());
        if (glGetError() != GL_NO_ERROR || written <= 0) {
            warnings.push_back("glGetProgramBinary failed; cache skipped");
            return;
        }
        binary.resize(static_cast<std::size_t>(written));

        const auto root = std::filesystem::path(cacheDir) / "program-cache-v1" /
                          cacheKey.substr(0, 2) / cacheKey.substr(2);
        const auto binaryPath = root.string() + ".program";
        const auto metadataPath = root.string() + ".json";
        writeBinaryFileAtomic(binaryPath, binary.data(), binary.size());
        json metadata{
            {"schema", "material-verify-program-cache/1"},
            {"key", cacheKey},
            {"gl_signature", contextSignature},
            {"binary_format", format},
            {"binary_sha256", Sha256::hex(std::string(
                                  reinterpret_cast<const char*>(binary.data()), binary.size()))},
        };
        writeTextFileAtomic(metadataPath, metadata.dump(2) + "\n");
    } catch (const std::exception& error) {
        warnings.push_back(std::string("cache write failed: ") + error.what());
    }
}

ProgramHandle obtainProgram(const MaterialSpec& material,
                            const VariantRequest& variant,
                            const CommandLine& command,
                            const std::string& contextSignature,
                            const std::string& cacheKey,
                            VariantResult& result) {
    auto cached = tryLoadProgram(material, variant, cacheKey, command.cacheDir,
                                 contextSignature);
    if (!cached.warning.empty()) {
        result.warnings.push_back(cached.warning);
    }
    if (cached.loaded) {
        checkOutputContract(cached.program.id, material);
        result.cacheHit = true;
        return std::move(cached.program);
    }

    ShaderHandle vertex =
        compileShader(GL_VERTEX_SHADER, material.vertexSource, variant.macros,
                      material.vertexPath, variant.name);
    ShaderHandle fragment =
        compileShader(GL_FRAGMENT_SHADER, material.fragmentSource, variant.macros,
                      material.fragmentPath, variant.name);
    ProgramHandle program;
    program.id = glCreateProgram();
    if (program.id == 0) {
        throw makeError(ExitCode::Runtime, "GL_PROGRAM_ALLOCATION_FAILED", "shader_link",
                        "glCreateProgram failed");
    }
    glAttachShader(program.id, vertex.id);
    glAttachShader(program.id, fragment.id);
    glLinkProgram(program.id);
    GLint linked = GL_FALSE;
    glGetProgramiv(program.id, GL_LINK_STATUS, &linked);
    GLint logLength = 0;
    glGetProgramiv(program.id, GL_INFO_LOG_LENGTH, &logLength);
    std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
    GLsizei written = 0;
    glGetProgramInfoLog(program.id, logLength, &written, log.data());
    log.resize(static_cast<std::size_t>(std::max<GLsizei>(written, 0)));
    result.linkLog = log;
    if (linked != GL_TRUE) {
        auto error = makeError(ExitCode::ShaderLink, "SHADER_LINK_FAILED", "shader_link",
                               "shader program linking failed");
        error.variant = variant.name;
        error.line = lineFromLog(log);
        error.log = log;
        throw error;
    }
    glDetachShader(program.id, vertex.id);
    glDetachShader(program.id, fragment.id);
    checkOutputContract(program.id, material);
    saveProgramCache(material, variant, cacheKey, command.cacheDir, contextSignature,
                     program.id, result.warnings);
    return program;
}

GLuint createTexture(const ColorFormat& format,
                     int width,
                     int height,
                     GlResources& resources) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    resources.textures.push_back(texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(format.internal), width, height, 0,
                 format.external, format.type, nullptr);
    return texture;
}

void createFramebuffer(const MaterialSpec& material, GlResources& resources) {
    GLint maxColorAttachments = 0;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxColorAttachments);
    int maxLocation = -1;
    for (const auto& attachment : material.attachments) {
        maxLocation = std::max(maxLocation, attachment.location);
        if (attachment.location >= maxColorAttachments) {
            throw makeError(
                ExitCode::Attachment, "ATTACHMENT_LOCATION_EXCEEDS_LIMIT", "attachment",
                "attachment " + attachment.name + " location " +
                    std::to_string(attachment.location) +
                    " exceeds GL_MAX_COLOR_ATTACHMENTS=" +
                    std::to_string(maxColorAttachments));
        }
    }

    glGenFramebuffers(1, &resources.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, resources.framebuffer);
    std::vector<GLenum> drawBuffers(static_cast<std::size_t>(maxLocation + 1), GL_NONE);
    for (const auto& attachment : material.attachments) {
        const auto format = colorFormat(attachment);
        const auto texture =
            createTexture(format, material.width, material.height, resources);
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_COLOR_ATTACHMENT0 + attachment.location,
                               GL_TEXTURE_2D, texture, 0);
        drawBuffers[static_cast<std::size_t>(attachment.location)] =
            GL_COLOR_ATTACHMENT0 + attachment.location;
    }
    glDrawBuffers(static_cast<GLsizei>(drawBuffers.size()), drawBuffers.data());

    if (material.depthStencil.enabled) {
        const ColorFormat depthFormat{GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL,
                                      GL_UNSIGNED_INT_24_8};
        const auto depthTexture =
            createTexture(depthFormat, material.width, material.height, resources);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                               depthTexture, 0);
    }

    const auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        auto error = makeError(ExitCode::Framebuffer, "FRAMEBUFFER_INCOMPLETE",
                               "framebuffer",
                               "framebuffer is incomplete: " +
                                   framebufferStatusText(status));
        error.detail = "status=0x" + std::to_string(status);
        throw error;
    }
}

GLenum depthFunction(const std::string& name) {
    if (name == "never") return GL_NEVER;
    if (name == "less") return GL_LESS;
    if (name == "equal") return GL_EQUAL;
    if (name == "lequal") return GL_LEQUAL;
    if (name == "greater") return GL_GREATER;
    if (name == "notequal") return GL_NOTEQUAL;
    if (name == "gequal") return GL_GEQUAL;
    if (name == "always") return GL_ALWAYS;
    throw makeError(ExitCode::Input, "DEPTH_FUNCTION_INVALID", "input",
                    "invalid depth_function: " + name);
}

GLenum stencilFunction(const std::string& name) {
    if (name == "never") return GL_NEVER;
    if (name == "less") return GL_LESS;
    if (name == "equal") return GL_EQUAL;
    if (name == "lequal") return GL_LEQUAL;
    if (name == "greater") return GL_GREATER;
    if (name == "notequal") return GL_NOTEQUAL;
    if (name == "gequal") return GL_GEQUAL;
    if (name == "always") return GL_ALWAYS;
    throw makeError(ExitCode::Input, "STENCIL_FUNCTION_INVALID", "input",
                    "invalid stencil_function: " + name);
}

GLenum stencilOp(const std::string& name) {
    if (name == "keep") return GL_KEEP;
    if (name == "zero") return GL_ZERO;
    if (name == "replace") return GL_REPLACE;
    if (name == "incr") return GL_INCR;
    if (name == "incr_wrap") return GL_INCR_WRAP;
    if (name == "decr") return GL_DECR;
    if (name == "decr_wrap") return GL_DECR_WRAP;
    if (name == "invert") return GL_INVERT;
    throw makeError(ExitCode::Input, "STENCIL_OP_INVALID", "input",
                    "invalid stencil operation: " + name);
}

void configureState(const MaterialSpec& material) {
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, material.width, material.height);
    const auto& depth = material.depthStencil;
    if (depth.enabled && depth.depthTest) {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(depthFunction(depth.depthFunction));
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    glDepthMask(depth.depthWrite ? GL_TRUE : GL_FALSE);

    if (depth.enabled && depth.stencilTest) {
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(stencilFunction(depth.stencilFunction), depth.stencilReference,
                      static_cast<GLuint>(depth.stencilReadMask));
        glStencilMask(static_cast<GLuint>(depth.stencilWriteMask));
        glStencilOp(stencilOp(depth.stencilFail), stencilOp(depth.stencilDepthFail),
                    stencilOp(depth.stencilDepthPass));
    } else {
        glDisable(GL_STENCIL_TEST);
    }
}

void createDependencies(const MaterialSpec& material,
                        GLuint program,
                        GlResources& resources) {
    GLint maxTextureSize = 0;
    GLint maxTextureUnits = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxTextureUnits);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (const auto& texture : material.textures) {
        if (texture.width > maxTextureSize || texture.height > maxTextureSize) {
            throw makeError(
                ExitCode::Input, "TEXTURE_SIZE_UNSUPPORTED", "dependency",
                "texture " + texture.name + " size " + std::to_string(texture.width) + "x" +
                    std::to_string(texture.height) + " exceeds GL_MAX_TEXTURE_SIZE=" +
                    std::to_string(maxTextureSize));
        }
        if (texture.binding >= maxTextureUnits) {
            throw makeError(ExitCode::Input, "TEXTURE_BINDING_UNSUPPORTED", "dependency",
                            "texture binding exceeds GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS");
        }
        const auto pixels = generateProceduralTexture(texture);
        GLuint id = 0;
        glGenTextures(1, &id);
        resources.textures.push_back(id);
        glActiveTexture(GL_TEXTURE0 + texture.binding);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture.width, texture.height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, pixels.data());
        const auto location = glGetUniformLocation(program, texture.uniform.c_str());
        if (location >= 0) {
            glProgramUniform1i(program, location, texture.binding);
        }
    }

    if (material.constants) {
        const auto& constants = *material.constants;
        GLint maxBlockSize = 0;
        glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxBlockSize);
        const auto bytes = static_cast<GLsizeiptr>(constants.layout.size() * 16u);
        if (bytes > maxBlockSize) {
            throw makeError(ExitCode::Input, "UBO_SIZE_UNSUPPORTED", "dependency",
                            "constant buffer exceeds GL_MAX_UNIFORM_BLOCK_SIZE");
        }
        const auto blockIndex = glGetUniformBlockIndex(program, constants.blockName.c_str());
        if (blockIndex != GL_INVALID_INDEX) {
            GLint activeSize = 0;
            glGetActiveUniformBlockiv(program, blockIndex, GL_UNIFORM_BLOCK_DATA_SIZE,
                                      &activeSize);
            if (activeSize > bytes) {
                throw makeError(
                    ExitCode::Runtime, "UBO_SIZE_MISMATCH", "dependency",
                    "shader constant block is larger than injected data: shader=" +
                        std::to_string(activeSize) + " bytes, injected=" +
                        std::to_string(bytes) + " bytes");
            }
            glUniformBlockBinding(program, blockIndex,
                                  static_cast<GLuint>(constants.binding));
        }
        GLuint buffer = 0;
        glGenBuffers(1, &buffer);
        resources.buffers.push_back(buffer);
        glBindBuffer(GL_UNIFORM_BUFFER, buffer);
        std::vector<std::uint8_t> data(static_cast<std::size_t>(bytes), 0);
        for (std::size_t i = 0; i < constants.layout.size(); ++i) {
            const auto& value = constants.values.at(constants.layout[i]);
            std::memcpy(data.data() + i * 16u, value.data(), 16u);
        }
        glBufferData(GL_UNIFORM_BUFFER, bytes, data.data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, static_cast<GLuint>(constants.binding), buffer);
    }
}

void createGeometry(GlResources& resources) {
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    resources.vaos.push_back(vao);
    glBindVertexArray(vao);
}

void clearTargets(const MaterialSpec& material) {
    for (const auto& attachment : material.attachments) {
        glClearBufferfv(GL_COLOR, attachment.location, attachment.clear.data());
    }
    if (material.depthStencil.enabled) {
        glClearBufferfi(GL_DEPTH_STENCIL, 0, material.depthStencil.clearDepth,
                        material.depthStencil.clearStencil);
    }
}

RawAttachment readColorAttachment(const MaterialSpec& material,
                                  const AttachmentSpec& attachment) {
    RawAttachment result;
    result.name = attachment.name;
    result.kind = "color";
    result.format = attachment.format;
    result.width = material.width;
    result.height = material.height;
    const auto pixels =
        static_cast<std::size_t>(material.width) * static_cast<std::size_t>(material.height);
    glReadBuffer(GL_COLOR_ATTACHMENT0 + attachment.location);

    if (attachment.pixelFormat == PixelFormat::RGBA8) {
        result.raw.resize(pixels * 4u);
        glReadPixels(0, 0, material.width, material.height, GL_RGBA, GL_UNSIGNED_BYTE,
                     result.raw.data());
        result.rgba.resize(pixels * 4u);
        for (std::size_t i = 0; i < result.raw.size(); ++i) {
            result.rgba[i] = static_cast<float>(result.raw[i]) / 255.0f;
        }
    } else if (attachment.pixelFormat == PixelFormat::RGBA16F) {
        result.rgba.resize(pixels * 4u);
        glReadPixels(0, 0, material.width, material.height, GL_RGBA, GL_FLOAT,
                     result.rgba.data());
        result.raw.resize(result.rgba.size() * sizeof(float));
        std::memcpy(result.raw.data(), result.rgba.data(), result.raw.size());
    } else {
        throw makeError(ExitCode::Framebuffer, "DEPTH_ATTACHMENT_READ_AS_COLOR",
                        "framebuffer",
                        "depth/stencil format cannot be read as a color attachment");
    }
    throwGlError("ATTACHMENT_READ_FAILED", "attachment", "glReadPixels(color)");
    result.sha256 = Sha256::hex(std::string(
        reinterpret_cast<const char*>(result.raw.data()), result.raw.size()));
    return result;
}

RawAttachment readDepth(const MaterialSpec& material) {
    RawAttachment result;
    result.name = "depth";
    result.kind = "depth";
    result.format = material.depthStencil.format;
    result.width = material.width;
    result.height = material.height;
    const auto pixels =
        static_cast<std::size_t>(material.width) * static_cast<std::size_t>(material.height);
    glReadBuffer(GL_NONE);
    result.depth.resize(pixels);
    glReadPixels(0, 0, material.width, material.height, GL_DEPTH_COMPONENT, GL_FLOAT,
                 result.depth.data());
    throwGlError("DEPTH_READ_FAILED", "attachment", "glReadPixels(depth)");
    result.raw.resize(result.depth.size() * sizeof(float));
    std::memcpy(result.raw.data(), result.depth.data(), result.raw.size());
    result.sha256 = Sha256::hex(std::string(
        reinterpret_cast<const char*>(result.raw.data()), result.raw.size()));
    return result;
}

RawAttachment readStencil(const MaterialSpec& material) {
    RawAttachment result;
    result.name = "stencil";
    result.kind = "stencil";
    result.format = material.depthStencil.format;
    result.width = material.width;
    result.height = material.height;
    const auto pixels =
        static_cast<std::size_t>(material.width) * static_cast<std::size_t>(material.height);
    glReadBuffer(GL_NONE);
    result.stencil.resize(pixels);
    glReadPixels(0, 0, material.width, material.height, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE,
                 result.stencil.data());
    throwGlError("STENCIL_READ_FAILED", "attachment", "glReadPixels(stencil)");
    result.raw = result.stencil;
    result.sha256 = Sha256::hex(std::string(
        reinterpret_cast<const char*>(result.raw.data()), result.raw.size()));
    return result;
}

void dumpAttachment(const std::string& pixelRoot,
                    const std::string& variant,
                    RawAttachment& attachment) {
    if (pixelRoot.empty()) {
        return;
    }
    const auto path = std::filesystem::path(pixelRoot) / sanitizeName(variant) /
                      (sanitizeName(attachment.name) + "." + attachment.kind + ".raw");
    try {
        writeBinaryFileAtomic(path, attachment.raw.data(), attachment.raw.size());
        attachment.dumpedPath = path.string();
    } catch (const std::exception& error) {
        throw makeError(ExitCode::Output, "PIXEL_DUMP_FAILED", "output", error.what());
    }
}

bool needsDepth(const MaterialSpec& material) {
    return std::any_of(material.rules.begin(), material.rules.end(), [](const RuleSpec& rule) {
        return rule.type == "depth_expected" || rule.type == "depth_normal";
    });
}

bool needsStencil(const MaterialSpec& material) {
    return std::any_of(material.rules.begin(), material.rules.end(), [](const RuleSpec& rule) {
        return rule.type == "stencil_expected";
    });
}

}  // namespace

Renderer::Renderer(const GlContext& context) : context_(context) {}

VariantResult Renderer::render(const MaterialSpec& material,
                               const VariantRequest& variant,
                               const CommandLine& command,
                               const std::string& pixelRoot) {
    const auto start = std::chrono::steady_clock::now();
    if (needsDepth(material) && !material.depthStencil.enabled) {
        throw makeError(ExitCode::Input, "DEPTH_RULE_WITHOUT_ATTACHMENT", "input",
                        "depth validation rules require depth_stencil.enabled=true");
    }
    if (needsStencil(material) && !material.depthStencil.enabled) {
        throw makeError(ExitCode::Input, "STENCIL_RULE_WITHOUT_ATTACHMENT", "input",
                        "stencil validation rules require depth_stencil.enabled=true");
    }

    VariantResult result;
    result.name = variant.name;
    result.macros = variant.macros;
    result.cacheKey = programCacheKey(material, variant, context_.signature());
    drainGlErrors();

    GlResources resources;
    auto program = obtainProgram(material, variant, command, context_.signature(),
                                 result.cacheKey, result);
    createFramebuffer(material, resources);
    createGeometry(resources);
    createDependencies(material, program.id, resources);
    glUseProgram(program.id);
    configureState(material);
    clearTargets(material);
    glBindVertexArray(resources.vaos.front());
    throwGlError("RENDER_STATE_ERROR", "runtime", "OpenGL state setup");
    drainGlErrors();
    glDrawArraysInstanced(GL_TRIANGLES, 0, material.geometry.vertices,
                          material.geometry.instances);
    throwGlError("ATTACHMENT_DRAW_ERROR", "attachment", "glDrawArraysInstanced");
    glFinish();

    for (const auto& attachment : material.attachments) {
        result.attachments.push_back(readColorAttachment(material, attachment));
    }
    if (material.depthStencil.enabled) {
        result.attachments.push_back(readDepth(material));
        result.attachments.push_back(readStencil(material));
    }
    for (auto& attachment : result.attachments) {
        dumpAttachment(pixelRoot, variant.name, attachment);
    }

    const auto validation =
        evaluateRules(material, result.attachments, material.maxDiffPixels);
    result.rules = validation.rules;
    result.pixelCount = validation.pixelCount;
    result.passedPixels = validation.passedPixels;
    result.failedPixels = validation.failedPixels;
    const auto end = std::chrono::steady_clock::now();
    result.durationMs =
        command.deterministic
            ? 0.0
            : std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

}  // namespace mv
