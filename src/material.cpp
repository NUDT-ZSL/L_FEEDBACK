#include "material.hpp"

#include "util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace mv {
namespace {

using json = nlohmann::json;

[[noreturn]] void fail(std::string code,
                       std::string message,
                       int line = -1,
                       std::string detail = {}) {
    auto error = makeError(ExitCode::Input, std::move(code), "input", std::move(message));
    error.line = line;
    error.detail = std::move(detail);
    throw error;
}

json requireObject(const json& object, const char* key, std::string context) {
    if (!object.contains(key) || !object.at(key).is_object()) {
        fail("METADATA_FIELD_INVALID",
             context + "." + key + " must be an object");
    }
    return object.at(key);
}

json requireArray(const json& object, const char* key, std::string context) {
    if (!object.contains(key) || !object.at(key).is_array()) {
        fail("METADATA_FIELD_INVALID",
             context + "." + key + " must be an array");
    }
    return object.at(key);
}

std::string requireString(const json& object,
                          const char* key,
                          const std::string& context,
                          bool allowEmpty = false) {
    if (!object.contains(key) || !object.at(key).is_string()) {
        fail("METADATA_FIELD_INVALID",
             context + "." + key + " must be a string");
    }
    auto value = object.at(key).get<std::string>();
    if (!allowEmpty && trim(value).empty()) {
        fail("METADATA_FIELD_INVALID", context + "." + key + " must not be empty");
    }
    return value;
}

std::string optionalString(const json& object,
                           const char* key,
                           const std::string& fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (!object.at(key).is_string()) {
        fail("METADATA_FIELD_INVALID", std::string("field ") + key + " must be a string");
    }
    return object.at(key).get<std::string>();
}

int optionalInt(const json& object, const char* key, int fallback, int minimum, int maximum) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (!object.at(key).is_number_integer()) {
        fail("METADATA_FIELD_INVALID", std::string("field ") + key + " must be an integer");
    }
    const auto value = object.at(key).get<int>();
    if (value < minimum || value > maximum) {
        fail("METADATA_FIELD_INVALID",
             std::string("field ") + key + " is outside [" + std::to_string(minimum) +
                 ", " + std::to_string(maximum) + "]");
    }
    return value;
}

float optionalFloat(const json& object, const char* key, float fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (!object.at(key).is_number()) {
        fail("METADATA_FIELD_INVALID", std::string("field ") + key + " must be a number");
    }
    return object.at(key).get<float>();
}

bool optionalBool(const json& object, const char* key, bool fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (!object.at(key).is_boolean()) {
        fail("METADATA_FIELD_INVALID", std::string("field ") + key + " must be a boolean");
    }
    return object.at(key).get<bool>();
}

std::array<float, 4> parseVec4(const json& value, const std::string& context) {
    if (!value.is_array() || value.size() != 4) {
        fail("METADATA_FIELD_INVALID", context + " must be an array of four numbers");
    }
    std::array<float, 4> result{};
    for (std::size_t i = 0; i < 4; ++i) {
        if (!value.at(i).is_number()) {
            fail("METADATA_FIELD_INVALID", context + " contains a non-number");
        }
        result[i] = value.at(i).get<float>();
    }
    return result;
}

PixelFormat parsePixelFormat(const std::string& format) {
    if (format == "RGBA8") {
        return PixelFormat::RGBA8;
    }
    if (format == "RGBA16F") {
        return PixelFormat::RGBA16F;
    }
    if (format == "DEPTH24_STENCIL8" || format == "D24S8") {
        return PixelFormat::Depth24Stencil8AsColor;
    }
    fail("METADATA_ATTACHMENT_FORMAT", "unsupported attachment format: " + format);
}

bool validMacro(const std::string& macro) {
    if (macro.empty()) {
        return false;
    }
    const auto nameEnd = macro.find('=');
    const auto name = macro.substr(0, nameEnd);
    if (name.empty() || (!std::isalpha(static_cast<unsigned char>(name[0])) && name[0] != '_')) {
        return false;
    }
    for (const auto c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            return false;
        }
    }
    if (nameEnd != std::string::npos) {
        const auto value = macro.substr(nameEnd + 1);
        return value.find('\n') == std::string::npos &&
               value.find('\r') == std::string::npos;
    }
    return true;
}

std::string normalizedMacro(const std::string& value) {
    auto macro = trim(value);
    if (!validMacro(macro)) {
        fail("METADATA_VARIANT_MACRO_INVALID", "invalid variant macro: " + macro);
    }
    return macro;
}

RuleSpec parseRule(const json& value, std::size_t index) {
    if (!value.is_object()) {
        fail("METADATA_RULE_INVALID", "rules[" + std::to_string(index) + "] must be an object");
    }
    RuleSpec rule;
    rule.type = requireString(value, "type", "rules[" + std::to_string(index) + "]");
    rule.id = optionalString(value, "id", "rule-" + std::to_string(index));
    rule.source = optionalString(value, "source", "");
    rule.target = optionalString(value, "target", "");
    rule.attachment = optionalString(value, "attachment", "");
    rule.texture = optionalString(value, "texture", "");
    rule.constant = optionalString(value, "constant", "");
    rule.sourceChannel = optionalString(value, "source_channel", "");
    rule.targetChannel = optionalString(value, "target_channel", "");
    rule.channels = optionalString(value, "channels", "rgb");
    rule.attachmentChannel = optionalString(value, "attachment_channel", "");
    rule.sampling = optionalString(value, "sampling", "texel_fetch_mod");
    rule.component = optionalInt(value, "component", 0, 0, 3);
    rule.expected = optionalFloat(value, "expected", 0.0f);
    rule.tolerance = optionalFloat(value, "tolerance", 0.0f);
    if (rule.tolerance < 0.0f) {
        fail("METADATA_RULE_INVALID", "rule tolerance cannot be negative: " + rule.id);
    }

    if (rule.type == "channel_match") {
        if (rule.source.empty() || rule.target.empty() || rule.sourceChannel.empty() ||
            rule.targetChannel.empty()) {
            fail("METADATA_RULE_INVALID",
                 "channel_match requires source, target, source_channel and target_channel: " +
                     rule.id);
        }
    } else if (rule.type == "depth_normal") {
        if (rule.source.empty() || rule.target.empty()) {
            fail("METADATA_RULE_INVALID",
                 "depth_normal requires source (depth), target (normal) and tolerance: " + rule.id);
        }
    } else if (rule.type == "texture_sample") {
        if (rule.texture.empty() || rule.attachment.empty()) {
            fail("METADATA_RULE_INVALID",
                 "texture_sample requires texture and attachment: " + rule.id);
        }
        if (rule.sampling != "texel_fetch_mod") {
            fail("METADATA_RULE_INVALID",
                 "unsupported texture sampling mode: " + rule.sampling);
        }
    } else if (rule.type == "constant_match") {
        if (rule.constant.empty() || rule.attachment.empty() ||
            rule.attachmentChannel.empty()) {
            fail("METADATA_RULE_INVALID",
                 "constant_match requires constant, attachment and attachment_channel: " +
                     rule.id);
        }
    } else if (rule.type == "channel_expected") {
        if (rule.attachment.empty() || rule.attachmentChannel.empty()) {
            fail("METADATA_RULE_INVALID",
                 "channel_expected requires attachment and attachment_channel: " + rule.id);
        }
    } else if (rule.type == "depth_expected" || rule.type == "stencil_expected") {
        // No additional fields.
    } else {
        fail("METADATA_RULE_UNKNOWN", "unsupported rule type: " + rule.type);
    }
    return rule;
}

}  // namespace

MaterialSpec loadMaterial(const std::string& metadataPath) {
    MaterialSpec material;
    material.metadataPath = std::filesystem::absolute(metadataPath).lexically_normal().string();
    std::string text;
    try {
        text = readTextFile(material.metadataPath);
    } catch (const std::exception& error) {
        fail("METADATA_READ_FAILED", error.what());
    }
    material.metadataSha256 = Sha256::hex(text);

    json root;
    try {
        root = json::parse(text);
    } catch (const json::parse_error& error) {
        int line = 1;
        const auto byte = error.byte;
        for (std::size_t i = 0; i < byte && i < text.size(); ++i) {
            if (text[i] == '\n') {
                ++line;
            }
        }
        fail("METADATA_PARSE_FAILED", error.what(), line, error.what());
    }
    if (!root.is_object()) {
        fail("METADATA_PARSE_FAILED", "material metadata root must be an object");
    }

    material.name = requireString(root, "name", "material");
    material.sourceDir =
        std::filesystem::path(material.metadataPath).parent_path().string();
    material.vertexPath = (std::filesystem::path(material.sourceDir) /
                           requireString(root, "vertex_shader", "material"))
                              .lexically_normal()
                              .string();
    material.fragmentPath = (std::filesystem::path(material.sourceDir) /
                             requireString(root, "fragment_shader", "material"))
                                .lexically_normal()
                                .string();
    material.width = optionalInt(root, "width", 64, 1, 8192);
    material.height = optionalInt(root, "height", 48, 1, 8192);
    material.maxDiffPixels = optionalInt(root, "max_diff_pixels", 4096, 1, 1000000);

    try {
        material.vertexSource = readTextFile(material.vertexPath);
        material.fragmentSource = readTextFile(material.fragmentPath);
    } catch (const std::exception& error) {
        fail("METADATA_SHADER_READ_FAILED", error.what());
    }

    if (root.contains("geometry")) {
        const auto& geometry = requireObject(root, "geometry", "material");
        material.geometry.mode = optionalString(geometry, "mode", "fullscreen_triangle");
        material.geometry.vertices = optionalInt(geometry, "vertices", 3, 1, 1024);
        material.geometry.instances = optionalInt(geometry, "instances", 1, 1, 1024);
        if (material.geometry.mode != "fullscreen_triangle" &&
            material.geometry.mode != "overlap") {
            fail("METADATA_GEOMETRY_INVALID",
                 "geometry.mode must be fullscreen_triangle or overlap");
        }
        if (material.geometry.vertices != 3) {
            fail("METADATA_GEOMETRY_INVALID",
                 "built-in geometry requires exactly 3 vertices");
        }
    }

    const auto& attachments = requireArray(root, "attachments", "material");
    if (attachments.empty()) {
        fail("METADATA_ATTACHMENT_INVALID", "at least one color attachment is required");
    }
    std::set<std::string> attachmentNames;
    std::set<int> locations;
    for (std::size_t i = 0; i < attachments.size(); ++i) {
        const auto& item = attachments.at(i);
        const auto context = "attachments[" + std::to_string(i) + "]";
        if (!item.is_object()) {
            fail("METADATA_ATTACHMENT_INVALID", context + " must be an object");
        }
        AttachmentSpec attachment;
        attachment.name = requireString(item, "name", context);
        attachment.location = optionalInt(item, "location", static_cast<int>(i), 0, 31);
        attachment.format = requireString(item, "format", context);
        attachment.pixelFormat = parsePixelFormat(attachment.format);
        if (item.contains("clear")) {
            attachment.clear = parseVec4(item.at("clear"), context + ".clear");
        }
        if (attachmentNames.count(attachment.name) != 0) {
            fail("METADATA_ATTACHMENT_INVALID",
                 "duplicate attachment name: " + attachment.name);
        }
        if (locations.count(attachment.location) != 0) {
            fail("METADATA_ATTACHMENT_INVALID",
                 "duplicate attachment location: " + std::to_string(attachment.location));
        }
        attachmentNames.insert(attachment.name);
        locations.insert(attachment.location);
        material.attachments.push_back(attachment);
    }

    if (root.contains("depth_stencil")) {
        const auto& value = requireObject(root, "depth_stencil", "material");
        auto& depth = material.depthStencil;
        depth.enabled = optionalBool(value, "enabled", true);
        depth.format = optionalString(value, "format", "D24S8");
        if (depth.format != "D24S8" && depth.format != "DEPTH24_STENCIL8") {
            fail("METADATA_DEPTH_FORMAT", "unsupported depth/stencil format: " + depth.format);
        }
        depth.clearDepth = optionalFloat(value, "clear_depth", 1.0f);
        depth.clearStencil = optionalInt(value, "clear_stencil", 0, 0, 255);
        depth.depthTest = optionalBool(value, "depth_test", true);
        depth.depthWrite = optionalBool(value, "depth_write", true);
        depth.depthFunction = optionalString(value, "depth_function", "less");
        depth.stencilTest = optionalBool(value, "stencil_test", false);
        depth.stencilFunction = optionalString(value, "stencil_function", "always");
        depth.stencilReference = optionalInt(value, "stencil_reference", 0, 0, 255);
        depth.stencilReadMask = optionalInt(value, "stencil_read_mask", 0xff, 0, 255);
        depth.stencilWriteMask = optionalInt(value, "stencil_write_mask", 0xff, 0, 255);
        depth.stencilFail = optionalString(value, "stencil_fail", "keep");
        depth.stencilDepthFail = optionalString(value, "stencil_depth_fail", "keep");
        depth.stencilDepthPass = optionalString(value, "stencil_depth_pass", "keep");
    }

    if (root.contains("variants")) {
        const auto& variants = requireObject(root, "variants", "material");
        for (auto it = variants.begin(); it != variants.end(); ++it) {
            VariantSpec variant;
            variant.name = it.key();
            if (!it.value().is_array()) {
                fail("METADATA_VARIANT_INVALID",
                     "variant " + variant.name + " must be an array of macros");
            }
            std::set<std::string> unique;
            for (const auto& macro : it.value()) {
                if (!macro.is_string()) {
                    fail("METADATA_VARIANT_INVALID",
                         "variant " + variant.name + " contains a non-string macro");
                }
                const auto normalized = normalizedMacro(macro.get<std::string>());
                if (unique.insert(normalized).second) {
                    variant.macros.push_back(normalized);
                }
            }
            std::sort(variant.macros.begin(), variant.macros.end());
            material.variants[variant.name] = std::move(variant);
        }
    }
    if (material.variants.empty()) {
        material.variants["default"] = VariantSpec{"default", {}};
    }

    if (root.contains("default_variants")) {
        if (!root.at("default_variants").is_array()) {
            fail("METADATA_FIELD_INVALID", "default_variants must be an array");
        }
        for (const auto& item : root.at("default_variants")) {
            if (!item.is_string()) {
                fail("METADATA_FIELD_INVALID", "default_variants contains a non-string");
            }
            material.defaultVariants.push_back(item.get<std::string>());
        }
    } else {
        material.defaultVariants.push_back("default");
    }
    for (const auto& name : material.defaultVariants) {
        if (material.variants.count(name) == 0) {
            fail("METADATA_VARIANT_UNKNOWN", "unknown default variant: " + name);
        }
    }

    if (root.contains("dependencies")) {
        const auto& dependencies = requireObject(root, "dependencies", "material");
        if (dependencies.contains("textures")) {
            const auto& textures = requireArray(dependencies, "textures", "dependencies");
            std::set<std::string> textureNames;
            std::set<int> bindings;
            for (std::size_t i = 0; i < textures.size(); ++i) {
                const auto& item = textures.at(i);
                const auto context = "dependencies.textures[" + std::to_string(i) + "]";
                if (!item.is_object()) {
                    fail("METADATA_DEPENDENCY_INVALID", context + " must be an object");
                }
                TextureSpec texture;
                texture.name = requireString(item, "name", context);
                texture.uniform = optionalString(item, "uniform", "u" + texture.name);
                texture.width = optionalInt(item, "width", 8, 1, 16384);
                texture.height = optionalInt(item, "height", 8, 1, 16384);
                if (item.contains("seed")) {
                    if (!item.at("seed").is_number_unsigned() &&
                        !item.at("seed").is_number_integer()) {
                        fail("METADATA_DEPENDENCY_INVALID", context + ".seed must be an integer");
                    }
                    texture.seed = item.at("seed").get<std::uint32_t>();
                }
                texture.binding = optionalInt(item, "binding", static_cast<int>(i), 0, 31);
                if (!textureNames.insert(texture.name).second) {
                    fail("METADATA_DEPENDENCY_INVALID",
                         "duplicate texture name: " + texture.name);
                }
                if (!bindings.insert(texture.binding).second) {
                    fail("METADATA_DEPENDENCY_INVALID",
                         "duplicate texture binding: " + std::to_string(texture.binding));
                }
                material.textures.push_back(texture);
            }
        }
        if (dependencies.contains("constants")) {
            const auto& constants = requireObject(dependencies, "constants", "dependencies");
            ConstantBufferSpec buffer;
            buffer.blockName = optionalString(constants, "block_name", "DepConstants");
            buffer.binding = optionalInt(constants, "binding", 0, 0, 31);
            if (constants.contains("seed")) {
                if (!constants.at("seed").is_number_integer() &&
                    !constants.at("seed").is_number_unsigned()) {
                    fail("METADATA_DEPENDENCY_INVALID", "constants.seed must be an integer");
                }
                buffer.seed = constants.at("seed").get<std::uint32_t>();
            }
            const auto& values = requireObject(constants, "values", "dependencies.constants");
            for (auto it = values.begin(); it != values.end(); ++it) {
                buffer.values[it.key()] = parseVec4(
                    it.value(), "dependencies.constants.values." + it.key());
            }
            if (constants.contains("layout")) {
                const auto& layout = requireArray(constants, "layout", "dependencies.constants");
                for (const auto& field : layout) {
                    if (!field.is_string()) {
                        fail("METADATA_DEPENDENCY_INVALID",
                             "constants.layout contains a non-string");
                    }
                    buffer.layout.push_back(field.get<std::string>());
                }
            } else {
                for (const auto& item : buffer.values) {
                    buffer.layout.push_back(item.first);
                }
            }
            for (const auto& field : buffer.layout) {
                if (buffer.values.count(field) == 0) {
                    fail("METADATA_DEPENDENCY_INVALID",
                         "constants.layout references missing value: " + field);
                }
            }
            material.constants = std::move(buffer);
        }
    }

    if (root.contains("rules")) {
        const auto& rules = requireArray(root, "rules", "material");
        std::set<std::string> ids;
        for (std::size_t i = 0; i < rules.size(); ++i) {
            auto rule = parseRule(rules.at(i), i);
            if (!ids.insert(rule.id).second) {
                fail("METADATA_RULE_INVALID", "duplicate rule id: " + rule.id);
            }
            if (!rule.source.empty() && rule.source != "depth" && rule.source != "stencil" &&
                attachmentNames.count(rule.source) == 0) {
                fail("METADATA_RULE_INVALID",
                     "rule " + rule.id + " references unknown source attachment: " + rule.source);
            }
            if (!rule.target.empty() && rule.target != "depth" && rule.target != "stencil" &&
                attachmentNames.count(rule.target) == 0) {
                fail("METADATA_RULE_INVALID",
                     "rule " + rule.id + " references unknown target attachment: " + rule.target);
            }
            if (!rule.attachment.empty() && rule.attachment != "depth" &&
                rule.attachment != "stencil" && attachmentNames.count(rule.attachment) == 0) {
                fail("METADATA_RULE_INVALID",
                     "rule " + rule.id + " references unknown attachment: " + rule.attachment);
            }
            if (!rule.texture.empty()) {
                const auto found = std::find_if(
                    material.textures.begin(), material.textures.end(),
                    [&](const TextureSpec& texture) { return texture.name == rule.texture; });
                if (found == material.textures.end()) {
                    fail("METADATA_RULE_INVALID",
                         "rule " + rule.id + " references unknown texture: " + rule.texture);
                }
            }
            if (!rule.constant.empty()) {
                if (!material.constants || material.constants->values.count(rule.constant) == 0) {
                    fail("METADATA_RULE_INVALID",
                         "rule " + rule.id + " references unknown constant: " + rule.constant);
                }
            }
            material.rules.push_back(std::move(rule));
        }
    }

    if (material.rules.empty()) {
        fail("METADATA_RULE_INVALID", "at least one validation rule is required");
    }
    return material;
}

std::vector<VariantRequest> resolveVariants(const MaterialSpec& material,
                                            const std::string& selection) {
    std::vector<std::string> tokens;
    if (trim(selection).empty()) {
        tokens = material.defaultVariants;
    } else {
        for (const auto& token : split(selection, ',')) {
            if (!trim(token).empty()) {
                tokens.push_back(trim(token));
            }
        }
    }
    if (tokens.empty()) {
        fail("VARIANT_SELECTION_EMPTY", "no variants selected");
    }
    if (tokens.size() == 1 && tokens[0] == "all") {
        tokens.clear();
        for (const auto& item : material.variants) {
            tokens.push_back(item.first);
        }
    }

    std::vector<VariantRequest> result;
    std::set<std::string> seenNames;
    for (const auto& token : tokens) {
        VariantRequest request;
        request.name = token;
        const auto found = material.variants.find(token);
        if (found != material.variants.end()) {
            request.name = found->first;
            request.macros = found->second.macros;
        } else if (token.rfind("macros:", 0) == 0) {
            request.name = token;
            for (const auto& macro : split(token.substr(7), '+')) {
                request.macros.push_back(normalizedMacro(macro));
            }
        } else {
            for (const auto& macro : split(token, '+')) {
                request.macros.push_back(normalizedMacro(macro));
            }
        }
        request.macros = sortedMacros(request.macros);
        if (!seenNames.insert(request.name).second) {
            continue;
        }
        result.push_back(std::move(request));
    }
    if (result.empty()) {
        fail("VARIANT_SELECTION_EMPTY", "no variants selected");
    }
    return result;
}

void applyDependencyOverrides(MaterialSpec& material, const CommandLine& command) {
    if (command.textureSize) {
        for (auto& texture : material.textures) {
            texture.width = command.textureSize->first;
            texture.height = command.textureSize->second;
        }
    }
    if (command.constantOverrides.empty()) {
        return;
    }
    if (!material.constants) {
        fail("CLI_CONSTANT_WITHOUT_UBO",
             "--const was provided but the material declares no constant buffer");
    }
    for (const auto& overrideValue : command.constantOverrides) {
        if (material.constants->values.count(overrideValue.first) == 0) {
            fail("CLI_CONSTANT_UNKNOWN",
                 "--const references unknown constant: " + overrideValue.first);
        }
        material.constants->values[overrideValue.first] = overrideValue.second;
    }
}

std::vector<std::string> sortedMacros(const std::vector<std::string>& macros) {
    auto result = macros;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

}  // namespace mv
