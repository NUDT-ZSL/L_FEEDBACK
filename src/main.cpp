#include "cli.hpp"
#include "context.hpp"
#include "material.hpp"
#include "renderer.hpp"
#include "types.hpp"
#include "util.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mv {
namespace {

using json = nlohmann::json;

json errorJson(const MaterialError& error) {
    json result{
        {"code", error.code},
        {"stage", error.stage},
        {"message", error.message},
        {"exit_code", static_cast<int>(error.exitCode)},
    };
    if (!error.shader.empty()) {
        result["shader"] = error.shader;
    }
    if (!error.variant.empty()) {
        result["variant"] = error.variant;
    }
    if (error.line >= 0) {
        result["line"] = error.line;
    }
    if (!error.log.empty()) {
        result["log"] = error.log;
    }
    if (!error.detail.empty()) {
        result["detail"] = error.detail;
    }
    return result;
}

json attachmentJson(const RawAttachment& attachment) {
    json result{
        {"name", attachment.name},
        {"kind", attachment.kind},
        {"format", attachment.format},
        {"width", attachment.width},
        {"height", attachment.height},
        {"bytes", attachment.raw.size()},
        {"sha256", attachment.sha256},
    };
    if (!attachment.dumpedPath.empty()) {
        result["pixel_file"] = attachment.dumpedPath;
    }
    return result;
}

json differenceJson(const PixelDifference& difference) {
    return json{
        {"x", difference.x},
        {"y", difference.y},
        {"source", difference.source},
        {"target", difference.target},
        {"expected", difference.expected},
        {"actual", difference.actual},
        {"delta", difference.delta},
    };
}

json ruleJson(const RuleResult& rule) {
    json differences = json::array();
    for (const auto& difference : rule.differences) {
        differences.push_back(differenceJson(difference));
    }
    return json{
        {"id", rule.id},
        {"type", rule.type},
        {"status", rule.passed ? "pass" : "fail"},
        {"tested_pixels", rule.testedPixels},
        {"passed_pixels", rule.passedPixels},
        {"failed_pixels", rule.failedPixels},
        {"differences_truncated", rule.differencesTruncated},
        {"differences", std::move(differences)},
    };
}

json variantJson(const VariantResult& variant) {
    json attachments = json::array();
    for (const auto& attachment : variant.attachments) {
        attachments.push_back(attachmentJson(attachment));
    }
    json rules = json::array();
    for (const auto& rule : variant.rules) {
        rules.push_back(ruleJson(rule));
    }
    json result{
        {"name", variant.name},
        {"macros", variant.macros},
        {"status", variant.failedPixels == 0 ? "pass" : "fail"},
        {"cache_key", variant.cacheKey},
        {"cache_hit", variant.cacheHit},
        {"compile_log", variant.compileLog},
        {"link_log", variant.linkLog},
        {"warnings", variant.warnings},
        {"attachment_count", variant.attachments.size()},
        {"pixel_count", variant.pixelCount},
        {"validation_passed_pixels", variant.passedPixels},
        {"validation_failed_pixels", variant.failedPixels},
        {"duration_ms", variant.durationMs},
        {"attachments", std::move(attachments)},
        {"rules", std::move(rules)},
    };
    return result;
}

json dependencyJson(const MaterialSpec& material) {
    json textures = json::array();
    for (const auto& texture : material.textures) {
        textures.push_back(json{
            {"name", texture.name},
            {"uniform", texture.uniform},
            {"width", texture.width},
            {"height", texture.height},
            {"seed", texture.seed},
            {"binding", texture.binding},
        });
    }
    json constants = nullptr;
    if (material.constants) {
        json values = json::object();
        for (const auto& field : material.constants->layout) {
            values[field] = material.constants->values.at(field);
        }
        constants = json{
            {"block_name", material.constants->blockName},
            {"binding", material.constants->binding},
            {"seed", material.constants->seed},
            {"layout", material.constants->layout},
            {"values", std::move(values)},
        };
    }
    return json{{"textures", std::move(textures)}, {"constants", std::move(constants)}};
}

json stableProjection(const MaterialSpec* material,
                      const std::vector<json>& variantResults,
                      const std::vector<MaterialError>& errors) {
    json result{
        {"schema", "material-verify-stable/1"},
        {"variants", variantResults},
        {"errors", json::array()},
    };
    for (const auto& error : errors) {
        result["errors"].push_back(errorJson(error));
    }
    if (material != nullptr) {
        result["material"] = json{
            {"name", material->name},
            {"metadata_sha256", material->metadataSha256},
            {"width", material->width},
            {"height", material->height},
            {"dependencies", dependencyJson(*material)},
        };
    }
    return result;
}

json stableVariantJson(const VariantResult& variant) {
    json attachments = json::array();
    for (const auto& attachment : variant.attachments) {
        attachments.push_back(json{{"name", attachment.name},
                                   {"kind", attachment.kind},
                                   {"sha256", attachment.sha256}});
    }
    json rules = json::array();
    for (const auto& rule : variant.rules) {
        json differences = json::array();
        for (const auto& difference : rule.differences) {
            differences.push_back(differenceJson(difference));
        }
        rules.push_back(json{
            {"id", rule.id},
            {"type", rule.type},
            {"status", rule.passed ? "pass" : "fail"},
            {"tested_pixels", rule.testedPixels},
            {"passed_pixels", rule.passedPixels},
            {"failed_pixels", rule.failedPixels},
            {"differences_truncated", rule.differencesTruncated},
            {"differences", std::move(differences)},
        });
    }
    return json{
        {"name", variant.name},
        {"macros", variant.macros},
        {"cache_key", variant.cacheKey},
        {"attachments", std::move(attachments)},
        {"rules", std::move(rules)},
        {"pixel_count", variant.pixelCount},
        {"validation_passed_pixels", variant.passedPixels},
        {"validation_failed_pixels", variant.failedPixels},
    };
}

std::filesystem::path defaultOutputPath(const std::string& materialPath) {
    const auto path = std::filesystem::path(materialPath);
    return path.parent_path() / (path.stem().string() + ".report.json");
}

int writeReport(const std::filesystem::path& path, const json& report) {
    try {
        writeTextFileAtomic(path, report.dump(2) + "\n");
        return static_cast<int>(ExitCode::Success);
    } catch (const std::exception& error) {
        std::cerr << errorJson(makeError(ExitCode::Output, "REPORT_WRITE_FAILED", "output",
                                         error.what()))
                         .dump()
                  << "\n";
        return static_cast<int>(ExitCode::Output);
    }
}

}  // namespace

int run(int argc, char** argv) {
    const auto started = std::chrono::steady_clock::now();
    const auto parsed = parseCommandLine(argc, argv);
    if (parsed.error) {
        std::cerr << errorJson(*parsed.error).dump(2) << "\n";
        return static_cast<int>(parsed.error->exitCode);
    }
    if (!parsed.command) {
        return static_cast<int>(ExitCode::Input);
    }
    const auto& command = *parsed.command;
    if (command.showHelp) {
        std::cout << helpText();
        return static_cast<int>(ExitCode::Success);
    }

    std::optional<MaterialSpec> material;
    std::vector<VariantRequest> variants;
    std::vector<VariantResult> results;
    std::vector<MaterialError> errors;
    std::optional<ContextInfo> contextInfo;
    std::string contextSignature;
    int exitCode = static_cast<int>(ExitCode::Success);

    try {
        material = loadMaterial(command.materialPath);
        applyDependencyOverrides(*material, command);
        if (command.maxDiffs > 0) {
            material->maxDiffPixels = command.maxDiffs;
        }
        variants = resolveVariants(*material, command.variants);
        auto context = GlContext::create();
        contextInfo = context->info();
        contextSignature = context->signature();
        Renderer renderer(*context);
        for (const auto& variant : variants) {
            try {
                auto result = renderer.render(*material, variant, command,
                                              command.pixelDir);
                if (result.failedPixels != 0) {
                    auto error =
                        makeError(ExitCode::Validation, "VALIDATION_MISMATCH", "validation",
                                  "pixel validation failed");
                    error.variant = variant.name;
                    error.detail = std::to_string(result.failedPixels) +
                                   " failing pixels across " +
                                   std::to_string(result.rules.size()) + " rules";
                    errors.push_back(error);
                }
                results.push_back(std::move(result));
            } catch (const MaterialError& error) {
                auto variantError = error;
                if (variantError.variant.empty()) {
                    variantError.variant = variant.name;
                }
                errors.push_back(variantError);
            }
        }
    } catch (const MaterialError& error) {
        errors.push_back(error);
    } catch (const std::exception& error) {
        errors.push_back(makeError(ExitCode::Runtime, "UNHANDLED_EXCEPTION", "runtime",
                                   error.what()));
    }

    if (!errors.empty()) {
        exitCode = static_cast<int>(errors.front().exitCode);
    }

    json variantArray = json::array();
    std::vector<json> stableVariants;
    std::uint64_t attachmentCount = 0;
    std::uint64_t pixelCount = 0;
    std::uint64_t passedPixels = 0;
    std::uint64_t failedPixels = 0;
    std::uint64_t failedVariants = 0;
    for (const auto& result : results) {
        attachmentCount += result.attachments.size();
        pixelCount += result.pixelCount;
        passedPixels += result.passedPixels;
        failedPixels += result.failedPixels;
        if (result.failedPixels != 0) {
            ++failedVariants;
        }
        stableVariants.push_back(stableVariantJson(result));
        variantArray.push_back(variantJson(result));
    }
    failedVariants += static_cast<std::uint64_t>(variants.size() - results.size());

    const auto stable =
        stableProjection(material ? &*material : nullptr, stableVariants, errors);
    const auto fingerprint = Sha256::hex(stable.dump());

    json errorArray = json::array();
    for (const auto& error : errors) {
        errorArray.push_back(errorJson(error));
    }

    json report{
        {"schema", "material-verify/1"},
        {"ok", errors.empty()},
        {"exit_code", exitCode},
        {"report_fingerprint", fingerprint},
        {"context_signature", contextSignature},
        {"material",
         material ? json{{"name", material->name},
                         {"metadata", material->metadataPath},
                         {"metadata_sha256", material->metadataSha256},
                         {"vertex_shader", material->vertexPath},
                         {"fragment_shader", material->fragmentPath},
                         {"width", material->width},
                         {"height", material->height},
                         {"dependencies", dependencyJson(*material)}}
                  : json(nullptr)},
        {"context",
         contextInfo ? json{{"api", contextInfo->api},
                            {"version", contextInfo->version},
                            {"vendor", contextInfo->vendor},
                            {"renderer", contextInfo->renderer},
                            {"glsl_version", contextInfo->glslVersion}}
                      : json(nullptr)},
        {"options",
         json{{"variants", command.variants},
              {"cache_dir", command.cacheDir},
              {"dump_pixels", command.pixelDir},
              {"deterministic", command.deterministic},
              {"max_diffs", command.maxDiffs}}},
        {"variants", std::move(variantArray)},
        {"errors", std::move(errorArray)},
    };

    const auto elapsed = command.deterministic
                             ? 0.0
                             : std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
    report["summary"] = json{
        {"variants_requested", variants.size()},
        {"variants_completed", results.size()},
        {"variants_failed", failedVariants},
        {"attachment_count", attachmentCount},
        {"pixel_count", pixelCount},
        {"validation_passed_pixels", passedPixels},
        {"validation_failed_pixels", failedPixels},
        {"all_passed", errors.empty()},
        {"duration_ms", elapsed},
    };

    const auto outputPath =
        command.outputPath.empty() ? defaultOutputPath(command.materialPath)
                                   : std::filesystem::path(command.outputPath);
    const auto writeCode = writeReport(outputPath, report);
    if (writeCode != 0) {
        return writeCode;
    }

    if (errors.empty()) {
        std::cout << json{{"ok", true},
                          {"output", outputPath.string()},
                          {"report_fingerprint", fingerprint},
                          {"summary", report["summary"]}}
                         .dump()
                  << "\n";
        return 0;
    }
    json stderrErrors = json::array();
    for (const auto& error : errors) {
        stderrErrors.push_back(errorJson(error));
    }
    std::cerr << json{{"ok", false},
                      {"output", outputPath.string()},
                      {"report_fingerprint", fingerprint},
                      {"errors", std::move(stderrErrors)}}
                     .dump(2)
              << "\n";
    return exitCode;
}

}  // namespace mv

int main(int argc, char** argv) {
    return mv::run(argc, argv);
}
