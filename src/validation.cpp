#include "validation.hpp"

#include "util.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace mv {
namespace {

int channelIndex(const std::string& channel) {
    if (channel == "r" || channel == "x" || channel == "0") {
        return 0;
    }
    if (channel == "g" || channel == "y" || channel == "1") {
        return 1;
    }
    if (channel == "b" || channel == "z" || channel == "2") {
        return 2;
    }
    if (channel == "a" || channel == "w" || channel == "3") {
        return 3;
    }
    throw makeError(ExitCode::Input, "RULE_CHANNEL_INVALID", "input",
                    "invalid channel selector: " + channel);
}

std::vector<int> channelList(const std::string& channels) {
    std::vector<int> result;
    for (const auto c : channels) {
        const auto text = std::string(1, c);
        const auto index = channelIndex(text);
        if (index > 2) {
            throw makeError(ExitCode::Input, "RULE_CHANNEL_INVALID", "input",
                            "texture channel lists support only r, g and b");
        }
        result.push_back(index);
    }
    if (result.empty()) {
        throw makeError(ExitCode::Input, "RULE_CHANNEL_INVALID", "input",
                        "channel list cannot be empty");
    }
    return result;
}

const RawAttachment* findAttachment(const std::vector<RawAttachment>& attachments,
                                    const std::string& name,
                                    std::string_view ruleId) {
    const auto found = std::find_if(
        attachments.begin(), attachments.end(),
        [&](const RawAttachment& attachment) { return attachment.name == name; });
    if (found == attachments.end()) {
        throw makeError(ExitCode::Input, "RULE_ATTACHMENT_MISSING", "input",
                        "rule " + std::string(ruleId) + " references missing readback: " +
                            name);
    }
    return &*found;
}

float attachmentChannelAt(const RawAttachment& attachment, int channel, int pixel) {
    if (attachment.kind == "depth") {
        return attachment.depth.at(static_cast<std::size_t>(pixel));
    }
    if (attachment.kind == "stencil") {
        return static_cast<float>(attachment.stencil.at(static_cast<std::size_t>(pixel)));
    }
    return attachment.rgba.at(static_cast<std::size_t>(pixel) * 4u +
                              static_cast<std::size_t>(channel));
}

float depthAt(const RawAttachment& depth, int x, int y) {
    const auto clampedX = std::clamp(x, 0, depth.width - 1);
    const auto clampedY = std::clamp(y, 0, depth.height - 1);
    return depth.depth.at(static_cast<std::size_t>(clampedY) *
                              static_cast<std::size_t>(depth.width) +
                          static_cast<std::size_t>(clampedX));
}

glm::vec3 reconstructNormal(const RawAttachment& depth, int x, int y) {
    const auto leftX = x > 0 ? x - 1 : x;
    const auto rightX = x + 1 < depth.width ? x + 1 : x;
    const auto downY = y > 0 ? y - 1 : y;
    const auto upY = y + 1 < depth.height ? y + 1 : y;
    const auto spanX = std::max(1, rightX - leftX);
    const auto spanY = std::max(1, upY - downY);
    const auto dzdx = (depthAt(depth, rightX, y) - depthAt(depth, leftX, y)) /
                      static_cast<float>(spanX);
    const auto dzdy = (depthAt(depth, x, upY) - depthAt(depth, x, downY)) /
                      static_cast<float>(spanY);
    const auto px = 2.0f / static_cast<float>(depth.width);
    const auto py = 2.0f / static_cast<float>(depth.height);
    const glm::vec3 tangentX(px * static_cast<float>(spanX), 0.0f, dzdx);
    const glm::vec3 tangentY(0.0f, py * static_cast<float>(spanY), dzdy);
    auto normal = glm::cross(tangentX, tangentY);
    const auto length = glm::length(normal);
    if (length < 1e-8f) {
        return glm::vec3(0.0f, 0.0f, 1.0f);
    }
    return normal / length;
}

void recordFailure(RuleResult& result,
                   int x,
                   int y,
                   std::vector<float> expected,
                   std::vector<float> actual,
                   float delta,
                   const std::string& source,
                   const std::string& target,
                   int maxDifferences) {
    ++result.failedPixels;
    if (static_cast<int>(result.differences.size()) >= maxDifferences) {
        result.differencesTruncated = true;
        return;
    }
    PixelDifference difference;
    difference.x = x;
    difference.y = y;
    difference.expected = std::move(expected);
    difference.actual = std::move(actual);
    difference.delta = delta;
    difference.source = source;
    difference.target = target;
    result.differences.push_back(std::move(difference));
}

}  // namespace

ValidationSummary evaluateRules(const MaterialSpec& material,
                                const std::vector<RawAttachment>& attachments,
                                int maxDifferences) {
    ValidationSummary summary;
    const auto width = material.width;
    const auto height = material.height;
    summary.pixelCount = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    std::vector<std::uint8_t> failedUnion(static_cast<std::size_t>(summary.pixelCount), 0);

    for (const auto& rule : material.rules) {
        RuleResult result;
        result.id = rule.id;
        result.type = rule.type;
        result.testedPixels = summary.pixelCount;

        if (rule.type == "channel_match") {
            const auto& source =
                *findAttachment(attachments, rule.source, rule.id);
            const auto& target =
                *findAttachment(attachments, rule.target, rule.id);
            const auto sourceChannel = channelIndex(rule.sourceChannel);
            const auto targetChannel = channelIndex(rule.targetChannel);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto pixel = y * width + x;
                    const auto expected =
                        attachmentChannelAt(source, sourceChannel, pixel);
                    const auto actual =
                        attachmentChannelAt(target, targetChannel, pixel);
                    const auto delta = std::fabs(expected - actual);
                    if (delta > rule.tolerance) {
                        failedUnion[static_cast<std::size_t>(pixel)] = 1;
                        recordFailure(result, x, y, {expected}, {actual}, delta,
                                      rule.source, rule.target, maxDifferences);
                    }
                }
            }
        } else if (rule.type == "depth_normal") {
            const auto& depth = *findAttachment(attachments, rule.source, rule.id);
            const auto& normal = *findAttachment(attachments, rule.target, rule.id);
            if (depth.kind != "depth") {
                throw makeError(ExitCode::Input, "RULE_DEPTH_ATTACHMENT_INVALID", "input",
                                "depth_normal source must be a depth readback: " + rule.id);
            }
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto pixel = y * width + x;
                    const auto reconstructed = reconstructNormal(depth, x, y);
                    glm::vec3 actual{
                        attachmentChannelAt(normal, 0, pixel) * 2.0f - 1.0f,
                        attachmentChannelAt(normal, 1, pixel) * 2.0f - 1.0f,
                        attachmentChannelAt(normal, 2, pixel) * 2.0f - 1.0f};
                    const auto length = glm::length(actual);
                    if (length > 1e-8f) {
                        actual /= length;
                    }
                    const auto delta = std::max(
                        {std::fabs(reconstructed.x - actual.x),
                         std::fabs(reconstructed.y - actual.y),
                         std::fabs(reconstructed.z - actual.z)});
                    if (delta > rule.tolerance) {
                        failedUnion[static_cast<std::size_t>(pixel)] = 1;
                        recordFailure(result, x, y,
                                      {reconstructed.x, reconstructed.y, reconstructed.z},
                                      {actual.x, actual.y, actual.z}, delta,
                                      rule.source, rule.target, maxDifferences);
                    }
                }
            }
        } else if (rule.type == "texture_sample") {
            const auto& target =
                *findAttachment(attachments, rule.attachment, rule.id);
            const auto textureIt = std::find_if(
                material.textures.begin(), material.textures.end(),
                [&](const TextureSpec& texture) { return texture.name == rule.texture; });
            if (textureIt == material.textures.end()) {
                throw makeError(ExitCode::Input, "RULE_TEXTURE_MISSING", "input",
                                "texture not found for rule: " + rule.id);
            }
            const auto textureChannels = channelList(rule.channels);
            const auto targetChannels = channelList(rule.attachmentChannel.empty()
                                                        ? rule.channels
                                                        : rule.attachmentChannel);
            if (textureChannels.size() != targetChannels.size()) {
                throw makeError(ExitCode::Input, "RULE_CHANNEL_INVALID", "input",
                                "texture_sample channel counts differ: " + rule.id);
            }
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto pixel = y * width + x;
                    const auto tx = x % textureIt->width;
                    const auto ty = y % textureIt->height;
                    const auto texel = proceduralTexel(*textureIt, tx, ty);
                    std::vector<float> expected;
                    std::vector<float> actual;
                    float delta = 0.0f;
                    for (std::size_t i = 0; i < textureChannels.size(); ++i) {
                        expected.push_back(
                            static_cast<float>(texel[static_cast<std::size_t>(
                                textureChannels[i])]) /
                            255.0f);
                        actual.push_back(
                            attachmentChannelAt(target, targetChannels[i], pixel));
                        delta = std::max(delta, std::fabs(expected.back() - actual.back()));
                    }
                    if (delta > rule.tolerance) {
                        failedUnion[static_cast<std::size_t>(pixel)] = 1;
                        recordFailure(result, x, y, expected, actual, delta,
                                      "texture:" + rule.texture, rule.attachment,
                                      maxDifferences);
                    }
                }
            }
        } else if (rule.type == "constant_match") {
            if (!material.constants) {
                throw makeError(ExitCode::Input, "RULE_CONSTANT_MISSING", "input",
                                "constant buffer not found for rule: " + rule.id);
            }
            const auto constantIt = material.constants->values.find(rule.constant);
            if (constantIt == material.constants->values.end()) {
                throw makeError(ExitCode::Input, "RULE_CONSTANT_MISSING", "input",
                                "constant not found for rule: " + rule.id);
            }
            const auto& target =
                *findAttachment(attachments, rule.attachment, rule.id);
            const auto targetChannel = channelIndex(rule.attachmentChannel);
            const auto expected =
                constantIt->second[static_cast<std::size_t>(rule.component)];
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto pixel = y * width + x;
                    const auto actual =
                        attachmentChannelAt(target, targetChannel, pixel);
                    const auto delta = std::fabs(expected - actual);
                    if (delta > rule.tolerance) {
                        failedUnion[static_cast<std::size_t>(pixel)] = 1;
                        recordFailure(result, x, y, {expected}, {actual}, delta,
                                      "constant:" + rule.constant, rule.attachment,
                                      maxDifferences);
                    }
                }
            }
        } else if (rule.type == "channel_expected") {
            const auto& target =
                *findAttachment(attachments, rule.attachment, rule.id);
            const auto targetChannel = channelIndex(rule.attachmentChannel);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto pixel = y * width + x;
                    const auto actual =
                        attachmentChannelAt(target, targetChannel, pixel);
                    const auto delta = std::fabs(rule.expected - actual);
                    if (delta > rule.tolerance) {
                        failedUnion[static_cast<std::size_t>(pixel)] = 1;
                        recordFailure(result, x, y, {rule.expected}, {actual}, delta,
                                      "expected", rule.attachment, maxDifferences);
                    }
                }
            }
        } else if (rule.type == "depth_expected" || rule.type == "stencil_expected") {
            const auto name = rule.type == "depth_expected" ? "depth" : "stencil";
            const auto& target = *findAttachment(attachments, name, rule.id);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto pixel = y * width + x;
                    const auto actual = attachmentChannelAt(target, 0, pixel);
                    const auto delta = std::fabs(rule.expected - actual);
                    if (delta > rule.tolerance) {
                        failedUnion[static_cast<std::size_t>(pixel)] = 1;
                        recordFailure(result, x, y, {rule.expected}, {actual}, delta,
                                      "expected", name, maxDifferences);
                    }
                }
            }
        }

        result.passedPixels = result.testedPixels - result.failedPixels;
        result.passed = result.failedPixels == 0;
        summary.rules.push_back(std::move(result));
    }

    summary.failedPixels = static_cast<std::uint64_t>(
        std::count(failedUnion.begin(), failedUnion.end(), static_cast<std::uint8_t>(1)));
    summary.passedPixels = summary.pixelCount - summary.failedPixels;
    return summary;
}

}  // namespace mv
