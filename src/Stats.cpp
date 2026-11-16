#include "Stats.h"

#include <fstream>

namespace {

std::string Number(double value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

std::string Indent(int spaces) {
    return std::string(static_cast<size_t>(spaces), ' ');
}

void AppendFrameJson(std::ostringstream& output, const FrameStats& stats,
                     int baseIndent, bool trailingComma) {
    const std::string indent = Indent(baseIndent);
    const std::string inner = Indent(baseIndent + 2);
    const std::string arrayIndent = Indent(baseIndent + 4);

    output << indent << "{\n";
    output << inner << "\"frame\": " << stats.frame << ",\n";
    output << inner << "\"animationTimeSeconds\": "
           << Number(stats.animationTimeSeconds) << ",\n";
    output << inner << "\"drawCalls\": " << stats.drawCalls << ",\n";
    output << inner << "\"shadowMapSizes\": [";
    for (size_t i = 0; i < stats.shadowMapSizes.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << stats.shadowMapSizes[i];
    }
    output << "],\n";
    output << inner << "\"cascadeSplitRatios\": [";
    for (size_t i = 0; i < stats.cascadeSplitRatios.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << Number(stats.cascadeSplitRatios[i]);
    }
    output << "],\n";
    output << inner << "\"cascadeSplitDistances\": [";
    for (size_t i = 0; i < stats.cascadeSplitDistances.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << Number(stats.cascadeSplitDistances[i]);
    }
    output << "],\n";
    output << inner << "\"slopeOffset\": " << Number(stats.slopeOffset) << ",\n";
    output << inner << "\"constOffset\": " << Number(stats.constOffset) << ",\n";
    output << inner << "\"deviceLossRecoveries\": " << stats.deviceLossRecoveries
           << ",\n";
    output << inner << "\"cascades\": [\n";
    for (size_t i = 0; i < stats.cascades.size(); ++i) {
        output << arrayIndent << "{\"index\": " << i
               << ", \"near\": " << Number(stats.cascadeSplitDistances[i])
               << ", \"far\": " << Number(stats.cascadeSplitDistances[i + 1])
               << ", \"shadowMapSize\": " << Renderer::kShadowMapSize << "}";
        if (i + 1 != stats.cascades.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << inner << "]\n";
    output << indent << "}" << (trailingComma ? "," : "") << "\n";
}

}  // namespace

std::string FrameBaseName(uint32_t frame, uint32_t totalFrames) {
    uint32_t digits = 4;
    for (uint32_t value = totalFrames > 0 ? totalFrames - 1 : 0; value >= 10000;
         value /= 10) {
        ++digits;
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "frame_" << std::setw(static_cast<int>(digits)) << std::setfill('0')
           << frame;
    return stream.str();
}

std::string BuildFrameJson(const FrameStats& stats) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    AppendFrameJson(output, stats, 0, false);
    return output.str();
}

std::string BuildAggregateJson(const AggregateStats& stats) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\n";
    output << "  \"program\": \"deterministic_csm\",\n";
    output << "  \"width\": " << stats.width << ",\n";
    output << "  \"height\": " << stats.height << ",\n";
    output << "  \"frameCount\": " << stats.frameCount << ",\n";
    output << "  \"fixedStepSeconds\": " << Number(1.0 / 60.0) << ",\n";
    output << "  \"vertexCount\": " << stats.vertexCount << ",\n";
    output << "  \"indexCount\": " << stats.indexCount << ",\n";
    output << "  \"drawCallsPerFrame\": " << Renderer::kDrawCallsPerFrame << ",\n";
    output << "  \"shadowMapSize\": " << Renderer::kShadowMapSize << ",\n";
    output << "  \"cascadeSplitRatios\": [";
    for (size_t i = 0; i < stats.cascadeSplitRatios.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << Number(stats.cascadeSplitRatios[i]);
    }
    output << "],\n";
    output << "  \"cascadeSplitDistances\": [";
    for (size_t i = 0; i < stats.cascadeSplitDistances.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << Number(stats.cascadeSplitDistances[i]);
    }
    output << "],\n";
    output << "  \"slopeOffset\": " << Number(stats.slopeOffset) << ",\n";
    output << "  \"constOffset\": " << Number(stats.constOffset) << ",\n";
    output << "  \"deviceLossInjectionAfterFrame\": ";
    if (stats.hasDeviceLossInjection) {
        output << stats.deviceLossInjectionFrame;
    } else {
        output << "null";
    }
    output << ",\n";
    output << "  \"deviceLossRecoveries\": " << stats.deviceLossRecoveries << ",\n";
    output << "  \"frames\": [\n";
    for (size_t i = 0; i < stats.frames.size(); ++i) {
        AppendFrameJson(output, stats.frames[i], 4, i + 1 != stats.frames.size());
    }
    output << "  ]\n";
    output << "}\n";
    return output.str();
}

void WriteTextFile(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create text file");
    }
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) {
        throw std::runtime_error("failed to write text file");
    }
}

