#include "core/benchmark.hpp"
#include "core/error.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    try {
        uid166::BenchmarkOptions options;
        options.data.dataDir = "artifacts/generated";
        options.outputPath = "artifacts/uid166_benchmark.json";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--frames" && i + 1 < argc) options.frames = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--objects" && i + 1 < argc) options.data.objectCount = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--output" && i + 1 < argc) options.outputPath = argv[++i];
            else if (arg == "--data-dir" && i + 1 < argc) options.data.dataDir = argv[++i];
            else if (arg == "--reuse-data") options.data.reuseExisting = true;
            else if (arg == "--no-gpu-load") options.simulateGpuLoad = false;
            else if (arg == "--help") {
                std::cout << "uid166 --frames N --objects N --data-dir DIR --output FILE "
                             "[--reuse-data] [--no-gpu-load]\n";
                return 0;
            }
        }
        auto data = uid166::prepareSceneData(options.data);
        const auto summary = uid166::runBenchmark(options, std::move(data));
        std::cout << "benchmark output=" << options.outputPath.string()
                  << " frameStatsHash=" << summary.frameStatsHash
                  << " validationPassed=" << (summary.validationPassed ? "true" : "false")
                  << '\n';
        return 0;
    } catch (const uid166::AppError& error) {
        uid166::reportError(error);
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] code=5000 name=INTERNAL message=" << error.what() << '\n';
        return 5000;
    }
}

