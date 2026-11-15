#include "test_support.h"

#include "automix/config.h"
#include "automix/drc.h"
#include "automix/json.h"
#include "automix/limiter.h"
#include "automix/logger.h"
#include "automix/loudness.h"
#include "automix/stereo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

void testJsonParser() {
    automix::JsonParseError error;
    const auto root = automix::parseJson(
        R"({"a":1,"b":[true,null,"x\u4e2d"],"c":{"d":-2.5e-3}})", error);
    test::require(error.message.empty(), "valid JSON reported an error");
    test::require(root.isObject(), "root should be object");
    test::require(root.find("a") != nullptr && root.find("a")->asNumber() == 1.0,
                  "number parse failed");
    test::require(root.find("b") != nullptr && root.find("b")->asArray().size() == 3,
                  "array parse failed");
    test::require(root.find("b")->asArray()[2].asString() == "x\u4e2d",
                  "Unicode escape parse failed");

    const auto bad = automix::parseJson("{\"a\":1,}", error);
    test::require(bad.isNull() && !error.message.empty(), "syntax error was not reported");
    test::require(error.line == 1 && error.column >= 1,
                  "syntax error location was not populated");
}

automix::StereoBuffer stepSignal(double attackMs, std::vector<double>& reduction) {
    automix::StereoBuffer buffer(48000, 48000);
    for (std::size_t frame = 2400; frame < buffer.frames; ++frame) {
        buffer.left(frame) = 0.9f;
        buffer.right(frame) = 0.9f;
    }
    automix::DrcConfig config;
    config.enabled = true;
    config.thresholdDb = -20.0;
    config.ratio = 4.0;
    config.attackMs = attackMs;
    config.releaseMs = 200.0;
    config.kneeDb = 0.0;
    automix::processDrc(buffer, config, &reduction);
    return buffer;
}

void testDrc() {
    automix::DrcConfig hard;
    hard.enabled = true;
    hard.thresholdDb = -20.0;
    hard.ratio = 4.0;
    hard.attackMs = 10.0;
    hard.releaseMs = 100.0;
    hard.kneeDb = 0.0;
    automix::DrcConfig soft = hard;
    soft.kneeDb = 8.0;

    const double hardAtThreshold = automix::drcStaticReductionDb(-20.0, hard);
    const double softAtThreshold = automix::drcStaticReductionDb(-20.0, soft);
    test::requireNear(hardAtThreshold, 0.0, 1e-12, "hard knee threshold mismatch");
    test::require(softAtThreshold > 0.5 && softAtThreshold < 0.9,
                  "soft knee steady-state gain is not in the expected range");
    test::requireNear(
        automix::drcStaticReductionDb(-20.0, soft), softAtThreshold, 1e-12,
        "soft knee reduction is not reproducible");

    auto steadyReduction = [](const automix::DrcConfig& config) {
        automix::StereoBuffer buffer(48000, 48000 * 2);
        std::fill(buffer.interleaved.begin(), buffer.interleaved.end(), 0.1f);
        std::vector<double> reduction;
        automix::processDrc(buffer, config, &reduction);
        return reduction.back();
    };
    const double hardSteady = steadyReduction(hard);
    const double softSteady = steadyReduction(soft);
    const double softSteadyAgain = steadyReduction(soft);
    test::require(std::abs(softSteady - hardSteady) > 0.6,
                  "hard and soft knee steady-state gains did not differ");
    test::requireNear(softSteadyAgain, softSteady, 1e-12,
                      "soft knee processed gain is not reproducible");

    std::vector<double> fast;
    std::vector<double> slow;
    stepSignal(5.0, fast);
    stepSignal(200.0, slow);
    test::require(fast.size() == slow.size(), "DRC gain curve size mismatch");
    for (std::size_t i = 2400; i < fast.size(); ++i) {
        test::require(fast[i] + 1e-12 >= slow[i],
                      "attack time monotonicity failed at sample " +
                          std::to_string(i));
    }
}

void testLimiter() {
    automix::StereoBuffer buffer(48000, 48000);
    for (std::size_t frame = 0; frame < buffer.frames; ++frame) {
        const double value =
            (frame == 20000 || frame == 20300) ? 1.8 : 0.2;
        buffer.left(frame) = static_cast<float>(value);
        buffer.right(frame) = static_cast<float>(-value);
    }
    automix::LimiterConfig config;
    config.enabled = true;
    config.thresholdDbfs = -6.0;
    config.lookaheadMs = 10.0;
    config.releaseMs = 80.0;

    const auto stats = automix::processLimiter(buffer, config);
    const auto levels = automix::measureLevels(buffer);
    const double threshold = automix::dbToLinear(-6.0);
    test::require(levels.peak <= threshold + 1e-7,
                  "limiter allowed an over-threshold sample");
    test::require(stats.lookaheadSamples == 480,
                  "limiter lookahead sample count mismatch");
    test::require(stats.triggeredSamples == 2,
                  "limiter trigger sample count mismatch");
    test::require(stats.maxAttenuationDb > 10.0,
                  "limiter attenuation is unexpectedly small");
}

void testLogger() {
    const auto directory = test::makeTempDirectory("logger");
    {
        automix::Logger logger(directory, "rotate.log", 512, 3);
        for (int index = 0; index < 80; ++index) {
            logger.info("rotation record " + std::to_string(index) +
                        " 012345678901234567890123456789");
        }
        test::require(!logger.degraded(), "logger unexpectedly degraded");
    }

    std::size_t fileCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().filename().string().rfind("rotate.log", 0) == 0) {
            ++fileCount;
        }
    }
    test::require(fileCount == 3, "logger did not retain exactly three files");

    const auto notADirectory = directory / "plain-file";
    {
        std::ofstream(notADirectory) << "x";
    }
    automix::Logger degraded(notADirectory, "x.log", 512, 3);
    degraded.info("this must not throw");
    test::require(degraded.degraded(), "unwritable log path did not degrade");
    std::filesystem::remove_all(directory);
}

void testLoudness() {
    automix::StereoBuffer buffer(48000, 48000 * 4);
    constexpr double pi = 3.14159265358979323846;
    const double amplitude = automix::dbToLinear(-20.0);
    for (std::size_t frame = 0; frame < buffer.frames; ++frame) {
        const double value =
            amplitude * std::sin(2.0 * pi * 997.0 * static_cast<double>(frame) /
                                 48000.0 + 0.3);
        buffer.left(frame) = static_cast<float>(value);
        buffer.right(frame) = static_cast<float>(value);
    }
    const auto result = automix::measureBs1770(buffer);
    test::require(result.hasIntegratedValue, "loudness gating produced no value");
    test::requireNear(result.integratedLufs, -20.0, 0.5,
                      "integrated loudness reference mismatch");
    test::requireNear(result.truePeakDbtp, -20.0, 0.5,
                      "true peak reference mismatch");
}

}  // namespace

void testResampler();

int main() {
    struct TestCase {
        const char* name;
        std::function<void()> function;
    };
    const std::vector<TestCase> tests = {
        {"json", testJsonParser},
        {"drc", testDrc},
        {"limiter", testLimiter},
        {"logger", testLogger},
        {"loudness", testLoudness},
        {"resampler", testResampler},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.function();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    std::cout << "RESULT " << (tests.size() - failures) << "/" << tests.size()
              << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
