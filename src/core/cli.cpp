#include "lightcluster/cli.hpp"

#include <charconv>
#include <cstdint>
#include <system_error>
#include <string>
#include <string_view>

namespace lightcluster {

namespace {

bool parseU64(std::string_view text, std::uint64_t& value) {
    if (text.empty()) {
        return false;
    }
    int base = 10;
    const char* begin = text.data();
    if (text.size() >= 2 && text[0] == '0' &&
        (text[1] == 'x' || text[1] == 'X')) {
        base = 16;
        begin += 2;
    }
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(begin, end, value, base);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parseU32(std::string_view text, std::uint32_t& value) {
    std::uint64_t wide = 0;
    if (!parseU64(text, wide) ||
        wide > static_cast<std::uint64_t>(UINT32_MAX)) {
        return false;
    }
    value = static_cast<std::uint32_t>(wide);
    return true;
}

bool splitOption(
    std::string_view argument,
    std::string_view& name,
    std::string_view& inlineValue,
    bool& hasInlineValue) {
    if (argument.size() < 2 || argument.substr(0, 2) != "--") {
        return false;
    }
    const std::size_t equal = argument.find('=');
    if (equal == std::string_view::npos) {
        name = argument;
        inlineValue = {};
        hasInlineValue = false;
        return true;
    }
    name = argument.substr(0, equal);
    inlineValue = argument.substr(equal + 1);
    hasInlineValue = true;
    return true;
}

}  // namespace

CliParseResult parseCommandLine(
    int argc,
    char** argv,
    CliOptions& options,
    std::string& error) {
    options = CliOptions{};
    options.run.syncMode = SyncMode::Immediate;
    options.run.delayFrames = 0;

    for (int index = 1; index < argc; ++index) {
        const std::string_view raw = argv[index];
        if (raw == "--help" || raw == "-h") {
            return CliParseResult::Help;
        }
        if (raw == "--version") {
            error = "lightcluster-deferred 1.0.0";
            return CliParseResult::Help;
        }

        std::string_view name;
        std::string_view inlineValue;
        bool hasInlineValue = false;
        if (!splitOption(raw, name, inlineValue, hasInlineValue)) {
            error = "unexpected positional argument: " + std::string(raw);
            return CliParseResult::Error;
        }

        const bool isFlag =
            name == "--debug" || name == "--inject-debug-fault" ||
            name == "--headless";
        if (isFlag && !hasInlineValue) {
            if (name == "--debug") {
                options.run.debug = true;
            } else if (name == "--inject-debug-fault") {
                options.run.injectDebugFault = true;
                options.run.debug = true;
            } else {
                options.run.headless = true;
            }
            continue;
        }

        std::string_view value = inlineValue;
        if (!hasInlineValue) {
            if (index + 1 >= argc) {
                error = "missing value for " + std::string(name);
                return CliParseResult::Error;
            }
            value = argv[++index];
        }

        if (name == "--frames") {
            if (!parseU32(value, options.run.frames) ||
                options.run.frames == 0) {
                error = "--frames must be a positive 32-bit integer";
                return CliParseResult::Error;
            }
        } else if (name == "--sync") {
            if (value == "immediate") {
                options.run.syncMode = SyncMode::Immediate;
            } else if (value == "deferred") {
                options.run.syncMode = SyncMode::Deferred;
            } else {
                error = "--sync must be immediate or deferred";
                return CliParseResult::Error;
            }
        } else if (name == "--delay") {
            if (!parseU32(value, options.run.delayFrames)) {
                error = "--delay must be a non-negative 32-bit integer";
                return CliParseResult::Error;
            }
        } else if (name == "--pool-capacity") {
            std::uint32_t capacity = 0;
            if (!parseU32(value, capacity) || capacity == 0) {
                error = "--pool-capacity must be a positive 32-bit integer";
                return CliParseResult::Error;
            }
            options.run.animationPoolCapacity = capacity;
        } else if (name == "--seed") {
            if (!parseU64(value, options.run.seed)) {
                error = "--seed must be an unsigned 64-bit integer";
                return CliParseResult::Error;
            }
        } else if (name == "--save-snapshot-frame") {
            std::uint32_t frame = 0;
            if (!parseU32(value, frame)) {
                error = "--save-snapshot-frame must be a 32-bit integer";
                return CliParseResult::Error;
            }
            options.run.saveSnapshotFrame = frame;
        } else if (name == "--restore-snapshot-frame") {
            std::uint32_t frame = 0;
            if (!parseU32(value, frame)) {
                error = "--restore-snapshot-frame must be a 32-bit integer";
                return CliParseResult::Error;
            }
            options.run.restoreSnapshotFrame = frame;
        } else if (name == "--debug") {
            std::uint64_t enabled = 0;
            if (!parseU64(value, enabled) || enabled > 1) {
                error = "--debug expects 0 or 1";
                return CliParseResult::Error;
            }
            options.run.debug = enabled != 0;
        } else if (name == "--inject-debug-fault") {
            std::uint64_t enabled = 0;
            if (!parseU64(value, enabled) || enabled > 1) {
                error = "--inject-debug-fault expects 0 or 1";
                return CliParseResult::Error;
            }
            options.run.injectDebugFault = enabled != 0;
            if (options.run.injectDebugFault) {
                options.run.debug = true;
            }
        } else if (name == "--headless") {
            std::uint64_t enabled = 0;
            if (!parseU64(value, enabled) || enabled > 1) {
                error = "--headless expects 0 or 1";
                return CliParseResult::Error;
            }
            options.run.headless = enabled != 0;
        } else {
            error = "unknown option: " + std::string(name);
            return CliParseResult::Error;
        }
    }

    if (options.run.syncMode == SyncMode::Immediate) {
        options.run.delayFrames = 0;
    }
    return CliParseResult::Ok;
}

std::string_view usageText() noexcept {
    return
        "Usage: lightcluster [options]\n"
        "  --frames N                 Frames to run (default 120)\n"
        "  --sync immediate|deferred  Light-buffer synchronization mode\n"
        "  --delay N                  Max deferred upload delay (frames)\n"
        "  --pool-capacity N          Fixed animation task slots (default 8)\n"
        "  --seed N                   Deterministic integer seed\n"
        "  --save-snapshot-frame N    Save a full scene snapshot after frame N\n"
        "  --restore-snapshot-frame N Restore it at start of frame N (> save)\n"
        "  --debug[=0|1]              Enable per-frame GPU readback validation\n"
        "  --inject-debug-fault[=0|1] Corrupt one readback for exit-code testing\n"
        "  --headless[=0|1]           Create a hidden OpenGL window\n"
        "  --help, --version\n";
}

}  // namespace lightcluster
