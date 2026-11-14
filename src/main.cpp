#include "common.hpp"
#include "material.hpp"
#include "renderer.hpp"
#include "sample_generator.hpp"
#include "scene.hpp"
#include "snapshot.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace cr {
namespace {

struct CliOptions {
    fs::path dataDirectory = "render_data";
    fs::path outputDirectory;
    RenderBackend backend = RenderBackend::Auto;
    bool forceRegenerate = false;
    bool help = false;
    std::string command;
    std::vector<std::string> arguments;
};

struct RenderResult {
    Image image;
    RenderBackend backend = RenderBackend::Cpu;
};

void printUsage() {
    std::cout
        << "character_renderer - deterministic layered-material deferred renderer\n\n"
        << "Usage:\n"
        << "  character_renderer [global options] [command] [arguments]\n\n"
        << "Global options:\n"
        << "  --data-dir PATH          generated samples, snapshots and state directory\n"
        << "  --output-dir PATH        render output directory (default: DATA/output)\n"
        << "  --backend auto|cpu|opengl\n"
        << "  --force-regenerate       regenerate every deterministic sample file\n"
        << "  --help\n\n"
        << "Commands (default: --render):\n"
        << "  --render\n"
        << "  --verify\n"
        << "  --list-materials\n"
        << "  --list-lights\n"
        << "  --add-light ID X,Y,Z R,G,B INTENSITY RADIUS\n"
        << "  --update-light ID FIELD VALUE\n"
        << "  --remove-light ID\n"
        << "  --reset-lights\n"
        << "  --create-snapshot\n"
        << "  --list-snapshots\n"
        << "  --show-snapshot VERSION\n"
        << "  --rollback VERSION\n"
        << "  --diff BASE_VERSION TARGET_VERSION\n"
        << "  --merge BASE_VERSION TARGET_VERSION\n";
}

float parseFloatStrict(const std::string& text, const std::string& context) {
    std::size_t consumed = 0;
    float value = 0.0F;
    try {
        value = std::stof(text, &consumed);
    } catch (const std::exception&) {
        throw AppError(E_DATA_INVALID, "invalid number for " + context + ": '" + text + "'");
    }
    if (consumed != text.size() || !std::isfinite(value)) {
        throw AppError(E_DATA_INVALID, "invalid number for " + context + ": '" + text + "'");
    }
    return value;
}

std::uint64_t parseVersion(const std::string& text, const std::string& context) {
    if (text.empty() ||
        !std::all_of(text.begin(), text.end(),
                     [](unsigned char character) { return std::isdigit(character) != 0; })) {
        throw AppError(E_DATA_INVALID, "invalid version for " + context + ": '" + text + "'");
    }
    try {
        return static_cast<std::uint64_t>(std::stoull(text));
    } catch (const std::exception&) {
        throw AppError(E_DATA_INVALID, "invalid version for " + context + ": '" + text + "'");
    }
}

std::vector<std::string> splitComma(const std::string& text, const std::string& context) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string::npos ? text.size() : comma;
        fields.push_back(text.substr(start, end - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (fields.size() != 3U) {
        throw AppError(E_DATA_INVALID, context + " requires three comma-separated numbers");
    }
    return fields;
}

Vec3 parseVec3(const std::string& text, const std::string& context) {
    const std::vector<std::string> fields = splitComma(text, context);
    return Vec3(parseFloatStrict(fields[0], context + ".x"),
                parseFloatStrict(fields[1], context + ".y"),
                parseFloatStrict(fields[2], context + ".z"));
}

Light* findLight(std::vector<Light>& lights, const std::string& id) {
    for (Light& light : lights) {
        if (light.id == id) {
            return &light;
        }
    }
    return nullptr;
}

void requireArgumentCount(const CliOptions& options, std::size_t count,
                          const std::string& usage) {
    if (options.arguments.size() != count) {
        throw AppError(E_DATA_INVALID, "usage: " + usage);
    }
}

std::string backendTag(RenderBackend backend) { return renderBackendName(backend); }

RenderResult renderOnce(const MaterialLibrary& materials, const Camera& camera,
                        const std::vector<Light>& lights, RenderBackend requested) {
    if (requested == RenderBackend::Cpu) {
        return {renderCpuDeferred(materials, camera, lights), RenderBackend::Cpu};
    }
#ifdef CHARACTER_RENDER_HAVE_OPENGL
    if (requested == RenderBackend::OpenGL) {
        return {renderOpenGLDeferred(materials, camera, lights), RenderBackend::OpenGL};
    }
    try {
        return {renderOpenGLDeferred(materials, camera, lights), RenderBackend::OpenGL};
    } catch (const AppError& error) {
        if (std::string(error.code()) != E_OPENGL_UNAVAILABLE) {
            throw;
        }
        std::cerr << "warning: OpenGL backend unavailable, using deterministic CPU "
                     "reference backend: "
                  << error.what() << '\n';
        return {renderCpuDeferred(materials, camera, lights), RenderBackend::Cpu};
    }
#else
    if (requested == RenderBackend::OpenGL) {
        throw AppError(E_OPENGL_UNAVAILABLE,
                       "this build does not include the OpenGL backend");
    }
    return {renderCpuDeferred(materials, camera, lights), RenderBackend::Cpu};
#endif
}

MaterialLibrary loadMaterials(const fs::path& dataDirectory, bool requireAll) {
    const std::vector<MaterialSpec> specs =
        parseMaterialDirectory(dataDirectory / "materials");
    MaterialLibrary library = loadMaterialLibrary(specs);
    if (requireAll) {
        for (const char* name : kRequiredMaterialNames) {
            (void)library.require(name);
        }
    }
    return library;
}

int commandRender(const CliOptions& options) {
    ensureGeneratedSamples(options.dataDirectory, options.forceRegenerate);
    const MaterialLibrary materials = loadMaterials(options.dataDirectory, true);
    const Camera camera = loadCamera(options.dataDirectory);
    SnapshotStore snapshots(options.dataDirectory,
                            loadDefaultLights(options.dataDirectory));
    snapshots.ensureInitialSnapshot();

    const RenderResult result =
        renderOnce(materials, camera, snapshots.currentLights(), options.backend);
    ensureDirectory(options.outputDirectory);
    const fs::path renderPath = options.outputDirectory / "render.png";
    savePngAtomic(renderPath, result.image);

    const fs::path referencePath =
        options.outputDirectory / ("reference_" + backendTag(result.backend) + ".png");
    bool referenceCreated = false;
    if (!pathExists(referencePath)) {
        savePngAtomic(referencePath, result.image);
        referenceCreated = true;
    }

    std::cout << "rendered backend=" << backendTag(result.backend)
              << " output=" << renderPath.string()
              << " reference=" << referencePath.string()
              << (referenceCreated ? " (created)" : " (kept)") << '\n';
    return 0;
}

int commandVerify(const CliOptions& options) {
    ensureGeneratedSamples(options.dataDirectory, options.forceRegenerate);
    const MaterialLibrary materials = loadMaterials(options.dataDirectory, true);
    const Camera camera = loadCamera(options.dataDirectory);
    SnapshotStore snapshots(options.dataDirectory,
                            loadDefaultLights(options.dataDirectory));
    snapshots.ensureInitialSnapshot();

    const RenderResult first =
        renderOnce(materials, camera, snapshots.currentLights(), options.backend);
    const RenderResult second =
        renderOnce(materials, camera, snapshots.currentLights(), first.backend);
    std::string difference;
    if (!imagesWithinTolerance(first.image, second.image, 1, &difference)) {
        throw AppError(E_RENDER_NONDETERMINISTIC,
                       "two fresh renders differ: " + difference);
    }

    ensureDirectory(options.outputDirectory);
    const fs::path referencePath =
        options.outputDirectory / ("reference_" + backendTag(first.backend) + ".png");
    bool referenceCreated = false;
    if (pathExists(referencePath)) {
        try {
            const Image reference = loadPng(referencePath);
            if (!imagesWithinTolerance(first.image, reference, 1, &difference)) {
                throw AppError(E_RENDER_NONDETERMINISTIC,
                               "render differs from saved reference: " + difference);
            }
        } catch (const AppError& error) {
            if (std::string(error.code()) == E_RENDER_NONDETERMINISTIC) {
                throw;
            }
            throw AppError(E_RENDER_NONDETERMINISTIC,
                           std::string("saved reference cannot be compared: ") +
                               error.what());
        }
    } else {
        savePngAtomic(referencePath, first.image);
        referenceCreated = true;
    }

    const fs::path renderPath = options.outputDirectory / "verify.png";
    savePngAtomic(renderPath, first.image);
    std::cout << "verify=ok backend=" << backendTag(first.backend)
              << " tolerance=1/255 output=" << renderPath.string()
              << " reference=" << referencePath.string()
              << (referenceCreated ? " (created)" : " (matched)") << '\n';
    return 0;
}

int commandListMaterials(const CliOptions& options) {
    ensureGeneratedSamples(options.dataDirectory, options.forceRegenerate);
    const MaterialLibrary materials = loadMaterials(options.dataDirectory, false);
    Json output = Json::array();
    for (const LoadedMaterial& material : materials.materials()) {
        output.push_back({{"name", material.name},
                          {"layers", material.layers.size()},
                          {"source", material.sourceFile.string()}});
    }
    std::cout << output.dump(2) << '\n';
    return 0;
}

int commandLights(const CliOptions&, SnapshotStore& snapshots) {
    Json output = Json::array();
    for (const Light& light : snapshots.currentLights()) {
        output.push_back(lightToJson(light));
    }
    std::cout << output.dump(2) << '\n';
    return 0;
}

int commandSnapshotList(const CliOptions&, SnapshotStore& snapshots) {
    Json output = Json::array();
    for (const auto& entry : snapshots.committed()) {
        const Snapshot snapshot = snapshots.loadSnapshot(entry.first);
        output.push_back({{"version", entry.first},
                          {"timestamp", entry.second.timestamp},
                          {"lights", snapshot.lights.size()},
                          {"file", entry.second.fileName}});
    }
    std::cout << Json{{"nextVersion", snapshots.nextVersion()}, {"snapshots", output}}.dump(2)
              << '\n';
    return 0;
}

int commandSnapshotShow(const CliOptions& options, SnapshotStore& snapshots) {
    requireArgumentCount(options, 1, "--show-snapshot VERSION");
    const std::uint64_t version = parseVersion(options.arguments[0], "snapshot");
    const Snapshot snapshot = snapshots.loadSnapshot(version);
    Json lights = Json::array();
    for (const Light& light : snapshot.lights) {
        lights.push_back(lightToJson(light));
    }
    std::cout << Json{{"version", snapshot.version},
                      {"timestamp", snapshot.timestamp},
                      {"lights", lights}}
                     .dump(2)
              << '\n';
    return 0;
}

int commandMutateLights(const CliOptions& options, SnapshotStore& snapshots) {
    std::vector<Light> lights = snapshots.currentLights();
    if (options.command == "--add-light") {
        requireArgumentCount(options, 5,
                             "--add-light ID X,Y,Z R,G,B INTENSITY RADIUS");
        if (findLight(lights, options.arguments[0]) != nullptr) {
            throw AppError(E_DATA_INVALID,
                           "light id already exists: " + options.arguments[0]);
        }
        Light light;
        light.id = options.arguments[0];
        light.position = parseVec3(options.arguments[1], "light.position");
        light.color = parseVec3(options.arguments[2], "light.color");
        light.intensity = parseFloatStrict(options.arguments[3], "light.intensity");
        light.radius = parseFloatStrict(options.arguments[4], "light.radius");
        if (light.id.empty() || light.intensity < 0.0F || light.radius <= 0.0F) {
            throw AppError(E_DATA_INVALID, "new light has invalid fields");
        }
        lights.push_back(light);
    } else if (options.command == "--update-light") {
        requireArgumentCount(options, 3, "--update-light ID FIELD VALUE");
        Light* light = findLight(lights, options.arguments[0]);
        if (light == nullptr) {
            throw AppError(E_DATA_INVALID,
                           "light does not exist: " + options.arguments[0]);
        }
        const std::string& field = options.arguments[1];
        const std::string& value = options.arguments[2];
        if (field == "position") {
            light->position = parseVec3(value, "light.position");
        } else if (field == "color") {
            light->color = parseVec3(value, "light.color");
        } else if (field == "intensity") {
            light->intensity = parseFloatStrict(value, "light.intensity");
        } else if (field == "radius") {
            light->radius = parseFloatStrict(value, "light.radius");
        } else {
            throw AppError(E_DATA_INVALID,
                           "unknown light field '" + field +
                               "' (expected position, color, intensity or radius)");
        }
        if (light->intensity < 0.0F || light->radius <= 0.0F) {
            throw AppError(E_DATA_INVALID, "updated light has invalid intensity or radius");
        }
    } else if (options.command == "--remove-light") {
        requireArgumentCount(options, 1, "--remove-light ID");
        const auto iterator =
            std::find_if(lights.begin(), lights.end(), [&](const Light& light) {
                return light.id == options.arguments[0];
            });
        if (iterator == lights.end()) {
            throw AppError(E_DATA_INVALID,
                           "light does not exist: " + options.arguments[0]);
        }
        lights.erase(iterator);
    } else if (options.command == "--reset-lights") {
        requireArgumentCount(options, 0, "--reset-lights");
        lights = loadDefaultLights(options.dataDirectory);
    }
    snapshots.setCurrentLights(std::move(lights));
    std::cout << "lights updated\n";
    return 0;
}

CliOptions parseCli(int argc, char** argv) {
    CliOptions options;
    int index = 1;
    while (index < argc) {
        const std::string argument = argv[index];
        if (argument == "--data-dir" || argument == "--output-dir" ||
            argument == "--backend") {
            if (index + 1 >= argc) {
                throw AppError(E_DATA_INVALID, "missing value for " + argument);
            }
            const std::string value = argv[++index];
            if (argument == "--data-dir") {
                options.dataDirectory = value;
            } else if (argument == "--output-dir") {
                options.outputDirectory = value;
            } else {
                options.backend = parseRenderBackend(value);
            }
            ++index;
            continue;
        }
        if (argument == "--force-regenerate") {
            options.forceRegenerate = true;
            ++index;
            continue;
        }
        if (argument == "--help" || argument == "-h") {
            options.help = true;
            ++index;
            continue;
        }
        if (!argument.empty() && argument[0] == '-') {
            if (!options.command.empty()) {
                throw AppError(E_DATA_INVALID, "unexpected extra option '" + argument + "'");
            }
            options.command = argument;
            ++index;
            continue;
        }
        if (options.command.empty()) {
            throw AppError(E_DATA_INVALID,
                           "command must start with '--': '" + argument + "'");
        }
        options.arguments.push_back(argument);
        ++index;
    }
    if (options.outputDirectory.empty()) {
        options.outputDirectory = options.dataDirectory / "output";
    }
    return options;
}

int run(int argc, char** argv) {
    CliOptions options = parseCli(argc, argv);
    if (options.help || options.command == "--help") {
        printUsage();
        return 0;
    }
    if (options.command.empty()) {
        options.command = "--render";
    }

    if (options.command == "--render") {
        requireArgumentCount(options, 0, "--render");
        return commandRender(options);
    }
    if (options.command == "--verify") {
        requireArgumentCount(options, 0, "--verify");
        return commandVerify(options);
    }
    if (options.command == "--list-materials") {
        requireArgumentCount(options, 0, "--list-materials");
        return commandListMaterials(options);
    }

    ensureGeneratedSamples(options.dataDirectory, options.forceRegenerate);
    SnapshotStore snapshots(options.dataDirectory,
                            loadDefaultLights(options.dataDirectory));

    if (options.command == "--list-lights") {
        requireArgumentCount(options, 0, "--list-lights");
        return commandLights(options, snapshots);
    }
    if (options.command == "--add-light" || options.command == "--update-light" ||
        options.command == "--remove-light" || options.command == "--reset-lights") {
        snapshots.ensureInitialSnapshot();
        return commandMutateLights(options, snapshots);
    }
    if (options.command == "--create-snapshot") {
        requireArgumentCount(options, 0, "--create-snapshot");
        const std::uint64_t version = snapshots.createSnapshot();
        std::cout << "snapshot=" << version << '\n';
        return 0;
    }
    if (options.command == "--list-snapshots") {
        requireArgumentCount(options, 0, "--list-snapshots");
        return commandSnapshotList(options, snapshots);
    }
    if (options.command == "--show-snapshot") {
        return commandSnapshotShow(options, snapshots);
    }
    if (options.command == "--rollback") {
        requireArgumentCount(options, 1, "--rollback VERSION");
        const std::uint64_t version = parseVersion(options.arguments[0], "rollback");
        snapshots.rollback(version);
        std::cout << "rolled_back=" << version << '\n';
        return 0;
    }
    if (options.command == "--diff") {
        requireArgumentCount(options, 2, "--diff BASE_VERSION TARGET_VERSION");
        const std::uint64_t base = parseVersion(options.arguments[0], "diff base");
        const std::uint64_t target = parseVersion(options.arguments[1], "diff target");
        std::cout << snapshots.diff(base, target).toJson().dump(2) << '\n';
        return 0;
    }
    if (options.command == "--merge") {
        requireArgumentCount(options, 2, "--merge BASE_VERSION TARGET_VERSION");
        const std::uint64_t base = parseVersion(options.arguments[0], "merge base");
        const std::uint64_t target = parseVersion(options.arguments[1], "merge target");
        snapshots.merge(base, target);
        std::cout << "merged=" << base << "->" << target << '\n';
        return 0;
    }

    throw AppError(E_DATA_INVALID, "unknown command '" + options.command + "'");
}

}  // namespace
}  // namespace cr

int main(int argc, char** argv) {
    try {
        return cr::run(argc, argv);
    } catch (const cr::AppError& error) {
        std::cerr << error.code() << ": " << error.what() << '\n';
        return cr::runProcessExitCodeForError();
    } catch (const std::exception& error) {
        std::cerr << cr::E_IO << ": unhandled exception: " << error.what() << '\n';
        return cr::runProcessExitCodeForError();
    }
}

