#include "indoor/DeferredRenderer.hpp"
#include "indoor/Scene.hpp"

#include <glad/gl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif

namespace {

int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void info(const std::string& message) {
    std::cout << "[gpu] " << message << '\n';
}

void ctxRecreate();

class EglContext {
public:
    EglContext() = default;
    ~EglContext() {
        destroySurfaceAndContext();
        if (display_ != EGL_NO_DISPLAY) {
            eglTerminate(display_);
        }
    }

    EglContext(const EglContext&) = delete;
    EglContext& operator=(const EglContext&) = delete;

    void create() {
        auto getPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (getPlatformDisplay != nullptr) {
            display_ = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                          EGL_DEFAULT_DISPLAY, nullptr);
        }
        if (display_ == EGL_NO_DISPLAY) {
            display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        }
        if (display_ == EGL_NO_DISPLAY || eglInitialize(display_, nullptr, nullptr) == EGL_FALSE) {
            throw std::runtime_error("Unable to initialize headless EGL display");
        }
        if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
            throw std::runtime_error("EGL desktop OpenGL API is unavailable");
        }

        const EGLint configAttributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_NONE};
        EGLint configCount = 0;
        if (eglChooseConfig(display_, configAttributes, &config_, 1, &configCount) == EGL_FALSE ||
            configCount < 1) {
            throw std::runtime_error("No EGL OpenGL configuration");
        }
        createSurfaceAndContext();
    }

    void recreateAfterLoss() {
        destroySurfaceAndContext();
        createSurfaceAndContext();
    }

    [[nodiscard]] EGLDisplay display() const { return display_; }

private:
    void createSurfaceAndContext() {
        const EGLint surfaceAttributes[] = {
            EGL_WIDTH, 16,
            EGL_HEIGHT, 16,
            EGL_NONE};
        surface_ = eglCreatePbufferSurface(display_, config_, surfaceAttributes);
        if (surface_ == EGL_NO_SURFACE) {
            throw std::runtime_error("Unable to create EGL pbuffer surface");
        }
        const EGLint contextAttributes[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 5,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE};
        context_ = eglCreateContext(display_, config_, EGL_NO_CONTEXT,
                                    contextAttributes);
        if (context_ == EGL_NO_CONTEXT) {
            throw std::runtime_error("Unable to create OpenGL 4.5 EGL context");
        }
        if (eglMakeCurrent(display_, surface_, surface_, context_) == EGL_FALSE) {
            throw std::runtime_error("eglMakeCurrent failed");
        }
        if (gladLoadGL(reinterpret_cast<GLADloadfunc>(eglGetProcAddress)) == 0) {
            throw std::runtime_error("gladLoadGLLoader failed");
        }
    }

    void destroySurfaceAndContext() {
        if (display_ == EGL_NO_DISPLAY) {
            return;
        }
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context_ != EGL_NO_CONTEXT) {
            eglDestroyContext(display_, context_);
            context_ = EGL_NO_CONTEXT;
        }
        if (surface_ != EGL_NO_SURFACE) {
            eglDestroySurface(display_, surface_);
            surface_ = EGL_NO_SURFACE;
        }
    }

    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLConfig config_ = nullptr;
    EGLSurface surface_ = EGL_NO_SURFACE;
    EGLContext context_ = EGL_NO_CONTEXT;
};

void writePpm(const std::filesystem::path& path,
              const std::vector<std::uint8_t>& rgba, int width, int height) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Unable to write " + path.string());
    }
    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t index =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) * 4U;
            output.write(reinterpret_cast<const char*>(&rgba[index]), 3);
        }
    }
}

struct SceneRun {
    indoor::FrameSnapshot snapshot;
    std::vector<std::uint8_t> pixels;
};

SceneRun renderFrame(indoor::DeferredRenderer& renderer, indoor::Scene& scene,
                     indoor::Camera& camera, indoor::MaterialPresetId preset,
                     const std::vector<std::uint32_t>& dirty,
                     std::uint64_t frameIndex, bool capture,
                     indoor::DebugView debugView = indoor::DebugView::Final) {
    SceneRun result;
    result.snapshot = renderer.prepareFrame(scene, camera, preset, dirty,
                                            frameIndex);
    renderer.render(scene, result.snapshot, debugView);
    if (capture) {
        result.pixels = renderer.readFinalPixels();
    }
    return result;
}

std::unordered_map<std::uint32_t, std::uint64_t> versionsFrom(
    const indoor::FrameSnapshot& snapshot) {
    std::unordered_map<std::uint32_t, std::uint64_t> result;
    for (const indoor::LightSnapshot& light : snapshot.lights) {
        result[light.light.id] = light.shadowVersion;
    }
    return result;
}

void testRenderer(const std::filesystem::path& artifacts) {
    indoor::RenderSettings settings;
    settings.width = 320;
    settings.height = 180;
    settings.shadowSize = 32;
    settings.shadowCacheCapacity = 48;

    indoor::Scene scene = indoor::buildDeterministicScene(1337, 48);
    indoor::Camera camera;
    indoor::DeferredRenderer renderer;
    renderer.initialize(scene, settings, INDOOR_SHADER_DIR);
    info("scene digest=" + indoor::sceneDigestHex(scene) +
         " lights=" + std::to_string(scene.lights.size()) +
         " objects=" + std::to_string(scene.objects.size()));

    SceneRun initial = renderFrame(renderer, scene, camera,
                                   indoor::MaterialPresetId::FullLayered, {},
                                   1, true);
    require(initial.pixels.size() == 320U * 180U * 4U,
            "initial readback dimensions");
    require(renderer.stats().totalLights == 48,
            "debug stats total light count");
    require(renderer.stats().residentShadowMaps == 48,
            "initial frame must allocate every resident shadow map");
    require(renderer.stats().validShadowMaps == 48,
            "initial frame shadow versions must all be valid");
    require(renderer.stats().shadowMapsRenderedThisFrame == 48,
            "initial shadow preparation renders all requested maps");
    require(renderer.stats().versionMigrations == 0,
            "initial allocation must not fabricate version migrations");
    for (const indoor::LightSnapshot& light : initial.snapshot.lights) {
        require(light.shadowVersion == 1,
                "each light starts at shadow version one");
    }
    writePpm(artifacts / "final_initial.ppm", initial.pixels, 320, 180);
    info("initial pixels digest=0x" + std::to_string(indoor::pixelDigest(initial.pixels)));

    const std::vector<std::pair<indoor::DebugView, std::string>> debugViews = {
        {indoor::DebugView::WorldPosition, "gbuffer_world.ppm"},
        {indoor::DebugView::ViewNormal, "gbuffer_normal.ppm"},
        {indoor::DebugView::Albedo, "gbuffer_albedo.ppm"},
        {indoor::DebugView::RoughMetal, "gbuffer_roughmetal.ppm"},
        {indoor::DebugView::ShadowVersions, "shadow_versions.ppm"},
        {indoor::DebugView::CacheOccupancy, "cache_occupancy.ppm"}};
    for (const auto& [view, filename] : debugViews) {
        renderer.render(scene, initial.snapshot, view);
        const std::vector<std::uint8_t> pixels = renderer.readFinalPixels();
        writePpm(artifacts / filename, pixels, 320, 180);
        require(indoor::pixelDigest(pixels) != indoor::pixelDigest(initial.pixels),
                std::string("debug channel must differ from final: ") + filename);
    }

    std::vector<std::uint64_t> presetDigests;
    for (int preset = 0; preset < 5; ++preset) {
        SceneRun presetRun = renderFrame(
            renderer, scene, camera,
            static_cast<indoor::MaterialPresetId>(preset), {}, 1, true);
        presetDigests.push_back(indoor::pixelDigest(presetRun.pixels));
        writePpm(artifacts / ("material_preset_" + std::to_string(preset) + ".ppm"),
                 presetRun.pixels, 320, 180);
    }
    std::vector<std::uint64_t> uniquePresets = presetDigests;
    std::sort(uniquePresets.begin(), uniquePresets.end());
    uniquePresets.erase(std::unique(uniquePresets.begin(), uniquePresets.end()),
                        uniquePresets.end());
    require(uniquePresets.size() == 5,
            "all five material-layer presets must change rendered output");
    info("five material presets produced five distinct final images");

    SceneRun baseline = renderFrame(renderer, scene, camera,
                                    indoor::MaterialPresetId::FullLayered, {},
                                    1, true);
    require(indoor::countPixelDifferences(initial.pixels, baseline.pixels) == 0,
            "replaying the same frame must be pixel-identical");

    const std::vector<indoor::MovedObject> moved =
        indoor::setAnimationTime(scene, 1.0 / 30.0);
    const std::vector<std::uint32_t> dirty =
        indoor::findAffectedLights(scene.lights, moved);
    require(!moved.empty() && !dirty.empty(),
            "animation must move objects and dirty lights");
    require(dirty.size() < scene.lights.size(),
            "movement must dirty only a local subset of all lights");
    const auto versionsBeforeMove = versionsFrom(baseline.snapshot);
    const std::uint64_t migrationsBeforeMove =
        renderer.versionLedger().migrations();
    SceneRun movedRun = renderFrame(renderer, scene, camera,
                                    indoor::MaterialPresetId::FullLayered, dirty,
                                    2, true);
    const auto versionsAfterMove = versionsFrom(movedRun.snapshot);
    require(renderer.stats().dirtyLightIds == dirty,
            "stats dirty set must exactly match plan");
    require(renderer.stats().shadowMapsRenderedThisFrame == dirty.size(),
            "only affected shadow maps may be re-rendered");
    require(renderer.stats().versionMigrationsThisFrame == dirty.size(),
            "only affected versions may migrate");
    require(renderer.versionLedger().migrations() ==
                migrationsBeforeMove + dirty.size(),
            "cumulative migration count must match dirty maps");
    for (const indoor::Light& light : scene.lights) {
        const bool affected =
            std::find(dirty.begin(), dirty.end(), light.id) != dirty.end();
        if (affected) {
            require(versionsAfterMove.at(light.id) ==
                        versionsBeforeMove.at(light.id) + 1,
                    "affected version increments by one");
        } else {
            require(versionsAfterMove.at(light.id) ==
                        versionsBeforeMove.at(light.id),
                    "unaffected version remains unchanged");
        }
    }
    writePpm(artifacts / "moved_incremental.ppm", movedRun.pixels, 320, 180);
    renderer.render(scene, movedRun.snapshot, indoor::DebugView::ShadowVersions);
    writePpm(artifacts / "shadow_versions_after_move.ppm",
             renderer.readFinalPixels(), 320, 180);
    info("movedObjects=" + std::to_string(moved.size()) +
         " dirtyLights=" + std::to_string(dirty.size()) +
         " rerenderedMaps=" +
         std::to_string(renderer.stats().shadowMapsRenderedThisFrame));

    const auto versionsBeforeResize = versionsAfterMove;
    const std::uint64_t migrationsBeforeResize =
        renderer.versionLedger().migrations();
    renderer.resize(scene, 640, 360);
    SceneRun resized = renderFrame(renderer, scene, camera,
                                   indoor::MaterialPresetId::FullLayered, {}, 3,
                                   true);
    require(renderer.settings().width == 640 && renderer.settings().height == 360,
            "G-Buffer size follows framebuffer resize");
    require(renderer.settings().shadowSize != 32,
            "shadow-map size must be rebuilt from the resized framebuffer");
    require(renderer.stats().validShadowMaps == 48,
            "resize must rebuild every resident shadow map");
    require(renderer.versionLedger().migrations() == migrationsBeforeResize,
            "resize must not reset or advance light versions");
    for (const auto& [lightId, version] : versionsBeforeResize) {
        require(renderer.versionLedger().version(lightId) == version,
                "light versions remain continuous across resize");
    }
    writePpm(artifacts / "resize_640x360.ppm", resized.pixels, 640, 360);

    renderer.resize(scene, 320, 180);
    SceneRun restored = renderFrame(renderer, scene, camera,
                                    indoor::MaterialPresetId::FullLayered, {}, 4,
                                    true);
    require(indoor::countPixelDifferences(movedRun.pixels, restored.pixels) == 0,
            "resize down/up must restore pixel-identical rendering");
    writePpm(artifacts / "resize_restored.ppm", restored.pixels, 320, 180);
    info("resize round trip restored at " +
         std::to_string(renderer.settings().width) + "x" +
         std::to_string(renderer.settings().height) + " diff=" +
         std::to_string(indoor::countPixelDifferences(movedRun.pixels,
                                                      restored.pixels)));

    const indoor::RenderSettings recoverySettings = renderer.settings();
    SceneRun preLoss = renderFrame(renderer, scene, camera,
                                   indoor::MaterialPresetId::FullLayered, {}, 5,
                                   true);
    const auto versionsBeforeLoss = versionsFrom(preLoss.snapshot);
    const std::uint64_t migrationsBeforeLoss =
        renderer.versionLedger().migrations();
    const indoor::FrameSnapshot recoverySnapshot = preLoss.snapshot;
    writePpm(artifacts / "context_before.ppm", preLoss.pixels, 320, 180);
    info("simulated context loss");
    renderer.invalidateContextWithoutDelete();
    ctxRecreate();
    renderer.recoverFromSnapshot(scene, recoverySnapshot, recoverySettings,
                                 INDOOR_SHADER_DIR);
    renderer.render(scene, recoverySnapshot, indoor::DebugView::Final);
    const std::vector<std::uint8_t> afterLoss = renderer.readFinalPixels();
    const int recoveryDifference =
        indoor::countPixelDifferences(preLoss.pixels, afterLoss);
    require(recoveryDifference == 0,
            "first frame after context recovery must be pixel-identical");
    require(renderer.stats().validShadowMaps == 48,
            "all shadow maps are reconstructed from the persistent snapshot");
    require(renderer.versionLedger().migrations() == migrationsBeforeLoss,
            "context recovery preserves the migration counter");
    for (const auto& [lightId, version] : versionsBeforeLoss) {
        require(renderer.versionLedger().version(lightId) == version,
                "light versions continue across context recovery");
    }
    writePpm(artifacts / "context_after.ppm", afterLoss, 320, 180);
    info("context recovery pixel diff=" + std::to_string(recoveryDifference) +
         " shadowMaps=48");

    const std::vector<indoor::MovedObject> movedAfterRecovery =
        indoor::advanceAnimation(scene, 1.0 / 30.0);
    const std::vector<std::uint32_t> dirtyAfterRecovery =
        indoor::findAffectedLights(scene.lights, movedAfterRecovery);
    SceneRun continued = renderFrame(
        renderer, scene, camera, indoor::MaterialPresetId::FullLayered,
        dirtyAfterRecovery, 6, false);
    const auto versionsAfterRecovery = versionsFrom(continued.snapshot);
    require(!dirtyAfterRecovery.empty(), "animation continues after recovery");
    for (std::uint32_t lightId : dirtyAfterRecovery) {
        require(versionsAfterRecovery.at(lightId) >
                    versionsBeforeLoss.at(lightId),
                "versions continue increasing after context recovery");
    }
    require(renderer.versionLedger().migrations() >
                migrationsBeforeLoss,
            "migration counter continues after context recovery");
    info("post-recovery version continuity verified");

    const std::uint32_t deletedLightId = scene.lights.back().id;
    const unsigned int deletedTexture =
        renderer.shadowTextureForLight(deletedLightId);
    require(deletedTexture != 0 && glIsTexture(deletedTexture) == GL_TRUE,
            "light to delete must own a live GL texture");
    renderer.releaseLightShadow(deletedLightId);
    indoor::removeLight(scene, deletedLightId);
    require(deletedTexture != 0 && glIsTexture(deletedTexture) == GL_FALSE,
            "deleted light GL texture must be physically released");
    require(renderer.shadowTextureForLight(deletedLightId) == 0,
            "deleted light must not remain in the GPU cache");
    require(renderer.shadowCacheResidentCount() == 47,
            "cache occupancy must drop by one after deletion");
    SceneRun afterDeletion = renderFrame(
        renderer, scene, camera, indoor::MaterialPresetId::FullLayered, {}, 7,
        false);
    require(renderer.stats().totalLights == 47,
            "debug total-light count follows deletion");
    require(renderer.stats().residentShadowMaps == 47 &&
                renderer.stats().validShadowMaps == 47,
            "remaining shadow maps stay valid after one light is deleted");
    renderer.render(scene, afterDeletion.snapshot,
                    indoor::DebugView::CacheOccupancy);
    writePpm(artifacts / "cache_occupancy_after_delete.ppm",
             renderer.readFinalPixels(), 320, 180);
    info("deleted light id=" + std::to_string(deletedLightId) +
         " cache=" + std::to_string(renderer.stats().residentShadowMaps) +
         "/" + std::to_string(renderer.stats().cacheCapacity));
    (void)afterDeletion;

    renderer.shutdown();
}

void testLruEviction() {
    info("testing live GPU LRU eviction with capacity < light count");
    indoor::RenderSettings settings;
    settings.width = 320;
    settings.height = 180;
    settings.shadowSize = 32;
    settings.shadowCacheCapacity = 12;

    indoor::Scene scene = indoor::buildDeterministicScene(1337, 32);
    indoor::Camera camera;
    indoor::DeferredRenderer renderer;
    renderer.initialize(scene, settings, INDOOR_SHADER_DIR);

    SceneRun first = renderFrame(renderer, scene, camera,
                                 indoor::MaterialPresetId::FullLayered, {}, 1,
                                 true);
    require(renderer.stats().residentShadowMaps == 12,
            "cache occupancy is limited by capacity");
    require(renderer.stats().validShadowMaps == 12,
            "all resident maps are valid under constrained capacity");
    require(renderer.stats().cacheEvictionsThisFrame == 0,
            "initial nearest-light selection fits capacity without churn");

    camera.yaw += 3.14159265f;
    camera.pitch = -0.05f;
    camera.distance = 30.0f;
    SceneRun second = renderFrame(renderer, scene, camera,
                                  indoor::MaterialPresetId::FullLayered, {}, 2,
                                  true);
    require(renderer.stats().residentShadowMaps == 12,
            "cache remains at capacity after camera-driven LRU churn");
    require(renderer.stats().cacheEvictionsThisFrame > 0,
            "new nearest-light set must evict stale LRU shadow textures");
    require(renderer.stats().validShadowMaps == 12,
            "all remaining resident maps are valid");
    const std::filesystem::path artifacts =
        std::filesystem::path(std::getenv("INDOOR_ARTIFACTS_DIR") != nullptr
                                  ? std::getenv("INDOOR_ARTIFACTS_DIR")
                                  : "artifacts");
    writePpm(artifacts / "lru_cache.ppm", second.pixels, 320, 180);
    info("LRU resident=" + std::to_string(renderer.stats().residentShadowMaps) +
         " evictionsThisFrame=" +
         std::to_string(renderer.stats().cacheEvictionsThisFrame) +
         " totalLights=" + std::to_string(scene.lights.size()));
    (void)first;
    renderer.shutdown();
}

std::filesystem::path g_artifacts = "artifacts";

} // namespace

namespace {
EglContext* g_context = nullptr;

void ctxRecreate() {
    if (g_context == nullptr) {
        throw std::runtime_error("EGL context not initialized");
    }
    g_context->recreateAfterLoss();
}
} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--artifacts" && i + 1 < argc) {
            g_artifacts = argv[++i];
        }
    }
    std::filesystem::create_directories(g_artifacts);
    setenv("INDOOR_ARTIFACTS_DIR", g_artifacts.c_str(), 1);

    try {
        EglContext context;
        g_context = &context;
        context.create();
        info("OpenGL vendor=" + std::string(
                 reinterpret_cast<const char*>(glGetString(GL_VENDOR))));
        info("OpenGL renderer=" + std::string(
                 reinterpret_cast<const char*>(glGetString(GL_RENDERER))));
        info("OpenGL version=" + std::string(
                 reinterpret_cast<const char*>(glGetString(GL_VERSION))));
        testRenderer(g_artifacts);
        testLruEviction();
        g_context = nullptr;
    } catch (const std::exception& error) {
        std::cerr << "FATAL GPU test exception: " << error.what() << '\n';
        return 2;
    }

    if (failures != 0) {
        std::cerr << failures << " GPU assertion(s) failed\n";
        return 1;
    }
    std::cout << "All headless GPU tests passed.\n";
    return 0;
}
