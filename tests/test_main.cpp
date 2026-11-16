#include "image.hpp"
#include "json.hpp"
#include "pipeline.hpp"
#include "render.hpp"
#include "scene.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const std::string& message) {
    ++gChecks;
    if (!condition) {
        ++gFailures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void checkNear(double actual, double expected, double tolerance,
               const std::string& message) {
    ++gChecks;
    if (!(std::abs(actual - expected) <= tolerance)) {
        ++gFailures;
        std::cerr << "FAIL: " << message << " (actual=" << actual
                  << ", expected=" << expected << ", tol=" << tolerance << ")\n";
    }
}

class TempDirectory {
public:
    explicit TempDirectory(const std::string& name) {
        static int serial = 0;
        path_ = fs::temp_directory_path() /
                ("tilegen-test-" + name + "-" + std::to_string(++serial));
        std::error_code error;
        fs::remove_all(path_, error);
        fs::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void writeText(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write test file " + path.string());
    }
    output << text;
}

std::string readText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

tilegen::PremulPixel fillPixel(double r, double g, double b, double a) {
    return {static_cast<float>(r), static_cast<float>(g), static_cast<float>(b),
            static_cast<float>(a)};
}

tilegen::Scene makeFillScene(tilegen::FillRule rule) {
    tilegen::Scene scene;
    scene.width = 16;
    scene.height = 16;
    tilegen::Layer layer;
    layer.id = "fill";
    tilegen::Primitive primitive;
    primitive.kind = tilegen::PrimitiveKind::Polygon;
    primitive.fillRule = rule;
    primitive.rings = {
        {{1, 1}, {15, 1}, {15, 15}, {1, 15}},
        {{4, 4}, {12, 4}, {12, 12}, {4, 12}}};
    tilegen::FillStyle fill;
    fill.color = {1.0, 1.0, 1.0, 1.0};
    primitive.fill = fill;
    layer.primitives.push_back(primitive);
    scene.layers.push_back(layer);
    return scene;
}

void testJsonAndSceneValidation() {
    tilegen::JsonParser parser;
    const tilegen::JsonValue value = parser.parse(R"({"a":[1,true,null]})");
    check(value.isObject(), "JSON object parses");
    check(value.find("a") != nullptr && value.find("a")->asArray().size() == 3,
          "JSON array parses");

    bool rejectedNan = false;
    try {
        parser.parse(R"({"a": NaN})");
    } catch (const tilegen::JsonError&) {
        rejectedNan = true;
    }
    check(rejectedNan, "NaN token is rejected");

    TempDirectory temp("scene-validation");
    const fs::path scenePath = temp.path() / "bad.json";
    writeText(scenePath, R"({
      "width": 16, "height": 16,
      "layers": [{"id": "bad", "primitives": [
        {"type": "circle", "center": [8, 8], "radius": -2, "fill": "#fff"}
      ]}]
    })");
    bool pathReported = false;
    try {
        (void)tilegen::loadScene(scenePath.string());
    } catch (const tilegen::SceneError& error) {
        pathReported =
            error.path() == "$.layers[0].primitives[0].radius" &&
            std::string(error.what()).find("greater than zero") != std::string::npos;
    }
    check(pathReported, "negative radius reports exact JSON path");

    const fs::path overflowPath = temp.path() / "overflow.json";
    writeText(overflowPath, R"({
      "width": 16, "height": 16,
      "layers": [{"id": "bad", "primitives": [
        {"type": "circle", "center": [8, 8], "radius": 1e999, "fill": "#fff"}
      ]}]
    })");
    bool overflowPathReported = false;
    try {
        (void)tilegen::loadScene(overflowPath.string());
    } catch (const tilegen::SceneError& error) {
        overflowPathReported =
            error.path() == "$.layers[0].primitives[0].radius" &&
            std::string(error.what()).find("finite") != std::string::npos;
    }
    check(overflowPathReported,
          "infinite numeric overflow reports exact JSON path");

    const fs::path missingFieldPath = temp.path() / "missing.json";
    writeText(missingFieldPath, R"({"width":16,"height":16,"layers":[
      {"id":"x","primitives":[{"type":"polygon","fill":"#fff"}]}
    ]})");
    bool missingPathReported = false;
    try {
        (void)tilegen::loadScene(missingFieldPath.string());
    } catch (const tilegen::SceneError& error) {
        missingPathReported =
            error.path() == "$.layers[0].primitives[0].points";
    }
    check(missingPathReported, "missing geometry reports exact JSON path");
}

void testPngAndBilinear() {
    const std::vector<tilegen::Rgba8> pixels = {
        {0, 0, 0, 255}, {255, 0, 0, 255},
        {0, 255, 0, 255}, {255, 255, 255, 255}};
    const std::vector<std::uint8_t> png =
        tilegen::encodePngRgba(2, 2, pixels);
    int width = 0;
    int height = 0;
    const std::vector<tilegen::Rgba8> decoded =
        tilegen::decodePngRgba(png, width, height);
    check(width == 2 && height == 2, "PNG round-trip dimensions");
    check(decoded.size() == 4 && decoded[3].r == 255 && decoded[3].a == 255,
          "PNG round-trip pixels");

    tilegen::Texture texture;
    texture.width = 2;
    texture.height = 2;
    texture.pixels = pixels;
    const tilegen::Rgba8 middle = texture.sampleBilinear(0.5, 0.5, false);
    check(std::abs(static_cast<int>(middle.r) - 128) <= 1,
          "bilinear red midpoint");
    check(std::abs(static_cast<int>(middle.g) - 128) <= 1,
          "bilinear green midpoint");
    check(std::abs(static_cast<int>(middle.b) - 64) <= 1,
          "bilinear blue midpoint");
    check(middle.a == 255, "bilinear alpha midpoint");

    const fs::path fixture =
        fs::path(TILEGEN_SOURCE_DIR) / "tests" / "fixtures" / "compressed.png";
    int fixtureWidth = 0;
    int fixtureHeight = 0;
    const std::vector<tilegen::Rgba8> fixturePixels =
        tilegen::decodePngRgbaFile(fixture.string(), fixtureWidth, fixtureHeight);
    check(fixtureWidth == 64 && fixtureHeight == 64,
          "dynamic-Huffman PNG dimensions");
    check(fixturePixels.size() == 64U * 64U && fixturePixels[0].a == 255,
          "dynamic-Huffman PNG decodes");

    std::vector<std::uint8_t> corrupt = png;
    corrupt[corrupt.size() / 2] ^= 0x40;
    bool corruptRejected = false;
    try {
        (void)tilegen::decodePngRgba(corrupt, width, height);
    } catch (const std::exception&) {
        corruptRejected = true;
    }
    check(corruptRejected, "corrupt PNG is rejected");
}

void testPremultipliedBlendReference() {
    const tilegen::PremulPixel red = fillPixel(0.5, 0.0, 0.0, 0.5);
    const tilegen::PremulPixel blue = fillPixel(0.0, 0.0, 0.5, 0.5);
    const tilegen::PremulPixel redThenBlue =
        tilegen::blendPremultiplied(red, blue, tilegen::BlendMode::Normal);
    const tilegen::PremulPixel blueThenRed =
        tilegen::blendPremultiplied(blue, red, tilegen::BlendMode::Normal);
    checkNear(redThenBlue.r, 0.25, 1.0 / 255.0, "normal blend red");
    checkNear(redThenBlue.b, 0.5, 1.0 / 255.0, "normal blend blue");
    checkNear(redThenBlue.a, 0.75, 1.0 / 255.0, "normal blend alpha");
    checkNear(blueThenRed.r, 0.5, 1.0 / 255.0, "blend is order sensitive A");
    checkNear(blueThenRed.b, 0.25, 1.0 / 255.0, "blend is order sensitive B");

    const tilegen::PremulPixel backdrop = fillPixel(0.5, 0.25, 0.75, 1.0);
    const tilegen::PremulPixel source = fillPixel(0.5, 0.5, 0.5, 1.0);
    const tilegen::PremulPixel multiplied =
        tilegen::blendPremultiplied(backdrop, source, tilegen::BlendMode::Multiply);
    checkNear(multiplied.r, 0.25, 1.0 / 255.0, "multiply red");
    checkNear(multiplied.g, 0.125, 1.0 / 255.0, "multiply green");
    checkNear(multiplied.b, 0.375, 1.0 / 255.0, "multiply blue");

    const tilegen::PremulPixel bright = fillPixel(0.8, 0.2, 0.3, 1.0);
    const tilegen::PremulPixel addedSource = fillPixel(0.4, 0.9, 0.2, 1.0);
    const tilegen::PremulPixel added =
        tilegen::blendPremultiplied(bright, addedSource, tilegen::BlendMode::Add);
    checkNear(added.r, 1.0, 1.0 / 255.0, "add overflow clamps red");
    checkNear(added.g, 1.0, 1.0 / 255.0, "add overflow clamps green");
    checkNear(added.b, 0.5, 1.0 / 255.0, "add blue");

    const tilegen::PremulPixel halfBackdrop = fillPixel(0.25, 0.25, 0.25, 0.5);
    const tilegen::PremulPixel halfSource =
        fillPixel(0.25, 0.125, 0.375, 0.5);
    const tilegen::PremulPixel halfMultiply = tilegen::blendPremultiplied(
        halfBackdrop, halfSource, tilegen::BlendMode::Multiply);
    const tilegen::PremulPixel halfAdd = tilegen::blendPremultiplied(
        halfBackdrop, halfSource, tilegen::BlendMode::Add);
    checkNear(halfMultiply.r, 0.3125, 1.0 / 255.0,
              "semi-transparent multiply reference red");
    checkNear(halfMultiply.g, 0.21875, 1.0 / 255.0,
              "semi-transparent multiply reference green");
    checkNear(halfMultiply.a, 0.75, 1.0 / 255.0,
              "semi-transparent multiply reference alpha");
    checkNear(halfAdd.r, 0.5, 1.0 / 255.0,
              "semi-transparent add reference red");
    checkNear(halfAdd.g, 0.375, 1.0 / 255.0,
              "semi-transparent add reference green");
    checkNear(halfAdd.a, 0.75, 1.0 / 255.0,
              "semi-transparent add reference alpha");

    const tilegen::PremulPixel negativeSource = fillPixel(-1.0, 2.0, -0.5, 0.5);
    const tilegen::PremulPixel truncated =
        tilegen::blendPremultiplied(backdrop, negativeSource, tilegen::BlendMode::Add);
    check(truncated.r >= 0.0F && truncated.g <= 1.0F && truncated.b >= 0.0F,
          "negative and overflowing inputs are clamped");
}

void testFillRulesAndStrokeScaling() {
    const tilegen::RenderSettings settings{4};
    const tilegen::Bounds full16{0, 0, 16, 16, true};
    const tilegen::RegionRenderResult nonzero = tilegen::renderSceneRegion(
        makeFillScene(tilegen::FillRule::NonZero), 1.0, full16, 16, 16, settings,
        false);
    const tilegen::RegionRenderResult evenOdd = tilegen::renderSceneRegion(
        makeFillScene(tilegen::FillRule::EvenOdd), 1.0, full16, 16, 16, settings,
        false);
    check(nonzero.image.at(8, 8).a > 0.99F,
          "nonzero winding fills clockwise inner ring");
    check(evenOdd.image.at(8, 8).a < 0.01F,
          "even-odd rule punches a hole for inner ring");
    check(evenOdd.image.at(0, 0).a < 0.01F,
          "even-odd remains transparent outside the outer ring");

    tilegen::Scene strokeScene;
    strokeScene.width = 64;
    strokeScene.height = 64;
    tilegen::Layer strokeLayer;
    strokeLayer.id = "stroke";
    tilegen::Primitive line;
    line.kind = tilegen::PrimitiveKind::Polyline;
    line.points = {{8, 32}, {56, 32}};
    tilegen::StrokeStyle stroke;
    stroke.width = 8.0;
    stroke.color = {1.0, 1.0, 1.0, 1.0};
    stroke.cap = tilegen::CapStyle::Butt;
    line.stroke = stroke;
    strokeLayer.primitives.push_back(line);
    strokeScene.layers.push_back(strokeLayer);

    const tilegen::RegionRenderResult level0 = tilegen::renderSceneRegion(
        strokeScene, 1.0, tilegen::Bounds{0, 0, 64, 64, true}, 64, 64, settings,
        false);
    const tilegen::RegionRenderResult level1 = tilegen::renderSceneRegion(
        strokeScene, 0.5, tilegen::Bounds{0, 0, 64, 64, true}, 32, 32, settings,
        false);
    auto opaqueCount = [](const tilegen::Image& image) {
        int count = 0;
        for (const auto& pixel : image.pixels) {
            if (pixel.a > 0.5F) {
                ++count;
            }
        }
        return count;
    };
    const int count0 = opaqueCount(level0.image);
    const int count1 = opaqueCount(level1.image);
    check(count0 > 0 && count1 > 0, "stroke renders at both levels");
    const double ratio = static_cast<double>(count1) / count0;
    check(ratio > 0.18 && ratio < 0.34,
          "stroke width scales proportionally with level resolution");
}

void testPipelineAndResume() {
    TempDirectory temp("pipeline");
    const fs::path texturePath = temp.path() / "texture.png";
    std::vector<tilegen::Rgba8> texturePixels(8U * 8U);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            texturePixels[static_cast<std::size_t>(y) * 8U +
                          static_cast<std::size_t>(x)] = {
                static_cast<std::uint8_t>(x * 31),
                static_cast<std::uint8_t>(y * 31),
                static_cast<std::uint8_t>((x + y) * 15), 255};
        }
    }
    tilegen::writeBinaryFile(texturePath.string(),
                             tilegen::encodePngRgba(8, 8, texturePixels));

    const fs::path scenePath = temp.path() / "scene.json";
    writeText(scenePath, R"({
      "width": 64, "height": 48, "background": "#102030",
      "layers": [
        {"id": "textured", "opacity": 0.8, "blend": "normal",
         "transform": {"origin": [24, 24], "rotate": 12, "scale": 1.0,
                       "translate": [4, 2]},
         "primitives": [
           {"type": "polygon",
            "points": [[4, 4], [42, 7], [38, 35], [8, 40]],
            "fill": {"color": [220, 240, 255, 230],
                     "texture": "texture.png", "repeat": false}}
         ]},
        {"id": "road", "opacity": 0.9, "blend": "add",
         "primitives": [
           {"type": "polyline", "points": [[0, 42], [20, 28], [40, 37]],
            "stroke": {"width": 4, "color": [240, 180, 60, 210],
                       "cap": "round", "join": "round"}}
         ]},
        {"id": "marker", "opacity": 0.7, "blend": "multiply",
         "primitives": [
           {"type": "circle", "center": [48, 12], "radius": 7,
            "fill": {"color": [230, 70, 90, 255]},
            "stroke": {"width": 2, "color": [20, 20, 20, 255],
                       "cap": "round", "join": "round"}}
         ]}
      ]
    })");

    tilegen::PipelineOptions options;
    options.scenePath = scenePath.string();
    options.outputDirectory = (temp.path() / "out").string();
    options.tileSize = 16;
    options.levels = 3;
    options.previewLevel = 1;
    options.previewWidth = 64;
    options.previewHeight = 48;
    options.samplesPerAxis = 2;
    check(tilegen::runPipeline(options) == 0, "pipeline completes");

    const fs::path manifestPath = temp.path() / "out" / "manifest.json";
    tilegen::JsonParser parser;
    const tilegen::JsonValue manifest = parser.parseFile(manifestPath.string());
    check(manifest.find("completed_levels") != nullptr &&
              manifest.find("completed_levels")->asArray().size() == 3,
          "manifest marks all levels complete");
    const tilegen::JsonValue& tiles = *manifest.find("tiles");
    check(tiles.asArray().size() == 17, "tile grid covers full scene");
    bool sawEmpty = false;
    int presentCount = 0;
    for (const auto& tile : tiles.asArray()) {
        if (tile.find("empty")->asBool()) {
            sawEmpty = true;
            check(!tile.find("present")->asBool(), "empty tile is marked absent");
            continue;
        }
        ++presentCount;
        const bool present = tile.find("present")->asBool();
        check(present, "non-empty tile is present");
        const fs::path rawPath =
            temp.path() / "out" / tile.find("path")->asString();
        const std::vector<std::uint8_t> raw =
            tilegen::readBinaryFile(rawPath.string());
        check(raw.size() == 16U * 16U * 4U, "tile has exact RGBA byte length");
        const std::uint32_t expected =
            tilegen::crc32(raw);
        check(tilegen::crc32Hex(expected) == tile.find("crc32")->asString(),
              "manifest CRC matches raw tile bytes");
        const fs::path pngPath =
            temp.path() / "out" / tile.find("png_path")->asString();
        const std::vector<std::uint8_t> png =
            tilegen::readBinaryFile(pngPath.string());
        check(tilegen::crc32Hex(tilegen::crc32(png)) ==
                  tile.find("png_crc32")->asString(),
              "manifest PNG CRC matches PNG bytes");
    }
    check(presentCount > 0, "at least one tile contains pixels");
    check(sawEmpty, "blank tiles are listed and marked empty");

    const fs::path previewPath =
        temp.path() / "out" / manifest.find("preview")->find("path")->asString();
    int previewWidth = 0;
    int previewHeight = 0;
    (void)tilegen::decodePngRgbaFile(previewPath.string(), previewWidth,
                                     previewHeight);
    check(previewWidth == 64 && previewHeight == 48,
          "preview PNG has requested dimensions");
    check(manifest.find("preview")->find("level")->asNumber() == 1,
          "preview records injected level");
    check(manifest.find("preview")
                  ->find("projection")
                  ->asString() == "equirectangular",
          "preview records equirectangular projection");

    tilegen::JsonValue firstTile;
    std::vector<std::uint8_t> firstTileBytes;
    for (const auto& tile : tiles.asArray()) {
        if (tile.find("present")->asBool()) {
            const fs::path rawPath =
                temp.path() / "out" / tile.find("path")->asString();
            firstTileBytes = tilegen::readBinaryFile(rawPath.string());
            firstTile = tile;
            break;
        }
    }
    check(!firstTileBytes.empty(), "captured deterministic tile");

    check(tilegen::runPipeline(options) == 0, "fresh rerun completes");
    const std::vector<std::uint8_t> repeated = tilegen::readBinaryFile(
        (temp.path() / "out" / firstTile.find("path")->asString()).string());
    check(repeated == firstTileBytes, "fresh rerun is byte-for-byte deterministic");

    options.simulateDiskFullLevel = 2;
    check(tilegen::runPipeline(options) == 3,
          "simulated disk full returns nonzero exit code");
    tilegen::JsonValue partial =
        parser.parseFile((temp.path() / "out" / "manifest.json").string());
    check(partial.find("completed_levels")->asArray().size() == 2,
          "failure preserves completed levels 0 and 1");
    check(tilegen::readBinaryFile(
              (temp.path() / "out" / firstTile.find("path")->asString()).string()) ==
              firstTileBytes,
          "failure preserves generated tile bytes");

    const fs::file_time_type before =
        fs::last_write_time(temp.path() / "out" / firstTile.find("path")->asString());
    options.simulateDiskFullLevel = -1;
    options.resume = true;
    check(tilegen::runPipeline(options) == 0, "resume completes");
    const fs::file_time_type after =
        fs::last_write_time(temp.path() / "out" / firstTile.find("path")->asString());
    check(before == after, "resume does not overwrite a manifest-valid tile");
    const tilegen::JsonValue resumed =
        parser.parseFile((temp.path() / "out" / "manifest.json").string());
    check(resumed.find("completed_levels")->asArray().size() == 3,
          "resume restores all completed levels");
}

void testMissingTextureError() {
    TempDirectory temp("missing-texture");
    const fs::path scenePath = temp.path() / "scene.json";
    writeText(scenePath, R"({
      "width": 8, "height": 8,
      "layers": [{"id": "x", "primitives": [
        {"type": "polygon", "points": [[0,0],[8,0],[8,8]],
         "fill": {"color": "#fff", "texture": "does-not-exist.png"}}
      ]}]
    })");
    tilegen::PipelineOptions options;
    options.scenePath = scenePath.string();
    options.outputDirectory = (temp.path() / "out").string();
    options.tileSize = 8;
    options.levels = 1;
    options.previewLevel = 0;
    options.previewWidth = 8;
    options.previewHeight = 8;
    options.samplesPerAxis = 1;
    check(tilegen::runPipeline(options) == 4,
          "missing texture returns texture error exit code");
    check(fs::is_regular_file(temp.path() / "out" / "tilegen.log"),
          "texture error is recorded in log");
    const std::string log =
        readText(temp.path() / "out" / "tilegen.log");
    check(log.find("does-not-exist.png") != std::string::npos,
          "texture error names the missing asset");
}

}  // namespace

int main() {
    try {
        testJsonAndSceneValidation();
        testPngAndBilinear();
        testPremultipliedBlendReference();
        testFillRulesAndStrokeScaling();
        testPipelineAndResume();
        testMissingTextureError();
    } catch (const std::exception& error) {
        std::cerr << "UNCAUGHT: " << error.what() << '\n';
        return 2;
    }
    std::cout << gChecks << " checks, " << gFailures << " failures\n";
    return gFailures == 0 ? 0 : 1;
}
