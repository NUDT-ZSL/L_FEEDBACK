#include "config.h"
#include "heightfield.h"
#include "image.h"
#include "renderer.h"
#include "terrain.h"
#include "window.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr int kExitConfig = 2;
constexpr int kExitMesh = 3;
constexpr int kExitRender = 4;
constexpr int kExitOutput = 5;

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string lowerExtension(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

std::string buildReport(const ct::AppConfig& config, const ct::HeightField& heightfield,
                        const ct::TerrainMesh& mesh, const ct::RenderStats& stats) {
    const ct::TerrainBuildStats terrain_stats = ct::summarizeTerrain(mesh);
    const ct::TerrainTopologyStats topology = ct::analyzeTopology(mesh);
    const std::uint64_t culled =
        stats.triangles_frustum_culled + stats.triangles_backface_culled;
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"report_version\": 1,\n";
    out << "  \"total_triangles\": " << stats.triangles_total << ",\n";
    out << "  \"clipped_triangles\": " << stats.triangles_after_frustum_clip
        << ",\n";
    out << "  \"culled_triangles\": " << culled << ",\n";
    out << "  \"lod_distribution\": {\"0\": " << terrain_stats.blocks_lod0
        << ", \"1\": " << terrain_stats.blocks_lod1 << ", \"2\": "
        << terrain_stats.blocks_lod2 << "},\n";
    out << "  \"average_anisotropic_samples\": "
        << stats.average_anisotropic_samples << ",\n";
    out << "  \"gbuffer_memory_bytes\": " << stats.gbuffer_bytes << ",\n";
    out << "  \"heightfield\": {\n";
    out << "    \"size\": " << config.heightfield.size << ",\n";
    out << "    \"frequency\": " << config.heightfield.base_frequency << ",\n";
    out << "    \"amplitude\": " << config.heightfield.amplitude << ",\n";
    out << "    \"seed\": " << config.heightfield.seed << ",\n";
    out << "    \"octaves\": " << config.heightfield.octaves << ",\n";
    out << "    \"minimum\": " << heightfield.minimum() << ",\n";
    out << "    \"maximum\": " << heightfield.maximum() << ",\n";
    out << "    \"mean\": " << heightfield.mean() << "\n";
    out << "  },\n";
    out << "  \"camera\": {\n";
    out << "    \"position\": [" << config.camera.position.x << ", "
        << config.camera.position.y << ", " << config.camera.position.z << "],\n";
    out << "    \"target\": [" << config.camera.target.x << ", " << config.camera.target.y
        << ", " << config.camera.target.z << "],\n";
    out << "    \"fov_degrees\": " << config.camera.fov_degrees << ",\n";
    out << "    \"near\": " << config.camera.near_plane << ",\n";
    out << "    \"far\": " << config.camera.far_plane << "\n";
    out << "  },\n";
    out << "  \"triangles\": {\n";
    out << "    \"unique_vertices\": " << mesh.vertices.size() << ",\n";
    out << "    \"total\": " << stats.triangles_total << ",\n";
    out << "    \"terrain\": " << mesh.terrain_triangles << ",\n";
    out << "    \"transition\": " << mesh.transition_triangles << ",\n";
    out << "    \"frustum_culled\": " << stats.triangles_frustum_culled << ",\n";
    out << "    \"partially_clipped\": " << stats.triangles_partially_clipped << ",\n";
    out << "    \"after_frustum_clip\": " << stats.triangles_after_frustum_clip
        << ",\n";
    out << "    \"backface_culled\": " << stats.triangles_backface_culled << ",\n";
    out << "    \"culled\": " << culled << ",\n";
    out << "    \"clipped_triangle_count\": " << stats.triangles_after_frustum_clip
        << ",\n";
    out << "    \"after_clipping\": " << stats.triangles_after_clipping << ",\n";
    out << "    \"rasterized\": " << stats.triangles_rasterized << "\n";
    out << "  },\n";
    out << "  \"lod\": {\n";
    out << "    \"block_size\": " << config.lod.block_size << ",\n";
    out << "    \"block_columns\": " << terrain_stats.block_columns << ",\n";
    out << "    \"block_rows\": " << terrain_stats.block_rows << ",\n";
    out << "    \"distribution\": {\"lod0_blocks\": " << terrain_stats.blocks_lod0
        << ", \"lod1_blocks\": " << terrain_stats.blocks_lod1
        << ", \"lod2_blocks\": " << terrain_stats.blocks_lod2 << "},\n";
    out << "    \"triangles_by_lod\": {\"lod0\": " << terrain_stats.triangles_lod0
        << ", \"lod1\": " << terrain_stats.triangles_lod1
        << ", \"lod2\": " << terrain_stats.triangles_lod2 << "},\n";
    out << "    \"seam_method\": \"transition_triangles\",\n";
    out << "    \"blocks\": [\n";
    for (std::size_t i = 0; i < mesh.blocks.size(); ++i) {
        const ct::BlockInfo& block = mesh.blocks[i];
        out << "      {\"x\": " << block.block_x << ", \"z\": " << block.block_z
            << ", \"lod\": " << block.lod << ", \"distance\": " << block.distance
            << ", \"projected_area_fraction\": " << block.projected_area_fraction
            << ", \"triangles\": " << block.triangles << "}";
        if (i + 1 < mesh.blocks.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"topology\": {\n";
    out << "    \"edges\": " << topology.edges << ",\n";
    out << "    \"boundary_edges\": " << topology.boundary_edges << ",\n";
    out << "    \"manifold_edges\": " << topology.manifold_edges << ",\n";
    out << "    \"non_manifold_edges\": " << topology.non_manifold_edges << ",\n";
    out << "    \"hanging_edges\": " << topology.hanging_edges << "\n";
    out << "  },\n";
    out << "  \"rasterization\": {\n";
    out << "    \"width\": " << stats.width << ",\n";
    out << "    \"height\": " << stats.height << ",\n";
    out << "    \"supersampling\": " << stats.supersampling << ",\n";
    out << "    \"internal_width\": " << stats.internal_width << ",\n";
    out << "    \"internal_height\": " << stats.internal_height << ",\n";
    out << "    \"covered_samples\": " << stats.covered_samples << "\n";
    out << "  },\n";
    out << "  \"g_buffer\": {\n";
    out << "    \"format\": \"float_position_normal_uv_aniso_depth_material\",\n";
    out << "    \"bytes_per_pixel\": " << stats.gbuffer_bytes_per_pixel << ",\n";
    out << "    \"pixel_count\": " << stats.gbuffer_pixel_count << ",\n";
    out << "    \"total_bytes\": " << stats.gbuffer_bytes << ",\n";
    out << "    \"total_mib\": "
        << static_cast<double>(stats.gbuffer_bytes) / (1024.0 * 1024.0) << "\n";
    out << "  },\n";
    out << "  \"anisotropic_sampling\": {\n";
    out << "    \"average_samples\": " << stats.average_anisotropic_samples << ",\n";
    out << "    \"max_samples\": " << stats.max_anisotropic_samples << ",\n";
    out << "    \"sample_budget\": 16,\n";
    out << "    \"total_samples\": " << stats.anisotropic_samples << "\n";
    out << "  },\n";
    out << "  \"lighting\": {\n";
    out << "    \"directional_lights\": " << config.directional_lights.size() << ",\n";
    out << "    \"point_lights\": " << config.point_lights.size() << ",\n";
    out << "    \"pass\": \"deferred_screen_space\"\n";
    out << "  },\n";
    out << "  \"output\": {\n";
    std::string image_format = lowerExtension(config.output.image_path);
    if (!image_format.empty() && image_format.front() == '.') {
        image_format.erase(image_format.begin());
    }
    out << "    \"image_format\": \"" << jsonEscape(image_format) << "\",\n";
    out << "    \"report_format\": \"json\"\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

void printUsage() {
    std::cout
        << "Usage: terrain_renderer --config <file.json> [--validate-only]\n"
        << "       terrain_renderer --help\n"
        << "\n"
        << "Renders the configured deterministic terrain on the CPU and writes\n"
        << "a BMP/PPM image plus a JSON statistics report.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    bool validate_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            printUsage();
            return 0;
        }
        if (argument == "--config") {
            if (i + 1 >= argc) {
                std::cerr << "error: --config requires a file path\n";
                return kExitConfig;
            }
            config_path = argv[++i];
        } else if (argument == "--validate-only") {
            validate_only = true;
        } else {
            std::cerr << "error: unknown argument '" << argument << "'\n";
            printUsage();
            return kExitConfig;
        }
    }
    if (config_path.empty()) {
        std::cerr << "error: a configuration file is required (--config <file.json>)\n";
        printUsage();
        return kExitConfig;
    }

    try {
        ct::WindowMessageLoop messages;
        if (!messages.pump()) {
            std::cerr << "error: window message loop terminated before rendering\n";
            return kExitRender;
        }

        ct::AppConfig config;
        std::string error;
        if (!ct::loadConfig(config_path, config, error)) {
            std::cerr << "error: " << error << '\n';
            return kExitConfig;
        }
        if (validate_only) {
            std::cout << "configuration is valid\n";
            return 0;
        }

        const ct::HeightField heightfield(config.heightfield);
        const ct::TerrainMesh mesh = ct::buildTerrain(heightfield, config);
        if (!ct::validateTerrainMesh(mesh, error)) {
            std::cerr << "error: mesh validation failed: " << error << '\n';
            return kExitMesh;
        }

        ct::ColorBuffer image;
        ct::RenderStats stats;
        if (!ct::renderTerrain(heightfield, mesh, config, image, stats, error)) {
            std::cerr << "error: render failed: " << error << '\n';
            return kExitRender;
        }
        if (!messages.pump()) {
            std::cerr << "error: window message loop requested shutdown\n";
            return kExitRender;
        }

        bool image_ok = false;
        const std::string extension = lowerExtension(config.output.image_path);
        if (extension == ".bmp") {
            image_ok = ct::writeBmp(config.output.image_path, image, error);
        } else if (extension == ".ppm") {
            image_ok = ct::writePpm(config.output.image_path, image, error);
        } else {
            error = "unsupported image extension '" + extension + "'";
        }
        if (!image_ok) {
            std::cerr << "error: image output failed: " << error << '\n';
            return kExitOutput;
        }

        const std::string report = buildReport(config, heightfield, mesh, stats);
        if (!ct::writeJsonText(config.output.report_path, report, error)) {
            std::cerr << "error: report output failed: " << error << '\n';
            return kExitOutput;
        }

        const ct::TerrainBuildStats terrain_stats = ct::summarizeTerrain(mesh);
        std::cout << "rendered " << stats.width << 'x' << stats.height
                  << " at " << stats.supersampling << "x SSAA"
                  << " | triangles=" << stats.triangles_total
                  << " | lod blocks=" << terrain_stats.blocks_lod0 << '/'
                  << terrain_stats.blocks_lod1 << '/' << terrain_stats.blocks_lod2
                  << " | avg aniso taps=" << std::fixed << std::setprecision(3)
                  << stats.average_anisotropic_samples << '\n';
        std::cout << "image: " << config.output.image_path << '\n';
        std::cout << "report: " << config.output.report_path << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: unexpected failure: " << exception.what() << '\n';
        return kExitRender;
    }
}
