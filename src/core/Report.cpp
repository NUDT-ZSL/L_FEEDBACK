#include "core/Report.hpp"

#include <fstream>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <sstream>

#include "core/Error.hpp"
#include "core/Serializer.hpp"

namespace lcr {
namespace {

std::string escapeJson(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char c : value) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

void writeVec3(std::ostream& out, const Vec3 value) {
  out << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

void writeStringArray(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      out << ',';
    }
    out << values[i];
  }
  out << ']';
}

void writeSwitchEvents(std::ostream& out, const std::vector<SwitchEvent>& events) {
  out << '[';
  for (std::size_t i = 0; i < events.size(); ++i) {
    const SwitchEvent& event = events[i];
    if (i != 0U) {
      out << ',';
    }
    out << "{\"tick\":" << event.tick
        << ",\"render_time\":" << event.renderTime
        << ",\"from\":\"" << escapeJson(event.fromEquipment)
        << "\",\"to\":\"" << escapeJson(event.toEquipment)
        << "\",\"pose_hash_before\":\"" << hexU64(event.poseHashBefore)
        << "\",\"pose_hash_after\":\"" << hexU64(event.poseHashAfter)
        << "\",\"max_joint_position_delta\":" << event.maxJointPositionDelta
        << ",\"triggered\":" << (event.triggered ? "true" : "false")
        << '}';
  }
  out << ']';
}

void writeInterpolation(std::ostream& out, const InterpolationComparison& value) {
  out << "{\"render_fps\":" << value.renderFps
      << ",\"duration_seconds\":" << value.durationSeconds
      << ",\"samples\":" << value.samples
      << ",\"without_interpolation\":{\"max_joint_error\":" << value.withoutInterpolation.maxPositionError
      << ",\"rms_joint_error\":" << value.withoutInterpolation.rmsPositionError
      << "},\"with_interpolation\":{\"max_joint_error\":" << value.withInterpolation.maxPositionError
      << ",\"rms_joint_error\":" << value.withInterpolation.rmsPositionError
      << "},\"max_error_improvement_ratio\":" << value.maxErrorImprovement
      << ",\"rms_error_improvement_ratio\":" << value.rmsErrorImprovement
      << '}';
}

void writeCapeSort(std::ostream& out, const CapeSortReport& sort) {
  out << "{\"triangle_count\":" << sort.triangleCount
      << ",\"order_valid\":" << (sort.orderValid ? "true" : "false")
      << ",\"order_violations\":" << sort.orderViolations
      << ",\"triangles_in_front_of_opaque\":" << sort.trianglesInFrontOfOpaque
      << ",\"triangles_behind_opaque\":" << sort.trianglesBehindOpaque
      << ",\"torso_occluded_triangles\":" << sort.torsoOccludedTriangles
      << ",\"weapon_occluded_triangles\":" << sort.weaponOccludedTriangles
      << ",\"min_edge_feather\":" << sort.minEdgeFeather
      << ",\"max_edge_feather\":" << sort.maxEdgeFeather
      << ",\"order_hash\":\"" << hexU64(sort.orderHash) << "\"}";
}

void writePipeline(std::ostream& out, const PipelineReport& pipeline) {
  out << "{\"initialized\":" << (pipeline.initialized ? "true" : "false")
      << ",\"shader_program_builds\":" << pipeline.shaderProgramBuilds
      << ",\"vao_builds\":" << pipeline.vaoBuilds
      << ",\"texture_uploads\":" << pipeline.textureUploads
      << ",\"equipment_pipeline_rebuilds\":" << pipeline.equipmentPipelineRebuilds
      << ",\"opengl_version\":\"" << escapeJson(pipeline.openGlVersion)
      << "\",\"renderer\":\"" << escapeJson(pipeline.rendererName) << "\"}";
}

void openOutput(std::ofstream& out, const std::string& path) {
  const std::filesystem::path outputPath(path);
  if (outputPath.has_parent_path()) {
    std::error_code error;
    std::filesystem::create_directories(outputPath.parent_path(), error);
    if (error) {
      fail(
          ErrorCode::Io,
          "report.directory",
          "cannot create report directory: " + outputPath.parent_path().string());
    }
  }
  out.open(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    fail(ErrorCode::Io, "report.write", "cannot open output file: " + path);
  }
  out.imbue(std::locale::classic());
  out << std::setprecision(9);
}

void finishOutput(std::ofstream& out, const std::string& path) {
  out.flush();
  if (!out) {
    fail(ErrorCode::Io, "report.write", "failed while writing output file: " + path);
  }
}

void writeEquipmentSets(std::ostream& out, const std::vector<EquipmentSet>& sets) {
  out << '[';
  for (std::size_t i = 0; i < sets.size(); ++i) {
    const EquipmentSet& set = sets[i];
    if (i != 0U) {
      out << ',';
    }
    out << "{\"id\":\"" << escapeJson(set.id)
        << "\",\"name\":\"" << escapeJson(set.displayName)
        << "\",\"set_hash\":\"" << hexU64(set.hash)
        << "\",\"body_texture_hash\":\"" << hexU64(set.bodyTexture.hash)
        << "\",\"weapon_texture_hash\":\"" << hexU64(set.weaponTexture.hash)
        << "\",\"body_emissive\":";
    writeVec3(out, set.bodyMaterial.emissiveColor);
    out << ",\"body_emissive_strength\":" << set.bodyMaterial.emissiveStrength
        << ",\"weapon_emissive\":";
    writeVec3(out, set.weaponMaterial.emissiveColor);
    out << ",\"weapon_emissive_strength\":" << set.weaponMaterial.emissiveStrength
        << '}';
  }
  out << ']';
}

}  // namespace

void writeValidationReport(const std::string& path, const RunReport& report) {
  std::ofstream out;
  openOutput(out, path);
  out << "{\n";
  out << "  \"schema\":\"lcr-validation-report-v1\",\n";
  out << "  \"timing_mode\":\"" << escapeJson(report.timingMode) << "\",\n";
  out << "  \"seed\":" << report.config.seed << ",\n";
  out << "  \"config\":{\"texture_size\":" << report.config.textureSize
      << ",\"initial_equipment\":\"" << escapeJson(report.config.initialEquipment)
      << "\",\"switch_times\":";
  writeStringArray(out, report.config.scheduledSwitchTimes);
  out << "},\n";
  out << "  \"assets\":{\"skeleton_hash\":\"" << hexU64(report.skeletonHash)
      << "\",\"mesh_hash\":\"" << hexU64(report.meshHash)
      << "\",\"character_hash\":\"" << hexU64(report.characterHash)
      << "\",\"equipment_catalog_hash\":\"" << hexU64(report.equipmentCatalogHash)
      << "\"},\n";
  out << "  \"logic\":{\"fixed_hz\":60,\"steps\":" << report.logicSteps
      << ",\"seconds\":" << report.simulatedSeconds << "},\n";
  out << "  \"render\":{\"frames\":" << report.renderFrames
      << ",\"min_ms\":" << report.frameStats.minMs
      << ",\"average_ms\":" << report.frameStats.averageMs
      << ",\"p95_ms\":" << report.frameStats.p95Ms
      << ",\"max_ms\":" << report.frameStats.maxMs << "},\n";
  out << "  \"final\":{\"state_hash\":\"" << hexU64(report.finalStateHash)
      << "\",\"equipment\":\"" << escapeJson(report.finalEquipment)
      << "\"},\n";
  out << "  \"equipment_sets\":";
  writeEquipmentSets(out, report.equipmentSets);
  out << ",\n";
  out << "  \"interpolation\":[";
  for (std::size_t i = 0; i < report.interpolation.size(); ++i) {
    if (i != 0U) {
      out << ',';
    }
    writeInterpolation(out, report.interpolation[i]);
  }
  out << "],\n";
  out << "  \"cape_samples\":[";
  for (std::size_t i = 0; i < report.capeSamples.size(); ++i) {
    if (i != 0U) {
      out << ',';
    }
    out << "{\"render_time\":" << report.capeSamples[i].renderTime
        << ",\"equipment\":\"" << escapeJson(report.capeSamples[i].equipment)
        << "\",\"sort\":";
    writeCapeSort(out, report.capeSamples[i].sort);
    out << '}';
  }
  out << "],\n";
  out << "  \"equipment_switch\":";
  writeSwitchEvents(out, report.switchEvents);
  out << ",\n  \"pipeline\":";
  writePipeline(out, report.pipeline);
  out << "\n}\n";
  finishOutput(out, path);
}

void writeStateSnapshot(
    const std::string& path,
    const Simulation& simulation,
    const double renderTime,
    const PipelineReport& pipeline) {
  std::ofstream out;
  openOutput(out, path);
  const Pose pose = simulation.interpolatedPose(renderTime);
  const std::vector<Vec3> joints = simulation.assets().skeleton.jointWorldPositions(pose);
  out << "{\n";
  out << "  \"schema\":\"lcr-state-snapshot-v1\",\n";
  out << "  \"seed\":" << simulation.config().seed << ",\n";
  out << "  \"logic_tick\":" << simulation.logicTick() << ",\n";
  out << "  \"logic_time\":" << simulation.logicTime() << ",\n";
  out << "  \"render_time\":" << renderTime << ",\n";
  out << "  \"interpolation_alpha\":" << simulation.interpolationAlpha() << ",\n";
  out << "  \"equipment\":\"" << escapeJson(simulation.equipment().active().id) << "\",\n";
  out << "  \"assets\":{\"skeleton_hash\":\"" << hexU64(simulation.assets().skeletonHash)
      << "\",\"mesh_hash\":\"" << hexU64(simulation.assets().meshHash)
      << "\",\"character_hash\":\"" << hexU64(simulation.assets().combinedHash)
      << "\",\"equipment_catalog_hash\":\"" << hexU64(simulation.equipment().catalogHash())
      << "\"},\n";
  out << "  \"state_hash\":\"" << hexU64(simulation.stateHash(renderTime)) << "\",\n";
  out << "  \"joints\":[";
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (i != 0U) {
      out << ',';
    }
    out << "{\"index\":" << i
        << ",\"name\":\"" << escapeJson(simulation.assets().skeleton.bones[i].name)
        << "\",\"position\":";
    writeVec3(out, joints[i]);
    out << '}';
  }
  out << "],\n";
  out << "  \"equipment_switch\":";
  writeSwitchEvents(out, simulation.switchEvents());
  out << ",\n  \"pipeline\":";
  writePipeline(out, pipeline);
  out << "\n}\n";
  finishOutput(out, path);
}

}  // namespace lcr
