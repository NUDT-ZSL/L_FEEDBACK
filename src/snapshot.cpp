#include "snapshot.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

namespace cr {
namespace {

std::string snapshotFileName(std::uint64_t version) {
    return "snapshot_" + zeroPaddedVersion(version) + ".json";
}

bool equalField(const Light& light, const std::string& field, const Light& reference) {
    if (field == "position") {
        return light.position == reference.position;
    }
    if (field == "color") {
        return light.color == reference.color;
    }
    if (field == "intensity") {
        return light.intensity == reference.intensity;
    }
    if (field == "radius") {
        return light.radius == reference.radius;
    }
    return false;
}

Json fieldValue(const Light& light, const std::string& field) {
    if (field == "position") {
        return vec3ToJson(light.position);
    }
    if (field == "color") {
        return vec3ToJson(light.color);
    }
    if (field == "intensity") {
        return Json(light.intensity);
    }
    if (field == "radius") {
        return Json(light.radius);
    }
    return Json();
}

std::uint64_t parseHash(const Json& value) {
    if (value.is_number_unsigned()) {
        return value.get<std::uint64_t>();
    }
    if (value.is_string()) {
        return std::stoull(value.get<std::string>(), nullptr, 16);
    }
    throw std::invalid_argument("invalid commit hash");
}

}  // namespace

Json SnapshotDiff::toJson() const {
    Json addedJson = Json::array();
    for (const Light& light : added) {
        addedJson.push_back(lightToJson(light));
    }
    Json modifiedJson = Json::array();
    for (const LightFieldChange& change : modified) {
        modifiedJson.push_back(
            {{"id", change.lightId}, {"fields", change.fields}, {"changes", change.changes}});
    }
    return Json{{"baseVersion", baseVersion},
                {"targetVersion", targetVersion},
                {"added", addedJson},
                {"removed", removed},
                {"modified", modifiedJson}};
}

SnapshotStore::SnapshotStore(fs::path dataDirectory, std::vector<Light> defaultLights)
    : dataDirectory_(std::move(dataDirectory)),
      snapshotDirectory_(dataDirectory_ / "snapshots"),
      stateDirectory_(dataDirectory_ / "state"),
      commitLogPath_(snapshotDirectory_ / "commits.log"),
      currentStatePath_(stateDirectory_ / "current_lights.json") {
    ensureDirectory(snapshotDirectory_);
    ensureDirectory(stateDirectory_);
    validateLights(defaultLights, "default lights");
    currentLights_ = std::move(defaultLights);
    recover();
    loadCurrentState();
}

void SnapshotStore::validateLights(const std::vector<Light>& lights,
                                   const std::string& context) {
    std::set<std::string> identifiers;
    for (const Light& light : lights) {
        if (light.id.empty()) {
            throw AppError(E_DATA_INVALID, context + " contains an empty light id");
        }
        if (!identifiers.insert(light.id).second) {
            throw AppError(E_DATA_INVALID, context + " contains duplicate light id '" +
                                                    light.id + "'");
        }
        if (light.intensity < 0.0F || light.radius <= 0.0F) {
            throw AppError(E_DATA_INVALID, context + " contains invalid light '" + light.id +
                                                    "'");
        }
    }
}

void SnapshotStore::recover() {
    committed_.clear();
    if (pathExists(commitLogPath_)) {
        const std::string content = readTextFile(commitLogPath_);
        std::istringstream lines(content);
        std::string line;
        std::size_t lineNumber = 0;
        while (std::getline(lines, line)) {
            ++lineNumber;
            if (line.empty()) {
                continue;
            }
            try {
                const Json record = Json::parse(line);
                const std::uint64_t version = record.at("version").get<std::uint64_t>();
                const std::string fileName = record.at("file").get<std::string>();
                const std::string timestamp = record.at("timestamp").get<std::string>();
                const std::uint64_t hash = parseHash(record.at("fnv1a64"));
                if (version == 0 || fileName != snapshotFileName(version)) {
                    std::cerr << "warning: ignoring malformed commit record at line "
                              << lineNumber << '\n';
                    continue;
                }
                const fs::path path = snapshotDirectory_ / fileName;
                if (!isRegularFile(path)) {
                    std::cerr << "warning: committed snapshot file is missing: "
                              << path.string() << '\n';
                    continue;
                }
                const std::uint64_t actualHash = fnv1a64(readBinaryFile(path));
                if (actualHash != hash) {
                    std::cerr << "warning: snapshot hash mismatch, not recovering version "
                              << version << '\n';
                    continue;
                }
                committed_[version] =
                    CommitRecord{version, fileName, timestamp, hash};
            } catch (const AppError& error) {
                if (std::string(error.code()) == E_IO_DENIED) {
                    throw;
                }
                std::cerr << "warning: unreadable commit record at line " << lineNumber
                          << ": " << error.what() << '\n';
            } catch (const std::exception& error) {
                std::cerr << "warning: malformed commit record at line " << lineNumber
                          << ": " << error.what() << '\n';
            }
        }
    }
    scanHighestSnapshotVersion();
}

void SnapshotStore::scanHighestSnapshotVersion() {
    std::uint64_t highest = 0;
    for (const auto& entry : committed_) {
        highest = std::max(highest, entry.first);
    }
    std::error_code error;
    for (fs::directory_iterator iterator(snapshotDirectory_, error), end; iterator != end;
         iterator.increment(error)) {
        if (error) {
            throw AppError(error == std::errc::permission_denied ? E_IO_DENIED : E_IO,
                           "cannot scan snapshot directory '" +
                               snapshotDirectory_.string() + "': " + error.message());
        }
        if (!iterator->is_regular_file()) {
            continue;
        }
        const std::string name = iterator->path().filename().string();
        static const std::regex pattern(R"(snapshot_(\d{6,})\.json)");
        std::smatch match;
        if (std::regex_match(name, match, pattern)) {
            try {
                highest = std::max(
                    highest, static_cast<std::uint64_t>(std::stoull(match[1].str())));
            } catch (const std::exception&) {
                // Ignore malformed names; they are not part of the committed sequence.
            }
        }
    }
    if (error) {
        throw AppError(error == std::errc::permission_denied ? E_IO_DENIED : E_IO,
                       "cannot scan snapshot directory '" + snapshotDirectory_.string() +
                           "': " + error.message());
    }
    nextVersion_ = highest + 1;
}

void SnapshotStore::loadCurrentState() {
    if (!pathExists(currentStatePath_)) {
        if (!committed_.empty()) {
            const Snapshot snapshot = loadSnapshot(committed_.rbegin()->first);
            currentLights_ = snapshot.lights;
        }
        return;
    }
    try {
        const Json document = Json::parse(readTextFile(currentStatePath_));
        std::vector<Light> lights;
        for (const Json& lightJson : document.at("lights")) {
            lights.push_back(lightFromJson(lightJson, "current state light"));
        }
        validateLights(lights, "current state");
        currentLights_ = std::move(lights);
    } catch (const AppError& error) {
        if (std::string(error.code()) == E_IO_DENIED) {
            throw;
        }
        std::cerr << "warning: current state is invalid; using last committed snapshot: "
                  << error.what() << '\n';
        if (!committed_.empty()) {
            currentLights_ = loadSnapshot(committed_.rbegin()->first).lights;
        }
    } catch (const std::exception& error) {
        std::cerr << "warning: current state is invalid; using last committed snapshot: "
                  << error.what() << '\n';
        if (!committed_.empty()) {
            currentLights_ = loadSnapshot(committed_.rbegin()->first).lights;
        }
    }
}

Snapshot SnapshotStore::parseSnapshotJson(const Json& document, const fs::path& path,
                                          std::uint64_t expectedVersion) const {
    const std::uint64_t version = document.at("version").get<std::uint64_t>();
    if (version != expectedVersion) {
        throw AppError(E_DATA_INVALID, "snapshot version mismatch in '" + path.string() + "'");
    }
    Snapshot snapshot;
    snapshot.version = version;
    snapshot.timestamp = document.at("timestamp").get<std::string>();
    for (const Json& lightJson : document.at("lights")) {
        snapshot.lights.push_back(lightFromJson(lightJson, "snapshot light"));
    }
    validateLights(snapshot.lights, "snapshot " + std::to_string(version));
    return snapshot;
}

Snapshot SnapshotStore::readSnapshotFile(std::uint64_t version) const {
    const fs::path path = snapshotDirectory_ / snapshotFileName(version);
    if (!pathExists(path) || !isRegularFile(path)) {
        throw AppError(E_SNAPSHOT_NOT_FOUND,
                       "snapshot version " + std::to_string(version) + " does not exist");
    }
    try {
        return parseSnapshotJson(Json::parse(readTextFile(path)), path, version);
    } catch (const AppError&) {
        throw;
    } catch (const std::exception& error) {
        throw AppError(E_DATA_INVALID,
                       "invalid snapshot version " + std::to_string(version) + ": " +
                           error.what());
    }
}

Snapshot SnapshotStore::loadSnapshot(std::uint64_t version) const {
    if (committed_.find(version) == committed_.end()) {
        throw AppError(E_SNAPSHOT_NOT_FOUND,
                       "snapshot version " + std::to_string(version) +
                           " is not committed");
    }
    return readSnapshotFile(version);
}

void SnapshotStore::writeCurrentState() const {
    const Json document = {
        {"savedAt", utcTimestampNow()},
        {"lights", Json::array()},
    };
    Json output = document;
    output["lights"] = Json::array();
    for (const Light& light : currentLights_) {
        output["lights"].push_back(lightToJson(light));
    }
    writeTextAtomic(currentStatePath_, output.dump(2) + "\n");
}

void SnapshotStore::ensureInitialSnapshot() {
    if (!committed_.empty()) {
        return;
    }
    createSnapshot();
}

std::uint64_t SnapshotStore::createSnapshot() {
    validateLights(currentLights_, "current lights");
    while (pathExists(snapshotDirectory_ / snapshotFileName(nextVersion_))) {
        ++nextVersion_;
    }
    const std::uint64_t version = nextVersion_;
    Snapshot snapshot;
    snapshot.version = version;
    snapshot.timestamp = utcTimestampNow();
    snapshot.lights = currentLights_;

    Json document = {
        {"version", version},
        {"timestamp", snapshot.timestamp},
        {"lights", Json::array()},
    };
    for (const Light& light : snapshot.lights) {
        document["lights"].push_back(lightToJson(light));
    }
    const std::string serialized = document.dump(2) + "\n";
    const fs::path path = snapshotDirectory_ / snapshotFileName(version);
    writeTextAtomic(path, serialized);

    const std::uint64_t hash = fnv1a64(serialized);
    const Json commit = {
        {"version", version},
        {"file", snapshotFileName(version)},
        {"timestamp", snapshot.timestamp},
        {"fnv1a64", hash},
    };
    appendTextDurable(commitLogPath_, commit.dump() + "\n");
    committed_[version] = CommitRecord{version, snapshotFileName(version),
                                       snapshot.timestamp, hash};
    ++nextVersion_;
    writeCurrentState();
    return version;
}

void SnapshotStore::rollback(std::uint64_t version) {
    const Snapshot snapshot = loadSnapshot(version);
    currentLights_ = snapshot.lights;
    writeCurrentState();
}

void SnapshotStore::setCurrentLights(std::vector<Light> lights) {
    validateLights(lights, "current lights");
    currentLights_ = std::move(lights);
    writeCurrentState();
}

const Light& SnapshotStore::requireCurrentLight(const std::string& id,
                                                const std::string& context) const {
    for (const Light& light : currentLights_) {
        if (light.id == id) {
            return light;
        }
    }
    throw AppError(E_SNAPSHOT_CONFLICT,
                   context + " references missing current light '" + id + "'");
}

void SnapshotStore::applyChangedFields(Light& destination, const Light& target,
                                       const std::vector<std::string>& fields) {
    for (const std::string& field : fields) {
        if (field == "position") {
            destination.position = target.position;
        } else if (field == "color") {
            destination.color = target.color;
        } else if (field == "intensity") {
            destination.intensity = target.intensity;
        } else if (field == "radius") {
            destination.radius = target.radius;
        }
    }
}

SnapshotDiff SnapshotStore::diff(std::uint64_t baseVersion,
                                 std::uint64_t targetVersion) const {
    const Snapshot base = loadSnapshot(baseVersion);
    const Snapshot target = loadSnapshot(targetVersion);
    SnapshotDiff result;
    result.baseVersion = baseVersion;
    result.targetVersion = targetVersion;

    std::map<std::string, const Light*> baseLights;
    std::map<std::string, const Light*> targetLights;
    for (const Light& light : base.lights) {
        baseLights[light.id] = &light;
    }
    for (const Light& light : target.lights) {
        targetLights[light.id] = &light;
    }

    for (const auto& entry : targetLights) {
        if (baseLights.find(entry.first) == baseLights.end()) {
            result.added.push_back(*entry.second);
        }
    }
    for (const auto& entry : baseLights) {
        if (targetLights.find(entry.first) == targetLights.end()) {
            result.removed.push_back(entry.first);
            continue;
        }
        const Light& before = *entry.second;
        const Light& after = *targetLights.at(entry.first);
        LightFieldChange change;
        change.lightId = entry.first;
        const std::array<std::string, 4> fieldNames = {"position", "color", "intensity",
                                                       "radius"};
        for (const std::string& field : fieldNames) {
            if (!equalField(before, field, after)) {
                change.fields.push_back(field);
                change.changes[field] = {{"base", fieldValue(before, field)},
                                         {"target", fieldValue(after, field)}};
            }
        }
        if (!change.fields.empty()) {
            result.modified.push_back(std::move(change));
        }
    }

    return result;
}

void SnapshotStore::merge(std::uint64_t baseVersion, std::uint64_t targetVersion) {
    const Snapshot base = loadSnapshot(baseVersion);
    const Snapshot target = loadSnapshot(targetVersion);
    const SnapshotDiff difference = diff(baseVersion, targetVersion);
    std::map<std::string, const Light*> baseLights;
    for (const Light& light : base.lights) {
        baseLights[light.id] = &light;
    }

    std::vector<Light> merged = currentLights_;
    auto findInMerged = [&](const std::string& id) -> Light* {
        for (Light& light : merged) {
            if (light.id == id) {
                return &light;
            }
        }
        return nullptr;
    };

    for (const Light& added : difference.added) {
        if (findInMerged(added.id) != nullptr) {
            throw AppError(E_SNAPSHOT_CONFLICT,
                           "cannot add light '" + added.id +
                               "': it already exists in the current scene");
        }
    }
    for (const std::string& removedId : difference.removed) {
        const Light* current = findInMerged(removedId);
        const Light& baseline = *baseLights.at(removedId);
        if (current == nullptr || *current != baseline) {
            throw AppError(E_SNAPSHOT_CONFLICT,
                           "cannot remove light '" + removedId +
                               "': current scene differs from baseline");
        }
    }
    for (const LightFieldChange& change : difference.modified) {
        const Light* current = findInMerged(change.lightId);
        const Light& baseline = *baseLights.at(change.lightId);
        if (current == nullptr) {
            throw AppError(E_SNAPSHOT_CONFLICT,
                           "cannot modify light '" + change.lightId +
                               "': current scene is missing it");
        }
        for (const std::string& field : change.fields) {
            if (!equalField(*current, field, baseline)) {
                throw AppError(E_SNAPSHOT_CONFLICT,
                               "cannot modify light '" + change.lightId + "' field '" +
                                   field + "': current value differs from baseline");
            }
        }
    }

    for (const Light& added : difference.added) {
        merged.push_back(added);
    }
    merged.erase(std::remove_if(merged.begin(), merged.end(),
                                [&](const Light& light) {
                                    return std::find(difference.removed.begin(),
                                                     difference.removed.end(),
                                                     light.id) !=
                                           difference.removed.end();
                                }),
                 merged.end());
    for (const LightFieldChange& change : difference.modified) {
        Light* current = findInMerged(change.lightId);
        const Light& targetLight = *std::find_if(
            target.lights.begin(), target.lights.end(),
            [&](const Light& light) { return light.id == change.lightId; });
        applyChangedFields(*current, targetLight, change.fields);
    }

    validateLights(merged, "merged lights");
    currentLights_ = std::move(merged);
    writeCurrentState();
}

}  // namespace cr

