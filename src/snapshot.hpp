#pragma once

#include "common.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cr {

struct Snapshot {
    std::uint64_t version = 0;
    std::string timestamp;
    std::vector<Light> lights;
};

struct CommitRecord {
    std::uint64_t version = 0;
    std::string fileName;
    std::string timestamp;
    std::uint64_t hash = 0;
};

struct LightFieldChange {
    std::string lightId;
    std::vector<std::string> fields;
    Json changes;
};

struct SnapshotDiff {
    std::uint64_t baseVersion = 0;
    std::uint64_t targetVersion = 0;
    std::vector<Light> added;
    std::vector<std::string> removed;
    std::vector<LightFieldChange> modified;

    [[nodiscard]] Json toJson() const;
};

class SnapshotStore {
public:
    SnapshotStore(fs::path dataDirectory, std::vector<Light> defaultLights);

    void ensureInitialSnapshot();
    std::uint64_t createSnapshot();
    [[nodiscard]] Snapshot loadSnapshot(std::uint64_t version) const;
    void rollback(std::uint64_t version);
    [[nodiscard]] SnapshotDiff diff(std::uint64_t baseVersion,
                                    std::uint64_t targetVersion) const;
    void merge(std::uint64_t baseVersion, std::uint64_t targetVersion);

    [[nodiscard]] const std::vector<Light>& currentLights() const noexcept {
        return currentLights_;
    }
    void setCurrentLights(std::vector<Light> lights);
    [[nodiscard]] const std::map<std::uint64_t, CommitRecord>& committed() const noexcept {
        return committed_;
    }
    [[nodiscard]] std::uint64_t nextVersion() const noexcept { return nextVersion_; }
    [[nodiscard]] const fs::path& snapshotDirectory() const noexcept {
        return snapshotDirectory_;
    }
    [[nodiscard]] const fs::path& commitLogPath() const noexcept { return commitLogPath_; }

private:
    void recover();
    void loadCurrentState();
    void scanHighestSnapshotVersion();
    [[nodiscard]] Snapshot parseSnapshotJson(const Json& document, const fs::path& path,
                                             std::uint64_t expectedVersion) const;
    [[nodiscard]] Snapshot readSnapshotFile(std::uint64_t version) const;
    void writeCurrentState() const;
    [[nodiscard]] const Light& requireCurrentLight(const std::string& id,
                                                   const std::string& context) const;
    static void validateLights(const std::vector<Light>& lights,
                               const std::string& context);
    static void applyChangedFields(Light& destination, const Light& target,
                                   const std::vector<std::string>& fields);

    fs::path dataDirectory_;
    fs::path snapshotDirectory_;
    fs::path stateDirectory_;
    fs::path commitLogPath_;
    fs::path currentStatePath_;
    std::vector<Light> currentLights_;
    std::map<std::uint64_t, CommitRecord> committed_;
    std::uint64_t nextVersion_ = 1;
};

}  // namespace cr

