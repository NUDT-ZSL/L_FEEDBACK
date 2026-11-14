#pragma once

#include "json.h"

#include <stdexcept>
#include <string>

namespace cloud {

enum ExitCode : int {
    kExitOk = 0,
    kExitSnapshotMissing = 10,
    kExitSnapshotCrc = 11,
    kExitVersionUnsupported = 12,
    kExitOutOfMemory = 13,
    kExitSnapshotFormat = 14,
    kExitConfig = 20,
    kExitOutput = 21,
    kExitInternal = 30,
};

class BusinessError : public std::runtime_error {
public:
    BusinessError(std::string code, int exitCode, const std::string& message,
                  json::Value fields = json::Value::Object{})
        : std::runtime_error(message),
          code_(std::move(code)),
          exitCode_(exitCode),
          fields_(std::move(fields)) {}

    const std::string& code() const { return code_; }
    int exitCode() const { return exitCode_; }
    const json::Value& fields() const { return fields_; }

private:
    std::string code_;
    int exitCode_;
    json::Value fields_;
};

}  // namespace cloud

