#include "json.hpp"
#include "sha256.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expectSha(const std::string& input, const std::string& expected) {
    const std::string actual = lfb::Sha256::hash(input);
    check(actual == expected, "SHA-256 mismatch for '" + input + "': " + actual);
}

}  // namespace

int main() {
    expectSha("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expectSha("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expectSha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    const auto parsed = lfb::json::parse(
        R"({"resolution":[8,4],"samples":[[0,0,1]],"layers":[],"lights":[]})");
    check(parsed.isObject(), "JSON root should be an object");
    const auto* resolution = parsed.find("resolution");
    check(resolution != nullptr && resolution->isArray() &&
              resolution->asArray().size() == 2 &&
              resolution->asArray()[0].asNumber() == 8.0,
          "JSON array parsing");
    check(parsed.find("missing") == nullptr, "missing key lookup");

    bool duplicateRejected = false;
    try {
        (void)lfb::json::parse(R"({"a":1,"a":2})");
    } catch (const lfb::json::ParseError&) {
        duplicateRejected = true;
    }
    check(duplicateRejected, "duplicate JSON key must be rejected");

    bool trailingRejected = false;
    try {
        (void)lfb::json::parse(R"({"a":1} trailing)");
    } catch (const lfb::json::ParseError&) {
        trailingRejected = true;
    }
    check(trailingRejected, "trailing JSON data must be rejected");

    if (failures != 0) {
        std::cerr << failures << " unit test(s) failed\n";
        return 1;
    }
    std::cout << "unit tests passed\n";
    return 0;
}
