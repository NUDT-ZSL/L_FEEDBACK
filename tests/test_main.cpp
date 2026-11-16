#include "test_runner.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: skelpreview_tests [work-directory]\n";
        return 2;
    }
    const std::filesystem::path workDirectory =
        argc == 2 ? std::filesystem::path(argv[1])
                  : std::filesystem::path("test_assets");
    try {
        return skel::runSelfTests(workDirectory);
    } catch (const std::exception& exception) {
        std::cerr << "FATAL: " << exception.what() << "\n";
        return 1;
    }
}

