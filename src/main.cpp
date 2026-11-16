#include "cli.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        return skel::runCli(argc, argv);
    } catch (const std::exception& exception) {
        std::cerr << "fatal: " << exception.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "fatal: unknown exception\n";
        return 1;
    }
}

