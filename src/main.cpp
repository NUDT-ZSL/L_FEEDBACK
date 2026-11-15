#include "application.hpp"
#include "error.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char **argv) {
    try {
        const demo::Options options = demo::parseCommandLine(argc, argv);
        if (options.showHelp) {
            demo::printUsage(argv[0]);
            return 0;
        }
        return demo::runApplication(options);
    } catch (const demo::AppError &error) {
        demo::reportError(error);
        return error.exitCode();
    } catch (const std::exception &error) {
        demo::reportUnexpectedError(error);
        return demo::exitCode(demo::ErrorCode::UnexpectedFailure);
    }
}
