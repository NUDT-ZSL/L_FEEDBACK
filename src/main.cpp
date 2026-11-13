#include "app.hpp"
#include "terrain/cli.hpp"

#include <iostream>

int main(int argc, char** argv) {
    const terrain::ParseResult parsed = terrain::parseCommandLine(argc, argv);
    if (!parsed.ok) {
        std::cerr << parsed.error << "\n";
        return parsed.isPathError ? 3 : 2;
    }
    if (parsed.options.help) {
        std::cout << terrain::usageText();
        return 0;
    }

#if TERRAIN_HAS_GL
    if (parsed.options.backend != terrain::Backend::Software) {
        return terrain::runGlApp(parsed.options);
    }
#else
    if (parsed.options.backend == terrain::Backend::Gl) {
        std::cerr << "E_GL_INIT: this build does not include the OpenGL backend\n";
        return 4;
    }
#endif

    return terrain::runSoftwareApp(parsed.options);
}

