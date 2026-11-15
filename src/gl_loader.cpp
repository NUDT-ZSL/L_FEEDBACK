#include "gl_loader.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <sstream>

namespace demo {

#define DEMO_DEFINE_GL_FUNCTION(type, name) type name = nullptr;
DEMO_GL_FUNCTION_LIST(DEMO_DEFINE_GL_FUNCTION)
#undef DEMO_DEFINE_GL_FUNCTION

bool loadOpenGlFunctions(std::string &missingFunctions) {
    std::ostringstream missing;

#define DEMO_LOAD_GL_FUNCTION(type, name) \
    name = reinterpret_cast<type>(glfwGetProcAddress(#name)); \
    if (name == nullptr) { \
        missing << (#name) << ' '; \
    }
    DEMO_GL_FUNCTION_LIST(DEMO_LOAD_GL_FUNCTION)
#undef DEMO_LOAD_GL_FUNCTION

    missingFunctions = missing.str();
    return missingFunctions.empty();
}

const char *glErrorName(GLenum error) {
    switch (error) {
        case GL_NO_ERROR:
            return "GL_NO_ERROR";
        case GL_INVALID_ENUM:
            return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE:
            return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION:
            return "GL_INVALID_OPERATION";
        case GL_OUT_OF_MEMORY:
            return "GL_OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            return "GL_INVALID_FRAMEBUFFER_OPERATION";
        default:
            return "GL_UNKNOWN_ERROR";
    }
}

} // namespace demo
