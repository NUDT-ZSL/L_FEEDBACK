#include "gl_api.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <vector>

namespace terrain::gl {

namespace {
ProcResolver gProcResolver = nullptr;
}

#define TERRAIN_GL_FUNCTIONS(X) \
    X(glCreateShader) \
    X(glShaderSource) \
    X(glCompileShader) \
    X(glGetShaderiv) \
    X(glGetShaderInfoLog) \
    X(glDeleteShader) \
    X(glCreateProgram) \
    X(glAttachShader) \
    X(glLinkProgram) \
    X(glGetProgramiv) \
    X(glGetProgramInfoLog) \
    X(glDeleteProgram) \
    X(glUseProgram) \
    X(glGetUniformLocation) \
    X(glUniform1i) \
    X(glUniform1f) \
    X(glUniform3f) \
    X(glUniformMatrix4fv) \
    X(glGenVertexArrays) \
    X(glBindVertexArray) \
    X(glDeleteVertexArrays) \
    X(glGenBuffers) \
    X(glBindBuffer) \
    X(glBufferData) \
    X(glBufferSubData) \
    X(glBindBufferBase) \
    X(glDeleteBuffers) \
    X(glGenTextures) \
    X(glBindTexture) \
    X(glTexImage2D) \
    X(glTexParameteri) \
    X(glDeleteTextures) \
    X(glActiveTexture) \
    X(glGenFramebuffers) \
    X(glBindFramebuffer) \
    X(glFramebufferTexture2D) \
    X(glCheckFramebufferStatus) \
    X(glGenRenderbuffers) \
    X(glBindRenderbuffer) \
    X(glRenderbufferStorage) \
    X(glFramebufferRenderbuffer) \
    X(glDeleteFramebuffers) \
    X(glDeleteRenderbuffers) \
    X(glDrawElementsInstanced) \
    X(glDrawArrays) \
    X(glEnableVertexAttribArray) \
    X(glDisableVertexAttribArray) \
    X(glVertexAttribPointer) \
    X(glClear) \
    X(glClearColor) \
    X(glEnable) \
    X(glDisable) \
    X(glDepthFunc) \
    X(glDepthMask) \
    X(glViewport) \
    X(glPolygonMode) \
    X(glLineWidth) \
    X(glReadPixels) \
    X(glReadBuffer) \
    X(glPixelStorei) \
    X(glFinish) \
    X(glFlush) \
    X(glBlitFramebuffer) \
    X(glGetError) \
    X(glGetString) \
    X(glGetIntegerv)

#define TERRAIN_DEFINE_POINTER(name) decltype(name) name = nullptr;
TERRAIN_GL_FUNCTIONS(TERRAIN_DEFINE_POINTER)
#undef TERRAIN_DEFINE_POINTER

bool loadFunctions(std::string& error) {
    std::vector<std::string> missing;
#define TERRAIN_LOAD_POINTER(name) \
    do { \
        const GLFWglproc proc = gProcResolver != nullptr \
            ? reinterpret_cast<GLFWglproc>(gProcResolver(#name)) \
            : glfwGetProcAddress(#name); \
        if (proc == nullptr) { \
            missing.emplace_back(#name); \
        } else { \
            name = reinterpret_cast<decltype(name)>(proc); \
        } \
    } while (false);
    TERRAIN_GL_FUNCTIONS(TERRAIN_LOAD_POINTER)
#undef TERRAIN_LOAD_POINTER

    if (!missing.empty()) {
        error = "missing OpenGL entry points:";
        for (const std::string& name : missing) {
            error += " " + name;
        }
        return false;
    }
    return true;
}

#undef TERRAIN_GL_FUNCTIONS

void setProcResolver(ProcResolver resolver) {
    gProcResolver = resolver;
}

}  // namespace terrain::gl

