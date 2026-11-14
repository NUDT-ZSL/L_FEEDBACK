#include "gl_loader.hpp"

#include "common.hpp"

#include <string>
#include <vector>

namespace cr::gl {

PFNGLGETERRORPROC glGetError = nullptr;
PFNGLGETINTEGERVPROC glGetIntegerv = nullptr;
PFNGLVIEWPORTPROC glViewport = nullptr;
PFNGLCLEARCOLORPROC glClearColor = nullptr;
PFNGLCLEARPROC glClear = nullptr;
PFNGLENABLEPROC glEnable = nullptr;
PFNGLDISABLEPROC glDisable = nullptr;
PFNGLDEPTHFUNCPROC glDepthFunc = nullptr;
PFNGLGENTEXTURESPROC glGenTextures = nullptr;
PFNGLDELETETEXTURESPROC glDeleteTextures = nullptr;
PFNGLBINDTEXTUREPROC glBindTexture = nullptr;
PFNGLACTIVETEXTUREPROC glActiveTexture = nullptr;
PFNGLTEXIMAGE2DPROC glTexImage2D = nullptr;
PFNGLTEXIMAGE3DPROC glTexImage3D = nullptr;
PFNGLTEXSUBIMAGE2DPROC glTexSubImage2D = nullptr;
PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D = nullptr;
PFNGLTEXPARAMETERIPROC glTexParameteri = nullptr;
PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers = nullptr;
PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer = nullptr;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers = nullptr;
PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D = nullptr;
PFNGLDRAWBUFFERSPROC glDrawBuffers = nullptr;
PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus = nullptr;
PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers = nullptr;
PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer = nullptr;
PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers = nullptr;
PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage = nullptr;
PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer = nullptr;
PFNGLCREATESHADERPROC glCreateShader = nullptr;
PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
PFNGLDELETESHADERPROC glDeleteShader = nullptr;
PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
PFNGLATTACHSHADERPROC glAttachShader = nullptr;
PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
PFNGLUNIFORM1IPROC glUniform1i = nullptr;
PFNGLUNIFORM1FPROC glUniform1f = nullptr;
PFNGLUNIFORM2FPROC glUniform2f = nullptr;
PFNGLUNIFORM3FPROC glUniform3f = nullptr;
PFNGLUNIFORM3FVPROC glUniform3fv = nullptr;
PFNGLUNIFORM1FVPROC glUniform1fv = nullptr;
PFNGLUNIFORM1IVPROC glUniform1iv = nullptr;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
PFNGLDRAWARRAYSPROC glDrawArrays = nullptr;
PFNGLREADPIXELSPROC glReadPixels = nullptr;
PFNGLFINISHPROC glFinish = nullptr;
PFNGLPIXELSTOREIPROC glPixelStorei = nullptr;

namespace {

template <typename Function>
void loadOne(const GlProcLoader& loader, const char* name, Function& target,
             std::vector<std::string>& missing) {
    target = reinterpret_cast<Function>(loader(name));
    if (target == nullptr) {
        missing.emplace_back(name);
    }
}

}  // namespace

void loadFunctions(const GlProcLoader& loader) {
    std::vector<std::string> missing;
    loadOne(loader, "glGetError", glGetError, missing);
    loadOne(loader, "glGetIntegerv", glGetIntegerv, missing);
    loadOne(loader, "glViewport", glViewport, missing);
    loadOne(loader, "glClearColor", glClearColor, missing);
    loadOne(loader, "glClear", glClear, missing);
    loadOne(loader, "glEnable", glEnable, missing);
    loadOne(loader, "glDisable", glDisable, missing);
    loadOne(loader, "glDepthFunc", glDepthFunc, missing);
    loadOne(loader, "glGenTextures", glGenTextures, missing);
    loadOne(loader, "glDeleteTextures", glDeleteTextures, missing);
    loadOne(loader, "glBindTexture", glBindTexture, missing);
    loadOne(loader, "glActiveTexture", glActiveTexture, missing);
    loadOne(loader, "glTexImage2D", glTexImage2D, missing);
    loadOne(loader, "glTexImage3D", glTexImage3D, missing);
    loadOne(loader, "glTexSubImage2D", glTexSubImage2D, missing);
    loadOne(loader, "glTexSubImage3D", glTexSubImage3D, missing);
    loadOne(loader, "glTexParameteri", glTexParameteri, missing);
    loadOne(loader, "glGenFramebuffers", glGenFramebuffers, missing);
    loadOne(loader, "glBindFramebuffer", glBindFramebuffer, missing);
    loadOne(loader, "glDeleteFramebuffers", glDeleteFramebuffers, missing);
    loadOne(loader, "glFramebufferTexture2D", glFramebufferTexture2D, missing);
    loadOne(loader, "glDrawBuffers", glDrawBuffers, missing);
    loadOne(loader, "glCheckFramebufferStatus", glCheckFramebufferStatus, missing);
    loadOne(loader, "glGenRenderbuffers", glGenRenderbuffers, missing);
    loadOne(loader, "glBindRenderbuffer", glBindRenderbuffer, missing);
    loadOne(loader, "glDeleteRenderbuffers", glDeleteRenderbuffers, missing);
    loadOne(loader, "glRenderbufferStorage", glRenderbufferStorage, missing);
    loadOne(loader, "glFramebufferRenderbuffer", glFramebufferRenderbuffer, missing);
    loadOne(loader, "glCreateShader", glCreateShader, missing);
    loadOne(loader, "glShaderSource", glShaderSource, missing);
    loadOne(loader, "glCompileShader", glCompileShader, missing);
    loadOne(loader, "glGetShaderiv", glGetShaderiv, missing);
    loadOne(loader, "glGetShaderInfoLog", glGetShaderInfoLog, missing);
    loadOne(loader, "glDeleteShader", glDeleteShader, missing);
    loadOne(loader, "glCreateProgram", glCreateProgram, missing);
    loadOne(loader, "glAttachShader", glAttachShader, missing);
    loadOne(loader, "glLinkProgram", glLinkProgram, missing);
    loadOne(loader, "glGetProgramiv", glGetProgramiv, missing);
    loadOne(loader, "glGetProgramInfoLog", glGetProgramInfoLog, missing);
    loadOne(loader, "glDeleteProgram", glDeleteProgram, missing);
    loadOne(loader, "glUseProgram", glUseProgram, missing);
    loadOne(loader, "glGetUniformLocation", glGetUniformLocation, missing);
    loadOne(loader, "glUniform1i", glUniform1i, missing);
    loadOne(loader, "glUniform1f", glUniform1f, missing);
    loadOne(loader, "glUniform2f", glUniform2f, missing);
    loadOne(loader, "glUniform3f", glUniform3f, missing);
    loadOne(loader, "glUniform3fv", glUniform3fv, missing);
    loadOne(loader, "glUniform1fv", glUniform1fv, missing);
    loadOne(loader, "glUniform1iv", glUniform1iv, missing);
    loadOne(loader, "glGenVertexArrays", glGenVertexArrays, missing);
    loadOne(loader, "glBindVertexArray", glBindVertexArray, missing);
    loadOne(loader, "glDeleteVertexArrays", glDeleteVertexArrays, missing);
    loadOne(loader, "glDrawArrays", glDrawArrays, missing);
    loadOne(loader, "glReadPixels", glReadPixels, missing);
    loadOne(loader, "glFinish", glFinish, missing);
    loadOne(loader, "glPixelStorei", glPixelStorei, missing);
    if (!missing.empty()) {
        std::string names;
        for (const std::string& name : missing) {
            if (!names.empty()) {
                names += ", ";
            }
            names += name;
        }
        throw AppError(E_OPENGL, "required OpenGL entry points are unavailable: " + names);
    }
}

}  // namespace cr::gl

