#include "gl_api.h"

#include <GLFW/glfw3.h>

#include <sstream>

namespace subdiv {
namespace gl {

PFNGLCLEARCOLORPROC ClearColor = nullptr;
PFNGLCLEARPROC Clear = nullptr;
PFNGLVIEWPORTPROC Viewport = nullptr;
PFNGLENABLEPROC Enable = nullptr;
PFNGLDISABLEPROC Disable = nullptr;
PFNGLDEPTHFUNCPROC DepthFunc = nullptr;
PFNGLCULLFACEPROC CullFace = nullptr;
PFNGLFRONTFACEPROC FrontFace = nullptr;
PFNGLPOLYGONMODEPROC PolygonMode = nullptr;
PFNGLGETERRORPROC GetError = nullptr;
PFNGLGETSTRINGPROC GetString = nullptr;
PFNGLPIXELSTOREIPROC PixelStorei = nullptr;
PFNGLREADPIXELSPROC ReadPixels = nullptr;
PFNGLFINISHPROC Finish = nullptr;

PFNGLGENVERTEXARRAYSPROC GenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC BindVertexArray = nullptr;
PFNGLDELETEVERTEXARRAYSPROC DeleteVertexArrays = nullptr;
PFNGLGENBUFFERSPROC GenBuffers = nullptr;
PFNGLBINDBUFFERPROC BindBuffer = nullptr;
PFNGLBUFFERDATAPROC BufferData = nullptr;
PFNGLDELETEBUFFERSPROC DeleteBuffers = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray = nullptr;
PFNGLVERTEXATTRIBPOINTERPROC VertexAttribPointer = nullptr;

PFNGLCREATESHADERPROC CreateShader = nullptr;
PFNGLSHADERSOURCEPROC ShaderSource = nullptr;
PFNGLCOMPILESHADERPROC CompileShader = nullptr;
PFNGLGETSHADERIVPROC GetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog = nullptr;
PFNGLDELETESHADERPROC DeleteShader = nullptr;
PFNGLCREATEPROGRAMPROC CreateProgram = nullptr;
PFNGLATTACHSHADERPROC AttachShader = nullptr;
PFNGLLINKPROGRAMPROC LinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC GetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog = nullptr;
PFNGLDELETEPROGRAMPROC DeleteProgram = nullptr;
PFNGLUSEPROGRAMPROC UseProgram = nullptr;
PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation = nullptr;
PFNGLUNIFORM1FPROC Uniform1f = nullptr;
PFNGLUNIFORM1IPROC Uniform1i = nullptr;
PFNGLUNIFORM3FPROC Uniform3f = nullptr;
PFNGLUNIFORMMATRIX4FVPROC UniformMatrix4fv = nullptr;
PFNGLDRAWELEMENTSPROC DrawElements = nullptr;

namespace {

template <typename Function>
bool load(Function& function, const char* name, std::vector<std::string>& missing) {
    function = reinterpret_cast<Function>(glfwGetProcAddress(name));
    if (function == nullptr) {
        missing.emplace_back(name);
        return false;
    }
    return true;
}

}  // namespace

bool loadFunctions(std::vector<std::string>& missingFunctions) {
    missingFunctions.clear();
    bool ok = true;

#define SUBDIV_LOAD(fn) ok = load(fn, #fn, missingFunctions) && ok
    SUBDIV_LOAD(ClearColor);
    SUBDIV_LOAD(Clear);
    SUBDIV_LOAD(Viewport);
    SUBDIV_LOAD(Enable);
    SUBDIV_LOAD(Disable);
    SUBDIV_LOAD(DepthFunc);
    SUBDIV_LOAD(CullFace);
    SUBDIV_LOAD(FrontFace);
    SUBDIV_LOAD(PolygonMode);
    SUBDIV_LOAD(GetError);
    SUBDIV_LOAD(GetString);
    SUBDIV_LOAD(PixelStorei);
    SUBDIV_LOAD(ReadPixels);
    SUBDIV_LOAD(Finish);

    SUBDIV_LOAD(GenVertexArrays);
    SUBDIV_LOAD(BindVertexArray);
    SUBDIV_LOAD(DeleteVertexArrays);
    SUBDIV_LOAD(GenBuffers);
    SUBDIV_LOAD(BindBuffer);
    SUBDIV_LOAD(BufferData);
    SUBDIV_LOAD(DeleteBuffers);
    SUBDIV_LOAD(EnableVertexAttribArray);
    SUBDIV_LOAD(VertexAttribPointer);

    SUBDIV_LOAD(CreateShader);
    SUBDIV_LOAD(ShaderSource);
    SUBDIV_LOAD(CompileShader);
    SUBDIV_LOAD(GetShaderiv);
    SUBDIV_LOAD(GetShaderInfoLog);
    SUBDIV_LOAD(DeleteShader);
    SUBDIV_LOAD(CreateProgram);
    SUBDIV_LOAD(AttachShader);
    SUBDIV_LOAD(LinkProgram);
    SUBDIV_LOAD(GetProgramiv);
    SUBDIV_LOAD(GetProgramInfoLog);
    SUBDIV_LOAD(DeleteProgram);
    SUBDIV_LOAD(UseProgram);
    SUBDIV_LOAD(GetUniformLocation);
    SUBDIV_LOAD(Uniform1f);
    SUBDIV_LOAD(Uniform1i);
    SUBDIV_LOAD(Uniform3f);
    SUBDIV_LOAD(UniformMatrix4fv);
    SUBDIV_LOAD(DrawElements);
#undef SUBDIV_LOAD

    return ok;
}

const char* errorName(GLenum error) {
    switch (error) {
        case GL_NO_ERROR: return "GL_NO_ERROR";
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        default: break;
    }
    return "unknown GL error";
}

}  // namespace gl
}  // namespace subdiv

