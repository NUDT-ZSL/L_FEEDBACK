#include "gl_api.hpp"

#include <GLFW/glfw3.h>

#include <sstream>
#include <string>

namespace skel::gl4 {

GenVertexArraysProc GenVertexArrays = nullptr;
BindVertexArrayProc BindVertexArray = nullptr;
DeleteVertexArraysProc DeleteVertexArrays = nullptr;
GenBuffersProc GenBuffers = nullptr;
BindBufferProc BindBuffer = nullptr;
BufferDataProc BufferData = nullptr;
DeleteBuffersProc DeleteBuffers = nullptr;
VertexAttribPointerProc VertexAttribPointer = nullptr;
EnableVertexAttribArrayProc EnableVertexAttribArray = nullptr;
CreateShaderProc CreateShader = nullptr;
ShaderSourceProc ShaderSource = nullptr;
CompileShaderProc CompileShader = nullptr;
GetShaderivProc GetShaderiv = nullptr;
GetShaderInfoLogProc GetShaderInfoLog = nullptr;
DeleteShaderProc DeleteShader = nullptr;
CreateProgramProc CreateProgram = nullptr;
AttachShaderProc AttachShader = nullptr;
LinkProgramProc LinkProgram = nullptr;
GetProgramivProc GetProgramiv = nullptr;
GetProgramInfoLogProc GetProgramInfoLog = nullptr;
DeleteProgramProc DeleteProgram = nullptr;
UseProgramProc UseProgram = nullptr;
GetUniformLocationProc GetUniformLocation = nullptr;
UniformMatrix4fvProc UniformMatrix4fv = nullptr;
Uniform3fvProc Uniform3fv = nullptr;
Uniform1iProc Uniform1i = nullptr;

namespace {

std::string missingFunctions;

template <typename Function>
bool loadOne(Function& function, const char* name) {
    function = reinterpret_cast<Function>(
        glfwGetProcAddress(name));
    if (function == nullptr) {
        if (!missingFunctions.empty()) {
            missingFunctions += ", ";
        }
        missingFunctions += name;
        return false;
    }
    return true;
}

} // namespace

bool load(std::string& diagnostic) {
    missingFunctions.clear();
    bool complete = true;
#define SKEL_LOAD(function) complete = loadOne(function, #function) && complete
    SKEL_LOAD(GenVertexArrays);
    SKEL_LOAD(BindVertexArray);
    SKEL_LOAD(DeleteVertexArrays);
    SKEL_LOAD(GenBuffers);
    SKEL_LOAD(BindBuffer);
    SKEL_LOAD(BufferData);
    SKEL_LOAD(DeleteBuffers);
    SKEL_LOAD(VertexAttribPointer);
    SKEL_LOAD(EnableVertexAttribArray);
    SKEL_LOAD(CreateShader);
    SKEL_LOAD(ShaderSource);
    SKEL_LOAD(CompileShader);
    SKEL_LOAD(GetShaderiv);
    SKEL_LOAD(GetShaderInfoLog);
    SKEL_LOAD(DeleteShader);
    SKEL_LOAD(CreateProgram);
    SKEL_LOAD(AttachShader);
    SKEL_LOAD(LinkProgram);
    SKEL_LOAD(GetProgramiv);
    SKEL_LOAD(GetProgramInfoLog);
    SKEL_LOAD(DeleteProgram);
    SKEL_LOAD(UseProgram);
    SKEL_LOAD(GetUniformLocation);
    SKEL_LOAD(UniformMatrix4fv);
    SKEL_LOAD(Uniform3fv);
    SKEL_LOAD(Uniform1i);
#undef SKEL_LOAD
    if (!complete) {
        diagnostic = "OpenGL 4.1 function loading failed: " + missingFunctions;
    }
    return complete;
}

} // namespace skel::gl4

