#include "lf/gl_api.h"

#include "lf/error_code.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <sstream>

namespace gl {

Api gl;

namespace {

template <typename Function>
bool load_proc(Function& function, const char* name, std::string& missing) {
  const GLFWglproc symbol = glfwGetProcAddress(name);
  if (symbol == nullptr) {
    if (missing.empty()) {
      missing = name;
    }
    return false;
  }
  function = reinterpret_cast<Function>(symbol);
  return true;
}

}  // namespace

bool load_api(std::string& missingEntryPoint) {
  missingEntryPoint.clear();
  bool complete = true;

#define LF_LOAD(field, name) \
  complete = load_proc(gl.field, name, missingEntryPoint) && complete

  LF_LOAD(Viewport, "glViewport");
  LF_LOAD(ClearColor, "glClearColor");
  LF_LOAD(Clear, "glClear");
  LF_LOAD(Enable, "glEnable");
  LF_LOAD(Disable, "glDisable");
  LF_LOAD(DepthFunc, "glDepthFunc");
  LF_LOAD(DepthMask, "glDepthMask");
  LF_LOAD(CullFace, "glCullFace");
  LF_LOAD(FrontFace, "glFrontFace");
  LF_LOAD(BlendFunc, "glBlendFunc");
  LF_LOAD(PixelStorei, "glPixelStorei");
  LF_LOAD(GetIntegerv, "glGetIntegerv");
  LF_LOAD(GetError, "glGetError");
  LF_LOAD(GetString, "glGetString");
  LF_LOAD(Finish, "glFinish");
  LF_LOAD(CreateShader, "glCreateShader");
  LF_LOAD(ShaderSource, "glShaderSource");
  LF_LOAD(CompileShader, "glCompileShader");
  LF_LOAD(GetShaderiv, "glGetShaderiv");
  LF_LOAD(GetShaderInfoLog, "glGetShaderInfoLog");
  LF_LOAD(DeleteShader, "glDeleteShader");
  LF_LOAD(CreateProgram, "glCreateProgram");
  LF_LOAD(AttachShader, "glAttachShader");
  LF_LOAD(LinkProgram, "glLinkProgram");
  LF_LOAD(GetProgramiv, "glGetProgramiv");
  LF_LOAD(GetProgramInfoLog, "glGetProgramInfoLog");
  LF_LOAD(UseProgram, "glUseProgram");
  LF_LOAD(DeleteProgram, "glDeleteProgram");
  LF_LOAD(GetUniformLocation, "glGetUniformLocation");
  LF_LOAD(Uniform1i, "glUniform1i");
  LF_LOAD(Uniform1f, "glUniform1f");
  LF_LOAD(Uniform2f, "glUniform2f");
  LF_LOAD(Uniform3f, "glUniform3f");
  LF_LOAD(Uniform4f, "glUniform4f");
  LF_LOAD(Uniform1fv, "glUniform1fv");
  LF_LOAD(Uniform1iv, "glUniform1iv");
  LF_LOAD(Uniform3fv, "glUniform3fv");
  LF_LOAD(Uniform4fv, "glUniform4fv");
  LF_LOAD(UniformMatrix4fv, "glUniformMatrix4fv");
  LF_LOAD(UniformMatrix3fv, "glUniformMatrix3fv");
  LF_LOAD(GenVertexArrays, "glGenVertexArrays");
  LF_LOAD(BindVertexArray, "glBindVertexArray");
  LF_LOAD(DeleteVertexArrays, "glDeleteVertexArrays");
  LF_LOAD(GenBuffers, "glGenBuffers");
  LF_LOAD(BindBuffer, "glBindBuffer");
  LF_LOAD(BufferData, "glBufferData");
  LF_LOAD(DeleteBuffers, "glDeleteBuffers");
  LF_LOAD(EnableVertexAttribArray, "glEnableVertexAttribArray");
  LF_LOAD(VertexAttribPointer, "glVertexAttribPointer");
  LF_LOAD(VertexAttribIPointer, "glVertexAttribIPointer");
  LF_LOAD(DrawArrays, "glDrawArrays");
  LF_LOAD(DrawElements, "glDrawElements");
  LF_LOAD(GenTextures, "glGenTextures");
  LF_LOAD(BindTexture, "glBindTexture");
  LF_LOAD(ActiveTexture, "glActiveTexture");
  LF_LOAD(TexParameteri, "glTexParameteri");
  LF_LOAD(TexImage2D, "glTexImage2D");
  LF_LOAD(GenerateMipmap, "glGenerateMipmap");
  LF_LOAD(DeleteTextures, "glDeleteTextures");
  LF_LOAD(GenFramebuffers, "glGenFramebuffers");
  LF_LOAD(BindFramebuffer, "glBindFramebuffer");
  LF_LOAD(DeleteFramebuffers, "glDeleteFramebuffers");
  LF_LOAD(FramebufferTexture2D, "glFramebufferTexture2D");
  LF_LOAD(FramebufferRenderbuffer, "glFramebufferRenderbuffer");
  LF_LOAD(CheckFramebufferStatus, "glCheckFramebufferStatus");
  LF_LOAD(DrawBuffers, "glDrawBuffers");
  LF_LOAD(GenRenderbuffers, "glGenRenderbuffers");
  LF_LOAD(BindRenderbuffer, "glBindRenderbuffer");
  LF_LOAD(RenderbufferStorage, "glRenderbufferStorage");
  LF_LOAD(DeleteRenderbuffers, "glDeleteRenderbuffers");
  LF_LOAD(ReadPixels, "glReadPixels");

#undef LF_LOAD

  return complete;
}

}  // namespace gl

