#include "gl_api.hpp"

#include <vector>

namespace orv {

PFNGLGETSTRINGPROC glGetString = nullptr;
PFNGLGETERRORPROC glGetError = nullptr;
PFNGLGETINTEGERVPROC glGetIntegerv = nullptr;
PFNGLVIEWPORTPROC glViewport = nullptr;
PFNGLCLEARCOLORPROC glClearColor = nullptr;
PFNGLCLEARPROC glClear = nullptr;
PFNGLCLEARBUFFERFVPROC glClearBufferfv = nullptr;
PFNGLCLEARBUFFERFIPROC glClearBufferfi = nullptr;
PFNGLENABLEPROC glEnable = nullptr;
PFNGLDISABLEPROC glDisable = nullptr;
PFNGLDEPTHFUNCPROC glDepthFunc = nullptr;
PFNGLDEPTHMASKPROC glDepthMask = nullptr;
PFNGLCULLFACEPROC glCullFace = nullptr;
PFNGLFRONTFACEPROC glFrontFace = nullptr;

PFNGLGENTEXTURESPROC glGenTextures = nullptr;
PFNGLBINDTEXTUREPROC glBindTexture = nullptr;
PFNGLACTIVETEXTUREPROC glActiveTexture = nullptr;
PFNGLTEXIMAGE2DPROC glTexImage2D = nullptr;
PFNGLTEXPARAMETERIPROC glTexParameteri = nullptr;
PFNGLDELETETEXTURESPROC glDeleteTextures = nullptr;
PFNGLPIXELSTOREIPROC glPixelStorei = nullptr;

PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers = nullptr;
PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer = nullptr;
PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D = nullptr;
PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus = nullptr;
PFNGLDRAWBUFFERSPROC glDrawBuffers = nullptr;
PFNGLDRAWBUFFERPROC glDrawBuffer = nullptr;
PFNGLREADBUFFERPROC glReadBuffer = nullptr;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers = nullptr;
PFNGLREADPIXELSPROC glReadPixels = nullptr;

PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
PFNGLBUFFERDATAPROC glBufferData = nullptr;
PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;

PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;

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
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
PFNGLUNIFORM1IPROC glUniform1i = nullptr;
PFNGLUNIFORM1FPROC glUniform1f = nullptr;
PFNGLUNIFORM1FVPROC glUniform1fv = nullptr;
PFNGLUNIFORM3FVPROC glUniform3fv = nullptr;
PFNGLUNIFORM4FVPROC glUniform4fv = nullptr;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = nullptr;

PFNGLDRAWELEMENTSPROC glDrawElements = nullptr;
PFNGLDRAWARRAYSPROC glDrawArrays = nullptr;
PFNGLFINISHPROC glFinish = nullptr;

bool loadOpenGlFunctions(const GlProcLoader& loader, std::string& missingFunctions) {
  std::vector<std::string> missing;
  const auto load = [&](const char* name, auto& function) {
    function = reinterpret_cast<std::decay_t<decltype(function)>>(loader(name));
    if (function == nullptr) {
      missing.emplace_back(name);
    }
  };

#define ORV_LOAD_GL(name) load(#name, name)
  ORV_LOAD_GL(glGetString);
  ORV_LOAD_GL(glGetError);
  ORV_LOAD_GL(glGetIntegerv);
  ORV_LOAD_GL(glViewport);
  ORV_LOAD_GL(glClearColor);
  ORV_LOAD_GL(glClear);
  ORV_LOAD_GL(glClearBufferfv);
  ORV_LOAD_GL(glClearBufferfi);
  ORV_LOAD_GL(glEnable);
  ORV_LOAD_GL(glDisable);
  ORV_LOAD_GL(glDepthFunc);
  ORV_LOAD_GL(glDepthMask);
  ORV_LOAD_GL(glCullFace);
  ORV_LOAD_GL(glFrontFace);
  ORV_LOAD_GL(glGenTextures);
  ORV_LOAD_GL(glBindTexture);
  ORV_LOAD_GL(glActiveTexture);
  ORV_LOAD_GL(glTexImage2D);
  ORV_LOAD_GL(glTexParameteri);
  ORV_LOAD_GL(glDeleteTextures);
  ORV_LOAD_GL(glPixelStorei);
  ORV_LOAD_GL(glGenFramebuffers);
  ORV_LOAD_GL(glBindFramebuffer);
  ORV_LOAD_GL(glFramebufferTexture2D);
  ORV_LOAD_GL(glCheckFramebufferStatus);
  ORV_LOAD_GL(glDrawBuffers);
  ORV_LOAD_GL(glDrawBuffer);
  ORV_LOAD_GL(glReadBuffer);
  ORV_LOAD_GL(glDeleteFramebuffers);
  ORV_LOAD_GL(glReadPixels);
  ORV_LOAD_GL(glGenBuffers);
  ORV_LOAD_GL(glBindBuffer);
  ORV_LOAD_GL(glBufferData);
  ORV_LOAD_GL(glDeleteBuffers);
  ORV_LOAD_GL(glGenVertexArrays);
  ORV_LOAD_GL(glBindVertexArray);
  ORV_LOAD_GL(glEnableVertexAttribArray);
  ORV_LOAD_GL(glVertexAttribPointer);
  ORV_LOAD_GL(glDeleteVertexArrays);
  ORV_LOAD_GL(glCreateShader);
  ORV_LOAD_GL(glShaderSource);
  ORV_LOAD_GL(glCompileShader);
  ORV_LOAD_GL(glGetShaderiv);
  ORV_LOAD_GL(glGetShaderInfoLog);
  ORV_LOAD_GL(glDeleteShader);
  ORV_LOAD_GL(glCreateProgram);
  ORV_LOAD_GL(glAttachShader);
  ORV_LOAD_GL(glLinkProgram);
  ORV_LOAD_GL(glGetProgramiv);
  ORV_LOAD_GL(glGetProgramInfoLog);
  ORV_LOAD_GL(glUseProgram);
  ORV_LOAD_GL(glDeleteProgram);
  ORV_LOAD_GL(glGetUniformLocation);
  ORV_LOAD_GL(glUniform1i);
  ORV_LOAD_GL(glUniform1f);
  ORV_LOAD_GL(glUniform1fv);
  ORV_LOAD_GL(glUniform3fv);
  ORV_LOAD_GL(glUniform4fv);
  ORV_LOAD_GL(glUniformMatrix4fv);
  ORV_LOAD_GL(glDrawElements);
  ORV_LOAD_GL(glDrawArrays);
  ORV_LOAD_GL(glFinish);
#undef ORV_LOAD_GL

  if (missing.empty()) {
    return true;
  }
  for (std::size_t i = 0; i < missing.size(); ++i) {
    if (i != 0) {
      missingFunctions += ", ";
    }
    missingFunctions += missing[i];
  }
  return false;
}

}  // namespace orv
