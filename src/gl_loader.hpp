#pragma once

#include <cstddef>
#include <string>

// Minimal OpenGL 4.5 core declaration set. Runtime entry points are obtained from
// GLFW, which keeps the executable free of external shader/model resources and
// works with both the Windows SDK and a manually installed GLFW.

using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLbyte = signed char;
using GLubyte = unsigned char;
using GLshort = short;
using GLushort = unsigned short;
using GLint = int;
using GLuint = unsigned int;
using GLsizei = int;
using GLfloat = float;
using GLclampf = float;
using GLdouble = double;
using GLchar = char;
using GLintptr = std::ptrdiff_t;
using GLsizeiptr = std::ptrdiff_t;

#ifndef GL_APIENTRY
#if defined(_WIN32)
#define GL_APIENTRY __stdcall
#else
#define GL_APIENTRY
#endif
#endif

using PFNGLGETSTRINGPROC = const GLubyte *(GL_APIENTRY *)(GLenum);
using PFNGLGETERRORPROC = GLenum (GL_APIENTRY *)(void);
using PFNGLCLEARPROC = void (GL_APIENTRY *)(GLbitfield);
using PFNGLCLEARCOLORPROC = void (GL_APIENTRY *)(GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLENABLEPROC = void (GL_APIENTRY *)(GLenum);
using PFNGLDISABLEPROC = void (GL_APIENTRY *)(GLenum);
using PFNGLVIEWPORTPROC = void (GL_APIENTRY *)(GLint, GLint, GLsizei, GLsizei);
using PFNGLDEPTHFUNCPROC = void (GL_APIENTRY *)(GLenum);
using PFNGLGENVERTEXARRAYSPROC = void (GL_APIENTRY *)(GLsizei, GLuint *);
using PFNGLBINDVERTEXARRAYPROC = void (GL_APIENTRY *)(GLuint);
using PFNGLDELETEVERTEXARRAYSPROC = void (GL_APIENTRY *)(GLsizei, const GLuint *);
using PFNGLGENBUFFERSPROC = void (GL_APIENTRY *)(GLsizei, GLuint *);
using PFNGLBINDBUFFERPROC = void (GL_APIENTRY *)(GLenum, GLuint);
using PFNGLBUFFERDATAPROC = void (GL_APIENTRY *)(GLenum, GLsizeiptr, const void *, GLenum);
using PFNGLBUFFERSUBDATAPROC = void (GL_APIENTRY *)(GLenum, GLintptr, GLsizeiptr, const void *);
using PFNGLGETBUFFERSUBDATAPROC = void (GL_APIENTRY *)(GLenum, GLintptr, GLsizeiptr, void *);
using PFNGLDELETEBUFFERSPROC = void (GL_APIENTRY *)(GLsizei, const GLuint *);
using PFNGLENABLEVERTEXATTRIBARRAYPROC = void (GL_APIENTRY *)(GLuint);
using PFNGLVERTEXATTRIBPOINTERPROC =
    void (GL_APIENTRY *)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void *);
using PFNGLCREATESHADERPROC = GLuint (GL_APIENTRY *)(GLenum);
using PFNGLSHADERSOURCEPROC =
    void (GL_APIENTRY *)(GLuint, GLsizei, const GLchar *const *, const GLint *);
using PFNGLCOMPILESHADERPROC = void (GL_APIENTRY *)(GLuint);
using PFNGLGETSHADERIVPROC = void (GL_APIENTRY *)(GLuint, GLenum, GLint *);
using PFNGLGETSHADERINFOLOGPROC = void (GL_APIENTRY *)(GLuint, GLsizei, GLsizei *, GLchar *);
using PFNGLDELETESHADERPROC = void (GL_APIENTRY *)(GLuint);
using PFNGLCREATEPROGRAMPROC = GLuint (GL_APIENTRY *)(void);
using PFNGLATTACHSHADERPROC = void (GL_APIENTRY *)(GLuint, GLuint);
using PFNGLLINKPROGRAMPROC = void (GL_APIENTRY *)(GLuint);
using PFNGLGETPROGRAMIVPROC = void (GL_APIENTRY *)(GLuint, GLenum, GLint *);
using PFNGLGETPROGRAMINFOLOGPROC = void (GL_APIENTRY *)(GLuint, GLsizei, GLsizei *, GLchar *);
using PFNGLDELETEPROGRAMPROC = void (GL_APIENTRY *)(GLuint);
using PFNGLUSEPROGRAMPROC = void (GL_APIENTRY *)(GLuint);
using PFNGLGETUNIFORMBLOCKINDEXPROC = GLuint (GL_APIENTRY *)(GLuint, const GLchar *);
using PFNGLUNIFORMBLOCKBINDINGPROC = void (GL_APIENTRY *)(GLuint, GLuint, GLuint);
using PFNGLBINDBUFFERBASEPROC = void (GL_APIENTRY *)(GLenum, GLuint, GLuint);
using PFNGLGETACTIVEUNIFORMBLOCKIVPROC =
    void (GL_APIENTRY *)(GLuint, GLuint, GLenum, GLint *);
using PFNGLDRAWELEMENTSPROC = void (GL_APIENTRY *)(GLenum, GLsizei, GLenum, const void *);
using PFNGLREADPIXELSPROC =
    void (GL_APIENTRY *)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *);
using PFNGLPIXELSTOREIPROC = void (GL_APIENTRY *)(GLenum, GLint);

namespace demo {

#define DEMO_GL_FUNCTION_LIST(X) \
    X(PFNGLGETSTRINGPROC, glGetString) \
    X(PFNGLGETERRORPROC, glGetError) \
    X(PFNGLCLEARPROC, glClear) \
    X(PFNGLCLEARCOLORPROC, glClearColor) \
    X(PFNGLENABLEPROC, glEnable) \
    X(PFNGLDISABLEPROC, glDisable) \
    X(PFNGLVIEWPORTPROC, glViewport) \
    X(PFNGLDEPTHFUNCPROC, glDepthFunc) \
    X(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays) \
    X(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray) \
    X(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays) \
    X(PFNGLGENBUFFERSPROC, glGenBuffers) \
    X(PFNGLBINDBUFFERPROC, glBindBuffer) \
    X(PFNGLBUFFERDATAPROC, glBufferData) \
    X(PFNGLBUFFERSUBDATAPROC, glBufferSubData) \
    X(PFNGLGETBUFFERSUBDATAPROC, glGetBufferSubData) \
    X(PFNGLDELETEBUFFERSPROC, glDeleteBuffers) \
    X(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray) \
    X(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer) \
    X(PFNGLCREATESHADERPROC, glCreateShader) \
    X(PFNGLSHADERSOURCEPROC, glShaderSource) \
    X(PFNGLCOMPILESHADERPROC, glCompileShader) \
    X(PFNGLGETSHADERIVPROC, glGetShaderiv) \
    X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog) \
    X(PFNGLDELETESHADERPROC, glDeleteShader) \
    X(PFNGLCREATEPROGRAMPROC, glCreateProgram) \
    X(PFNGLATTACHSHADERPROC, glAttachShader) \
    X(PFNGLLINKPROGRAMPROC, glLinkProgram) \
    X(PFNGLGETPROGRAMIVPROC, glGetProgramiv) \
    X(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog) \
    X(PFNGLDELETEPROGRAMPROC, glDeleteProgram) \
    X(PFNGLUSEPROGRAMPROC, glUseProgram) \
    X(PFNGLGETUNIFORMBLOCKINDEXPROC, glGetUniformBlockIndex) \
    X(PFNGLUNIFORMBLOCKBINDINGPROC, glUniformBlockBinding) \
    X(PFNGLBINDBUFFERBASEPROC, glBindBufferBase) \
    X(PFNGLGETACTIVEUNIFORMBLOCKIVPROC, glGetActiveUniformBlockiv) \
    X(PFNGLDRAWELEMENTSPROC, glDrawElements) \
    X(PFNGLREADPIXELSPROC, glReadPixels) \
    X(PFNGLPIXELSTOREIPROC, glPixelStorei)

#define DEMO_DECLARE_GL_FUNCTION(type, name) extern type name;
DEMO_GL_FUNCTION_LIST(DEMO_DECLARE_GL_FUNCTION)
#undef DEMO_DECLARE_GL_FUNCTION

constexpr GLenum GL_NO_ERROR = 0x0000;
constexpr GLenum GL_INVALID_ENUM = 0x0500;
constexpr GLenum GL_INVALID_VALUE = 0x0501;
constexpr GLenum GL_INVALID_OPERATION = 0x0502;
constexpr GLenum GL_OUT_OF_MEMORY = 0x0505;
constexpr GLenum GL_INVALID_FRAMEBUFFER_OPERATION = 0x0506;
constexpr GLenum GL_UNIFORM_BLOCK_BINDING = 0x8A3F;
constexpr GLenum GL_UNIFORM_BLOCK_DATA_SIZE = 0x8A40;

constexpr GLbitfield GL_DEPTH_BUFFER_BIT = 0x00000100;
constexpr GLbitfield GL_COLOR_BUFFER_BIT = 0x00004000;

constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_LESS = 0x0201;
constexpr GLenum GL_MULTISAMPLE = 0x809D;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_ELEMENT_ARRAY_BUFFER = 0x8893;
constexpr GLenum GL_UNIFORM_BUFFER = 0x8A11;
constexpr GLenum GL_STATIC_DRAW = 0x88E4;
constexpr GLenum GL_DYNAMIC_DRAW = 0x88E8;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_UNSIGNED_INT = 0x1405;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
constexpr GLenum GL_RGB = 0x1907;
constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
constexpr GLenum GL_UNPACK_ALIGNMENT = 0x0CF5;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_RENDERER = 0x1F01;
constexpr GLenum GL_SHADING_LANGUAGE_VERSION = 0x8B8C;
constexpr GLboolean GL_FALSE = 0;
constexpr GLboolean GL_TRUE = 1;
constexpr GLuint GL_INVALID_INDEX = 0xFFFFFFFFu;

bool loadOpenGlFunctions(std::string &missingFunctions);
const char *glErrorName(GLenum error);

} // namespace demo
