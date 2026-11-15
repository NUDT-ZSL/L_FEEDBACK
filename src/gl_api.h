#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
#define SUBDIV_APIENTRY __stdcall
#else
#define SUBDIV_APIENTRY
#endif

namespace subdiv {
namespace gl {

using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLbyte = signed char;
using GLshort = short;
using GLint = int;
using GLsizei = int;
using GLubyte = unsigned char;
using GLushort = unsigned short;
using GLuint = unsigned int;
using GLfloat = float;
using GLclampf = float;
using GLdouble = double;
using GLchar = char;
using GLsizeiptr = std::ptrdiff_t;
using GLintptr = std::ptrdiff_t;

constexpr GLenum GL_FALSE = 0;
constexpr GLenum GL_TRUE = 1;
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_LINES = 0x0001;
constexpr GLenum GL_UNSIGNED_INT = 0x1405;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_LEQUAL = 0x0203;
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_BACK = 0x0405;
constexpr GLenum GL_CCW = 0x0901;
constexpr GLenum GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr GLenum GL_DEPTH_BUFFER_BIT = 0x00000100;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_ELEMENT_ARRAY_BUFFER = 0x8893;
constexpr GLenum GL_STATIC_DRAW = 0x88E4;
constexpr GLenum GL_LINE = 0x1B01;
constexpr GLenum GL_FILL = 0x1B02;
constexpr GLenum GL_FRONT_AND_BACK = 0x0408;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_RENDERER = 0x1F01;
constexpr GLenum GL_VENDOR = 0x1F00;
constexpr GLenum GL_SHADING_LANGUAGE_VERSION = 0x8B8C;
constexpr GLenum GL_MULTISAMPLE = 0x809D;
constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_INVALID_ENUM = 0x0500;
constexpr GLenum GL_INVALID_VALUE = 0x0501;
constexpr GLenum GL_INVALID_OPERATION = 0x0502;
constexpr GLenum GL_OUT_OF_MEMORY = 0x0505;
constexpr GLenum GL_INVALID_FRAMEBUFFER_OPERATION = 0x0506;

using PFNGLCLEARCOLORPROC = void (SUBDIV_APIENTRY*)(GLclampf, GLclampf, GLclampf, GLclampf);
using PFNGLCLEARPROC = void (SUBDIV_APIENTRY*)(GLbitfield);
using PFNGLVIEWPORTPROC = void (SUBDIV_APIENTRY*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLENABLEPROC = void (SUBDIV_APIENTRY*)(GLenum);
using PFNGLDISABLEPROC = void (SUBDIV_APIENTRY*)(GLenum);
using PFNGLDEPTHFUNCPROC = void (SUBDIV_APIENTRY*)(GLenum);
using PFNGLCULLFACEPROC = void (SUBDIV_APIENTRY*)(GLenum);
using PFNGLFRONTFACEPROC = void (SUBDIV_APIENTRY*)(GLenum);
using PFNGLPOLYGONMODEPROC = void (SUBDIV_APIENTRY*)(GLenum, GLenum);
using PFNGLGETERRORPROC = GLenum (SUBDIV_APIENTRY*)(void);
using PFNGLGETSTRINGPROC = const GLubyte* (SUBDIV_APIENTRY*)(GLenum);
using PFNGLPIXELSTOREIPROC = void (SUBDIV_APIENTRY*)(GLenum, GLint);
using PFNGLREADPIXELSPROC = void (SUBDIV_APIENTRY*)(GLint, GLint, GLsizei, GLsizei, GLenum,
                                                   GLenum, void*);
using PFNGLFINISHPROC = void (SUBDIV_APIENTRY*)(void);

using PFNGLGENVERTEXARRAYSPROC = void (SUBDIV_APIENTRY*)(GLsizei, GLuint*);
using PFNGLBINDVERTEXARRAYPROC = void (SUBDIV_APIENTRY*)(GLuint);
using PFNGLDELETEVERTEXARRAYSPROC = void (SUBDIV_APIENTRY*)(GLsizei, const GLuint*);
using PFNGLGENBUFFERSPROC = void (SUBDIV_APIENTRY*)(GLsizei, GLuint*);
using PFNGLBINDBUFFERPROC = void (SUBDIV_APIENTRY*)(GLenum, GLuint);
using PFNGLBUFFERDATAPROC = void (SUBDIV_APIENTRY*)(GLenum, GLsizeiptr, const void*, GLenum);
using PFNGLDELETEBUFFERSPROC = void (SUBDIV_APIENTRY*)(GLsizei, const GLuint*);
using PFNGLENABLEVERTEXATTRIBARRAYPROC = void (SUBDIV_APIENTRY*)(GLuint);
using PFNGLVERTEXATTRIBPOINTERPROC = void (SUBDIV_APIENTRY*)(GLuint, GLint, GLenum, GLboolean,
                                                             GLsizei, const void*);

using PFNGLCREATESHADERPROC = GLuint (SUBDIV_APIENTRY*)(GLenum);
using PFNGLSHADERSOURCEPROC = void (SUBDIV_APIENTRY*)(GLuint, GLsizei, const GLchar* const*,
                                                      const GLint*);
using PFNGLCOMPILESHADERPROC = void (SUBDIV_APIENTRY*)(GLuint);
using PFNGLGETSHADERIVPROC = void (SUBDIV_APIENTRY*)(GLuint, GLenum, GLint*);
using PFNGLGETSHADERINFOLOGPROC = void (SUBDIV_APIENTRY*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFNGLDELETESHADERPROC = void (SUBDIV_APIENTRY*)(GLuint);
using PFNGLCREATEPROGRAMPROC = GLuint (SUBDIV_APIENTRY*)(void);
using PFNGLATTACHSHADERPROC = void (SUBDIV_APIENTRY*)(GLuint, GLuint);
using PFNGLLINKPROGRAMPROC = void (SUBDIV_APIENTRY*)(GLuint);
using PFNGLGETPROGRAMIVPROC = void (SUBDIV_APIENTRY*)(GLuint, GLenum, GLint*);
using PFNGLGETPROGRAMINFOLOGPROC = void (SUBDIV_APIENTRY*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFNGLDELETEPROGRAMPROC = void (SUBDIV_APIENTRY*)(GLuint);
using PFNGLUSEPROGRAMPROC = void (SUBDIV_APIENTRY*)(GLuint);
using PFNGLGETUNIFORMLOCATIONPROC = GLint (SUBDIV_APIENTRY*)(GLuint, const GLchar*);
using PFNGLUNIFORM1FPROC = void (SUBDIV_APIENTRY*)(GLint, GLfloat);
using PFNGLUNIFORM1IPROC = void (SUBDIV_APIENTRY*)(GLint, GLint);
using PFNGLUNIFORM3FPROC = void (SUBDIV_APIENTRY*)(GLint, GLfloat, GLfloat, GLfloat);
using PFNGLUNIFORMMATRIX4FVPROC = void (SUBDIV_APIENTRY*)(GLint, GLsizei, GLboolean,
                                                          const GLfloat*);
using PFNGLDRAWELEMENTSPROC = void (SUBDIV_APIENTRY*)(GLenum, GLsizei, GLenum, const void*);

extern PFNGLCLEARCOLORPROC ClearColor;
extern PFNGLCLEARPROC Clear;
extern PFNGLVIEWPORTPROC Viewport;
extern PFNGLENABLEPROC Enable;
extern PFNGLDISABLEPROC Disable;
extern PFNGLDEPTHFUNCPROC DepthFunc;
extern PFNGLCULLFACEPROC CullFace;
extern PFNGLFRONTFACEPROC FrontFace;
extern PFNGLPOLYGONMODEPROC PolygonMode;
extern PFNGLGETERRORPROC GetError;
extern PFNGLGETSTRINGPROC GetString;
extern PFNGLPIXELSTOREIPROC PixelStorei;
extern PFNGLREADPIXELSPROC ReadPixels;
extern PFNGLFINISHPROC Finish;

extern PFNGLGENVERTEXARRAYSPROC GenVertexArrays;
extern PFNGLBINDVERTEXARRAYPROC BindVertexArray;
extern PFNGLDELETEVERTEXARRAYSPROC DeleteVertexArrays;
extern PFNGLGENBUFFERSPROC GenBuffers;
extern PFNGLBINDBUFFERPROC BindBuffer;
extern PFNGLBUFFERDATAPROC BufferData;
extern PFNGLDELETEBUFFERSPROC DeleteBuffers;
extern PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray;
extern PFNGLVERTEXATTRIBPOINTERPROC VertexAttribPointer;

extern PFNGLCREATESHADERPROC CreateShader;
extern PFNGLSHADERSOURCEPROC ShaderSource;
extern PFNGLCOMPILESHADERPROC CompileShader;
extern PFNGLGETSHADERIVPROC GetShaderiv;
extern PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog;
extern PFNGLDELETESHADERPROC DeleteShader;
extern PFNGLCREATEPROGRAMPROC CreateProgram;
extern PFNGLATTACHSHADERPROC AttachShader;
extern PFNGLLINKPROGRAMPROC LinkProgram;
extern PFNGLGETPROGRAMIVPROC GetProgramiv;
extern PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog;
extern PFNGLDELETEPROGRAMPROC DeleteProgram;
extern PFNGLUSEPROGRAMPROC UseProgram;
extern PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation;
extern PFNGLUNIFORM1FPROC Uniform1f;
extern PFNGLUNIFORM1IPROC Uniform1i;
extern PFNGLUNIFORM3FPROC Uniform3f;
extern PFNGLUNIFORMMATRIX4FVPROC UniformMatrix4fv;
extern PFNGLDRAWELEMENTSPROC DrawElements;

bool loadFunctions(std::vector<std::string>& missingFunctions);
const char* errorName(GLenum error);

}  // namespace gl
}  // namespace subdiv

