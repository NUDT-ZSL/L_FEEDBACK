#pragma once

#include <cstddef>
#include <functional>

#ifdef _WIN32
#define CR_GLAPIENTRY __stdcall
#else
#define CR_GLAPIENTRY
#endif

namespace cr::gl {

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
using GLintptr = std::ptrdiff_t;
using GLsizeiptr = std::ptrdiff_t;

using GlProc = void (*)();
using GlProcLoader = std::function<GlProc(const char*)>;

inline constexpr GLenum GL_NO_ERROR = 0;
inline constexpr GLenum GL_VERSION = 0x1F02;
inline constexpr GLenum GL_MAJOR_VERSION = 0x821B;
inline constexpr GLenum GL_MINOR_VERSION = 0x821C;
inline constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
inline constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
inline constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
inline constexpr GLenum GL_LINK_STATUS = 0x8B82;
inline constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
inline constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
inline constexpr GLenum GL_TEXTURE_2D_ARRAY = 0x8C1A;
inline constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
inline constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
inline constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
inline constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
inline constexpr GLenum GL_TEXTURE_WRAP_R = 0x8072;
inline constexpr GLenum GL_LINEAR = 0x2601;
inline constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;
inline constexpr GLenum GL_RGBA8 = 0x8058;
inline constexpr GLenum GL_RGBA16F = 0x881A;
inline constexpr GLenum GL_R32F = 0x822E;
inline constexpr GLenum GL_RGBA = 0x1908;
inline constexpr GLenum GL_RED = 0x1903;
inline constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
inline constexpr GLenum GL_FLOAT = 0x1406;
inline constexpr GLenum GL_TEXTURE0 = 0x84C0;
inline constexpr GLenum GL_FRAMEBUFFER = 0x8D40;
inline constexpr GLenum GL_COLOR_ATTACHMENT0 = 0x8CE0;
inline constexpr GLenum GL_COLOR_ATTACHMENT1 = 0x8CE1;
inline constexpr GLenum GL_COLOR_ATTACHMENT2 = 0x8CE2;
inline constexpr GLenum GL_COLOR_ATTACHMENT3 = 0x8CE3;
inline constexpr GLenum GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
inline constexpr GLenum GL_RENDERBUFFER = 0x8D41;
inline constexpr GLenum GL_DEPTH_ATTACHMENT = 0x8D00;
inline constexpr GLenum GL_DEPTH_COMPONENT24 = 0x81A6;
inline constexpr GLenum GL_DEPTH_TEST = 0x0B71;
inline constexpr GLenum GL_ALWAYS = 0x0207;
inline constexpr GLenum GL_BLEND = 0x0BE2;
inline constexpr GLenum GL_CULL_FACE = 0x0B44;
inline constexpr GLenum GL_COLOR_BUFFER_BIT = 0x00004000;
inline constexpr GLenum GL_DEPTH_BUFFER_BIT = 0x00000100;
inline constexpr GLenum GL_TRIANGLES = 0x0004;
inline constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
inline constexpr GLenum GL_UNPACK_ALIGNMENT = 0x0CF5;

using PFNGLGETERRORPROC = GLenum(CR_GLAPIENTRY*)();
using PFNGLGETINTEGERVPROC = void(CR_GLAPIENTRY*)(GLenum, GLint*);
using PFNGLVIEWPORTPROC = void(CR_GLAPIENTRY*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLCLEARCOLORPROC = void(CR_GLAPIENTRY*)(GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLCLEARPROC = void(CR_GLAPIENTRY*)(GLbitfield);
using PFNGLENABLEPROC = void(CR_GLAPIENTRY*)(GLenum);
using PFNGLDISABLEPROC = void(CR_GLAPIENTRY*)(GLenum);
using PFNGLDEPTHFUNCPROC = void(CR_GLAPIENTRY*)(GLenum);
using PFNALGORITHMPROC =
    void(CR_GLAPIENTRY*)(GLenum, GLenum, GLenum, GLenum, GLenum, GLenum, GLenum);
using PFNGLGENTEXTURESPROC = void(CR_GLAPIENTRY*)(GLsizei, GLuint*);
using PFNGLDELETETEXTURESPROC = void(CR_GLAPIENTRY*)(GLsizei, const GLuint*);
using PFNGLBINDTEXTUREPROC = void(CR_GLAPIENTRY*)(GLenum, GLuint);
using PFNGLACTIVETEXTUREPROC = void(CR_GLAPIENTRY*)(GLenum);
using PFNGLTEXIMAGE2DPROC = void(CR_GLAPIENTRY*)(GLenum, GLint, GLint, GLsizei, GLsizei,
                                                 GLint, GLenum, GLenum, const void*);
using PFNGLTEXIMAGE3DPROC = void(CR_GLAPIENTRY*)(GLenum, GLint, GLint, GLsizei, GLsizei,
                                                 GLsizei, GLint, GLenum, GLenum,
                                                 const void*);
using PFNGLTEXSUBIMAGE2DPROC = void(CR_GLAPIENTRY*)(GLenum, GLint, GLint, GLint, GLsizei,
                                                    GLsizei, GLenum, GLenum, const void*);
using PFNGLTEXSUBIMAGE3DPROC = void(CR_GLAPIENTRY*)(GLenum, GLint, GLint, GLint, GLint,
                                                    GLsizei, GLsizei, GLsizei, GLenum,
                                                    GLenum, const void*);
using PFNGLTEXPARAMETERIPROC = void(CR_GLAPIENTRY*)(GLenum, GLenum, GLint);
using PFNGLGENFRAMEBUFFERSPROC = void(CR_GLAPIENTRY*)(GLsizei, GLuint*);
using PFNGLBINDFRAMEBUFFERPROC = void(CR_GLAPIENTRY*)(GLenum, GLuint);
using PFNGLDELETEFRAMEBUFFERSPROC = void(CR_GLAPIENTRY*)(GLsizei, const GLuint*);
using PFNGLFRAMEBUFFERTEXTURE2DPROC = void(CR_GLAPIENTRY*)(GLenum, GLenum, GLenum, GLuint,
                                                          GLint);
using PFNGLDRAWBUFFERSPROC = void(CR_GLAPIENTRY*)(GLsizei, const GLenum*);
using PFNGLCHECKFRAMEBUFFERSTATUSPROC = GLenum(CR_GLAPIENTRY*)(GLenum);
using PFNGLGENRENDERBUFFERSPROC = void(CR_GLAPIENTRY*)(GLsizei, GLuint*);
using PFNGLBINDRENDERBUFFERPROC = void(CR_GLAPIENTRY*)(GLenum, GLuint);
using PFNGLDELETERENDERBUFFERSPROC = void(CR_GLAPIENTRY*)(GLsizei, const GLuint*);
using PFNGLRENDERBUFFERSTORAGEPROC = void(CR_GLAPIENTRY*)(GLenum, GLenum, GLsizei, GLsizei);
using PFNGLFRAMEBUFFERRENDERBUFFERPROC = void(CR_GLAPIENTRY*)(GLenum, GLenum, GLenum,
                                                             GLuint);
using PFNGLCREATESHADERPROC = GLuint(CR_GLAPIENTRY*)(GLenum);
using PFNGLSHADERSOURCEPROC = void(CR_GLAPIENTRY*)(GLuint, GLsizei, const GLchar* const*,
                                                   const GLint*);
using PFNGLCOMPILESHADERPROC = void(CR_GLAPIENTRY*)(GLuint);
using PFNGLGETSHADERIVPROC = void(CR_GLAPIENTRY*)(GLuint, GLenum, GLint*);
using PFNGLGETSHADERINFOLOGPROC = void(CR_GLAPIENTRY*)(GLuint, GLsizei, GLsizei*,
                                                       GLchar*);
using PFNGLDELETESHADERPROC = void(CR_GLAPIENTRY*)(GLuint);
using PFNGLCREATEPROGRAMPROC = GLuint(CR_GLAPIENTRY*)();
using PFNGLATTACHSHADERPROC = void(CR_GLAPIENTRY*)(GLuint, GLuint);
using PFNGLLINKPROGRAMPROC = void(CR_GLAPIENTRY*)(GLuint);
using PFNGLGETPROGRAMIVPROC = void(CR_GLAPIENTRY*)(GLuint, GLenum, GLint*);
using PFNGLGETPROGRAMINFOLOGPROC = void(CR_GLAPIENTRY*)(GLuint, GLsizei, GLsizei*,
                                                        GLchar*);
using PFNGLDELETEPROGRAMPROC = void(CR_GLAPIENTRY*)(GLuint);
using PFNGLUSEPROGRAMPROC = void(CR_GLAPIENTRY*)(GLuint);
using PFNGLGETUNIFORMLOCATIONPROC = GLint(CR_GLAPIENTRY*)(GLuint, const GLchar*);
using PFNGLUNIFORM1IPROC = void(CR_GLAPIENTRY*)(GLint, GLint);
using PFNGLUNIFORM1FPROC = void(CR_GLAPIENTRY*)(GLint, GLfloat);
using PFNGLUNIFORM2FPROC = void(CR_GLAPIENTRY*)(GLint, GLfloat, GLfloat);
using PFNGLUNIFORM3FPROC = void(CR_GLAPIENTRY*)(GLint, GLfloat, GLfloat, GLfloat);
using PFNGLUNIFORM3FVPROC = void(CR_GLAPIENTRY*)(GLint, GLsizei, const GLfloat*);
using PFNGLUNIFORM1FVPROC = void(CR_GLAPIENTRY*)(GLint, GLsizei, const GLfloat*);
using PFNGLUNIFORM1IVPROC = void(CR_GLAPIENTRY*)(GLint, GLsizei, const GLint*);
using PFNGLGENVERTEXARRAYSPROC = void(CR_GLAPIENTRY*)(GLsizei, GLuint*);
using PFNGLBINDVERTEXARRAYPROC = void(CR_GLAPIENTRY*)(GLuint);
using PFNGLDELETEVERTEXARRAYSPROC = void(CR_GLAPIENTRY*)(GLsizei, const GLuint*);
using PFNGLDRAWARRAYSPROC = void(CR_GLAPIENTRY*)(GLenum, GLint, GLsizei);
using PFNGLREADPIXELSPROC = void(CR_GLAPIENTRY*)(GLint, GLint, GLsizei, GLsizei, GLenum,
                                                GLenum, void*);
using PFNGLFINISHPROC = void(CR_GLAPIENTRY*)();
using PFNGLPIXELSTOREIPROC = void(CR_GLAPIENTRY*)(GLenum, GLint);

extern PFNGLGETERRORPROC glGetError;
extern PFNGLGETINTEGERVPROC glGetIntegerv;
extern PFNGLVIEWPORTPROC glViewport;
extern PFNGLCLEARCOLORPROC glClearColor;
extern PFNGLCLEARPROC glClear;
extern PFNGLENABLEPROC glEnable;
extern PFNGLDISABLEPROC glDisable;
extern PFNGLDEPTHFUNCPROC glDepthFunc;
extern PFNGLGENTEXTURESPROC glGenTextures;
extern PFNGLDELETETEXTURESPROC glDeleteTextures;
extern PFNGLBINDTEXTUREPROC glBindTexture;
extern PFNGLACTIVETEXTUREPROC glActiveTexture;
extern PFNGLTEXIMAGE2DPROC glTexImage2D;
extern PFNGLTEXIMAGE3DPROC glTexImage3D;
extern PFNGLTEXSUBIMAGE2DPROC glTexSubImage2D;
extern PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D;
extern PFNGLTEXPARAMETERIPROC glTexParameteri;
extern PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
extern PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;
extern PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;
extern PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D;
extern PFNGLDRAWBUFFERSPROC glDrawBuffers;
extern PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus;
extern PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers;
extern PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer;
extern PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers;
extern PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage;
extern PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer;
extern PFNGLCREATESHADERPROC glCreateShader;
extern PFNGLSHADERSOURCEPROC glShaderSource;
extern PFNGLCOMPILESHADERPROC glCompileShader;
extern PFNGLGETSHADERIVPROC glGetShaderiv;
extern PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
extern PFNGLDELETESHADERPROC glDeleteShader;
extern PFNGLCREATEPROGRAMPROC glCreateProgram;
extern PFNGLATTACHSHADERPROC glAttachShader;
extern PFNGLLINKPROGRAMPROC glLinkProgram;
extern PFNGLGETPROGRAMIVPROC glGetProgramiv;
extern PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
extern PFNGLDELETEPROGRAMPROC glDeleteProgram;
extern PFNGLUSEPROGRAMPROC glUseProgram;
extern PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
extern PFNGLUNIFORM1IPROC glUniform1i;
extern PFNGLUNIFORM1FPROC glUniform1f;
extern PFNGLUNIFORM2FPROC glUniform2f;
extern PFNGLUNIFORM3FPROC glUniform3f;
extern PFNGLUNIFORM3FVPROC glUniform3fv;
extern PFNGLUNIFORM1FVPROC glUniform1fv;
extern PFNGLUNIFORM1IVPROC glUniform1iv;
extern PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
extern PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
extern PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays;
extern PFNGLDRAWARRAYSPROC glDrawArrays;
extern PFNGLREADPIXELSPROC glReadPixels;
extern PFNGLFINISHPROC glFinish;
extern PFNGLPIXELSTOREIPROC glPixelStorei;

void loadFunctions(const GlProcLoader& loader);

}  // namespace cr::gl

