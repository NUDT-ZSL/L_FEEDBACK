#pragma once

#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif
#include <GL/gl.h>

#include <cstddef>

#if defined(_WIN32)
#define SKEL_GL_CALL APIENTRY
#else
#define SKEL_GL_CALL
#endif

#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_INFO_LOG_LENGTH
#define GL_INFO_LOG_LENGTH 0x8B84
#endif
#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_ELEMENT_ARRAY_BUFFER
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_UNSIGNED_INT
#define GL_UNSIGNED_INT 0x1405
#endif

namespace skel::gl4 {

using GenVertexArraysProc = void (SKEL_GL_CALL*)(GLsizei, GLuint*);
using BindVertexArrayProc = void (SKEL_GL_CALL*)(GLuint);
using DeleteVertexArraysProc = void (SKEL_GL_CALL*)(GLsizei,
                                                    const GLuint*);
using GenBuffersProc = void (SKEL_GL_CALL*)(GLsizei, GLuint*);
using BindBufferProc = void (SKEL_GL_CALL*)(GLenum, GLuint);
using BufferDataProc = void (SKEL_GL_CALL*)(GLenum, std::ptrdiff_t,
                                            const void*, GLenum);
using DeleteBuffersProc = void (SKEL_GL_CALL*)(GLsizei, const GLuint*);
using VertexAttribPointerProc = void (SKEL_GL_CALL*)(
    GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
using EnableVertexAttribArrayProc = void (SKEL_GL_CALL*)(GLuint);
using CreateShaderProc = GLuint (SKEL_GL_CALL*)(GLenum);
using ShaderSourceProc = void (SKEL_GL_CALL*)(
    GLuint, GLsizei, const char* const*, const GLint*);
using CompileShaderProc = void (SKEL_GL_CALL*)(GLuint);
using GetShaderivProc = void (SKEL_GL_CALL*)(GLuint, GLenum, GLint*);
using GetShaderInfoLogProc = void (SKEL_GL_CALL*)(GLuint, GLsizei, GLsizei*,
                                                  char*);
using DeleteShaderProc = void (SKEL_GL_CALL*)(GLuint);
using CreateProgramProc = GLuint (SKEL_GL_CALL*)(void);
using AttachShaderProc = void (SKEL_GL_CALL*)(GLuint, GLuint);
using LinkProgramProc = void (SKEL_GL_CALL*)(GLuint);
using GetProgramivProc = void (SKEL_GL_CALL*)(GLuint, GLenum, GLint*);
using GetProgramInfoLogProc = void (SKEL_GL_CALL*)(GLuint, GLsizei, GLsizei*,
                                                   char*);
using DeleteProgramProc = void (SKEL_GL_CALL*)(GLuint);
using UseProgramProc = void (SKEL_GL_CALL*)(GLuint);
using GetUniformLocationProc = GLint (SKEL_GL_CALL*)(GLuint, const char*);
using UniformMatrix4fvProc = void (SKEL_GL_CALL*)(GLint, GLsizei, GLboolean,
                                                 const GLfloat*);
using Uniform3fvProc = void (SKEL_GL_CALL*)(GLint, GLsizei,
                                            const GLfloat*);
using Uniform1iProc = void (SKEL_GL_CALL*)(GLint, GLint);

extern GenVertexArraysProc GenVertexArrays;
extern BindVertexArrayProc BindVertexArray;
extern DeleteVertexArraysProc DeleteVertexArrays;
extern GenBuffersProc GenBuffers;
extern BindBufferProc BindBuffer;
extern BufferDataProc BufferData;
extern DeleteBuffersProc DeleteBuffers;
extern VertexAttribPointerProc VertexAttribPointer;
extern EnableVertexAttribArrayProc EnableVertexAttribArray;
extern CreateShaderProc CreateShader;
extern ShaderSourceProc ShaderSource;
extern CompileShaderProc CompileShader;
extern GetShaderivProc GetShaderiv;
extern GetShaderInfoLogProc GetShaderInfoLog;
extern DeleteShaderProc DeleteShader;
extern CreateProgramProc CreateProgram;
extern AttachShaderProc AttachShader;
extern LinkProgramProc LinkProgram;
extern GetProgramivProc GetProgramiv;
extern GetProgramInfoLogProc GetProgramInfoLog;
extern DeleteProgramProc DeleteProgram;
extern UseProgramProc UseProgram;
extern GetUniformLocationProc GetUniformLocation;
extern UniformMatrix4fvProc UniformMatrix4fv;
extern Uniform3fvProc Uniform3fv;
extern Uniform1iProc Uniform1i;

bool load(std::string& diagnostic);

} // namespace skel::gl4

// The public OpenGL 1.x entry points are linked normally. These aliases keep
// all 4.1-specific calls routed through the local loader.
#define glGenVertexArrays skel::gl4::GenVertexArrays
#define glBindVertexArray skel::gl4::BindVertexArray
#define glDeleteVertexArrays skel::gl4::DeleteVertexArrays
#define glGenBuffers skel::gl4::GenBuffers
#define glBindBuffer skel::gl4::BindBuffer
#define glBufferData skel::gl4::BufferData
#define glDeleteBuffers skel::gl4::DeleteBuffers
#define glVertexAttribPointer skel::gl4::VertexAttribPointer
#define glEnableVertexAttribArray skel::gl4::EnableVertexAttribArray
#define glCreateShader skel::gl4::CreateShader
#define glShaderSource skel::gl4::ShaderSource
#define glCompileShader skel::gl4::CompileShader
#define glGetShaderiv skel::gl4::GetShaderiv
#define glGetShaderInfoLog skel::gl4::GetShaderInfoLog
#define glDeleteShader skel::gl4::DeleteShader
#define glCreateProgram skel::gl4::CreateProgram
#define glAttachShader skel::gl4::AttachShader
#define glLinkProgram skel::gl4::LinkProgram
#define glGetProgramiv skel::gl4::GetProgramiv
#define glGetProgramInfoLog skel::gl4::GetProgramInfoLog
#define glDeleteProgram skel::gl4::DeleteProgram
#define glUseProgram skel::gl4::UseProgram
#define glGetUniformLocation skel::gl4::GetUniformLocation
#define glUniformMatrix4fv skel::gl4::UniformMatrix4fv
#define glUniform3fv skel::gl4::Uniform3fv
#define glUniform1i skel::gl4::Uniform1i

