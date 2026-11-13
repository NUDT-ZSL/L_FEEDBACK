#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace terrain::gl {

using ProcAddress = void (*)(void);
using ProcResolver = ProcAddress (*)(const char* name);

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

constexpr GLenum GL_FALSE = 0;
constexpr GLenum GL_TRUE = 1;
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_LESS = 0x0201;
constexpr GLenum GL_LEQUAL = 0x0203;
constexpr GLenum GL_FILL = 0x1B02;
constexpr GLenum GL_LINE = 0x1B01;
constexpr GLenum GL_FRONT_AND_BACK = 0x0408;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_LINES = 0x0001;
constexpr GLenum GL_UNSIGNED_INT = 0x1405;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_RGB = 0x1907;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_RED = 0x1903;
constexpr GLenum GL_RGBA8 = 0x8058;
constexpr GLenum GL_R32F = 0x822E;
constexpr GLenum GL_DEPTH_COMPONENT24 = 0x81A6;
constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
constexpr GLenum GL_TEXTURE0 = 0x84C0;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
constexpr GLenum GL_NEAREST = 0x2600;
constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_ELEMENT_ARRAY_BUFFER = 0x8893;
constexpr GLenum GL_SHADER_STORAGE_BUFFER = 0x90D2;
constexpr GLenum GL_STATIC_DRAW = 0x88E4;
constexpr GLenum GL_DYNAMIC_DRAW = 0x88E8;
constexpr GLenum GL_FRAMEBUFFER = 0x8D40;
constexpr GLenum GL_READ_FRAMEBUFFER = 0x8CA8;
constexpr GLenum GL_DRAW_FRAMEBUFFER = 0x8CA9;
constexpr GLenum GL_RENDERBUFFER = 0x8D41;
constexpr GLenum GL_COLOR_ATTACHMENT0 = 0x8CE0;
constexpr GLenum GL_DEPTH_ATTACHMENT = 0x8D00;
constexpr GLenum GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
constexpr GLenum GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr GLenum GL_DEPTH_BUFFER_BIT = 0x00000100;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
constexpr GLenum GL_UNPACK_ALIGNMENT = 0x0CF5;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_RENDERER = 0x1F01;
constexpr GLenum GL_BACK = 0x0405;
constexpr GLenum GL_DEPTH_BUFFER_BIT_MASK = GL_DEPTH_BUFFER_BIT;

extern GLuint (*glCreateShader)(GLenum type);
extern void (*glShaderSource)(
    GLuint shader,
    GLsizei count,
    const GLchar* const* string,
    const GLint* length);
extern void (*glCompileShader)(GLuint shader);
extern void (*glGetShaderiv)(GLuint shader, GLenum pname, GLint* params);
extern void (*glGetShaderInfoLog)(
    GLuint shader,
    GLsizei bufferSize,
    GLsizei* length,
    GLchar* infoLog);
extern void (*glDeleteShader)(GLuint shader);
extern GLuint (*glCreateProgram)();
extern void (*glAttachShader)(GLuint program, GLuint shader);
extern void (*glLinkProgram)(GLuint program);
extern void (*glGetProgramiv)(GLuint program, GLenum pname, GLint* params);
extern void (*glGetProgramInfoLog)(
    GLuint program,
    GLsizei bufferSize,
    GLsizei* length,
    GLchar* infoLog);
extern void (*glDeleteProgram)(GLuint program);
extern void (*glUseProgram)(GLuint program);
extern GLint (*glGetUniformLocation)(GLuint program, const GLchar* name);
extern void (*glUniform1i)(GLint location, GLint value);
extern void (*glUniform1f)(GLint location, GLfloat value);
extern void (*glUniform3f)(
    GLint location,
    GLfloat value0,
    GLfloat value1,
    GLfloat value2);
extern void (*glUniformMatrix4fv)(
    GLint location,
    GLsizei count,
    GLboolean transpose,
    const GLfloat* value);
extern void (*glGenVertexArrays)(GLsizei count, GLuint* arrays);
extern void (*glBindVertexArray)(GLuint array);
extern void (*glDeleteVertexArrays)(GLsizei count, const GLuint* arrays);
extern void (*glGenBuffers)(GLsizei count, GLuint* buffers);
extern void (*glBindBuffer)(GLenum target, GLuint buffer);
extern void (*glBufferData)(
    GLenum target,
    GLsizeiptr size,
    const void* data,
    GLenum usage);
extern void (*glBufferSubData)(
    GLenum target,
    GLintptr offset,
    GLsizeiptr size,
    const void* data);
extern void (*glBindBufferBase)(
    GLenum target,
    GLuint index,
    GLuint buffer);
extern void (*glDeleteBuffers)(GLsizei count, const GLuint* buffers);
extern void (*glGenTextures)(GLsizei count, GLuint* textures);
extern void (*glBindTexture)(GLenum target, GLuint texture);
extern void (*glTexImage2D)(
    GLenum target,
    GLint level,
    GLint internalFormat,
    GLsizei width,
    GLsizei height,
    GLint border,
    GLenum format,
    GLenum type,
    const void* data);
extern void (*glTexParameteri)(GLenum target, GLenum pname, GLint parameter);
extern void (*glDeleteTextures)(GLsizei count, const GLuint* textures);
extern void (*glActiveTexture)(GLenum texture);
extern void (*glGenFramebuffers)(GLsizei count, GLuint* framebuffers);
extern void (*glBindFramebuffer)(GLenum target, GLuint framebuffer);
extern void (*glFramebufferTexture2D)(
    GLenum target,
    GLenum attachment,
    GLenum textureTarget,
    GLuint texture,
    GLint level);
extern GLenum (*glCheckFramebufferStatus)(GLenum target);
extern void (*glGenRenderbuffers)(GLsizei count, GLuint* renderbuffers);
extern void (*glBindRenderbuffer)(GLenum target, GLuint renderbuffer);
extern void (*glRenderbufferStorage)(
    GLenum target,
    GLenum internalFormat,
    GLsizei width,
    GLsizei height);
extern void (*glFramebufferRenderbuffer)(
    GLenum target,
    GLenum attachment,
    GLenum renderbufferTarget,
    GLuint renderbuffer);
extern void (*glDeleteFramebuffers)(GLsizei count, const GLuint* framebuffers);
extern void (*glDeleteRenderbuffers)(GLsizei count, const GLuint* renderbuffers);
extern void (*glDrawElementsInstanced)(
    GLenum mode,
    GLsizei count,
    GLenum type,
    const void* indices,
    GLsizei instanceCount);
extern void (*glDrawArrays)(GLenum mode, GLint first, GLsizei count);
extern void (*glEnableVertexAttribArray)(GLuint index);
extern void (*glDisableVertexAttribArray)(GLuint index);
extern void (*glVertexAttribPointer)(
    GLuint index,
    GLint size,
    GLenum type,
    GLboolean normalized,
    GLsizei stride,
    const void* pointer);
extern void (*glClear)(GLbitfield mask);
extern void (*glClearColor)(
    GLclampf red,
    GLclampf green,
    GLclampf blue,
    GLclampf alpha);
extern void (*glEnable)(GLenum capability);
extern void (*glDisable)(GLenum capability);
extern void (*glDepthFunc)(GLenum function);
extern void (*glDepthMask)(GLboolean flag);
extern void (*glViewport)(GLint x, GLint y, GLsizei width, GLsizei height);
extern void (*glPolygonMode)(GLenum face, GLenum mode);
extern void (*glLineWidth)(GLfloat width);
extern void (*glReadPixels)(
    GLint x,
    GLint y,
    GLsizei width,
    GLsizei height,
    GLenum format,
    GLenum type,
    void* pixels);
extern void (*glReadBuffer)(GLenum source);
extern void (*glPixelStorei)(GLenum pname, GLint parameter);
extern void (*glFinish)();
extern void (*glFlush)();
extern void (*glBlitFramebuffer)(
    GLint sourceX0,
    GLint sourceY0,
    GLint sourceX1,
    GLint sourceY1,
    GLint destinationX0,
    GLint destinationY0,
    GLint destinationX1,
    GLint destinationY1,
    GLbitfield mask,
    GLenum filter);
extern GLenum (*glGetError)();
extern const GLubyte* (*glGetString)(GLenum name);
extern void (*glGetIntegerv)(GLenum pname, GLint* data);

bool loadFunctions(std::string& error);
void setProcResolver(ProcResolver resolver);

}  // namespace terrain::gl

