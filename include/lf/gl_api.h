#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gl {

using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLbyte = signed char;
using GLubyte = unsigned char;
using GLshort = short;
using GLint = int;
using GLsizei = int;
using GLuint = unsigned int;
using GLfloat = float;
using GLdouble = double;
using GLchar = char;
using GLintptr = std::ptrdiff_t;
using GLsizeiptr = std::ptrdiff_t;

constexpr GLenum GL_FALSE = 0;
constexpr GLenum GL_TRUE = 1;
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_UNSIGNED_INT = 0x1405;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_LESS = 0x0201;
constexpr GLenum GL_BLEND = 0x0BE2;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_BACK = 0x0405;
constexpr GLenum GL_CCW = 0x0901;
constexpr GLenum GL_SCISSOR_TEST = 0x0C11;
constexpr GLenum GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr GLenum GL_DEPTH_BUFFER_BIT = 0x00000100;
constexpr GLenum GL_SRC_ALPHA = 0x0302;
constexpr GLenum GL_ONE_MINUS_SRC_ALPHA = 0x0303;
constexpr GLenum GL_TEXTURE0 = 0x84C0;
constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
constexpr GLenum GL_LINEAR = 0x2601;
constexpr GLenum GL_NEAREST = 0x2600;
constexpr GLenum GL_LINEAR_MIPMAP_LINEAR = 0x2703;
constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;
constexpr GLenum GL_REPEAT = 0x2901;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_RG = 0x8227;
constexpr GLenum GL_RGBA8 = 0x8058;
constexpr GLenum GL_RGBA16F = 0x881A;
constexpr GLenum GL_RGBA32F = 0x8814;
constexpr GLenum GL_RG16F = 0x822F;
constexpr GLenum GL_RED = 0x1903;
constexpr GLenum GL_DEPTH_COMPONENT = 0x1902;
constexpr GLenum GL_DEPTH_COMPONENT24 = 0x81A6;
constexpr GLenum GL_FRAMEBUFFER = 0x8D40;
constexpr GLenum GL_RENDERBUFFER = 0x8D41;
constexpr GLenum GL_COLOR_ATTACHMENT0 = 0x8CE0;
constexpr GLenum GL_COLOR_ATTACHMENT1 = 0x8CE1;
constexpr GLenum GL_COLOR_ATTACHMENT2 = 0x8CE2;
constexpr GLenum GL_COLOR_ATTACHMENT3 = 0x8CE3;
constexpr GLenum GL_COLOR_ATTACHMENT4 = 0x8CE4;
constexpr GLenum GL_DEPTH_ATTACHMENT = 0x8D00;
constexpr GLenum GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
constexpr GLenum GL_FRAMEBUFFER_SRGB = 0x8DB9;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_ELEMENT_ARRAY_BUFFER = 0x8893;
constexpr GLenum GL_STATIC_DRAW = 0x88E4;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
constexpr GLenum GL_UNPACK_ALIGNMENT = 0x0CF5;
constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_SHADING_LANGUAGE_VERSION = 0x8B8C;
constexpr GLenum GL_RENDERER = 0x1F01;
constexpr GLenum GL_VENDOR = 0x1F00;
constexpr GLenum GL_VIEWPORT = 0x0BA2;
constexpr GLenum GL_MAJOR_VERSION = 0x821B;
constexpr GLenum GL_MINOR_VERSION = 0x821C;

struct Api {
  void (*Viewport)(GLint, GLint, GLsizei, GLsizei) = nullptr;
  void (*ClearColor)(GLfloat, GLfloat, GLfloat, GLfloat) = nullptr;
  void (*Clear)(GLbitfield) = nullptr;
  void (*Enable)(GLenum) = nullptr;
  void (*Disable)(GLenum) = nullptr;
  void (*DepthFunc)(GLenum) = nullptr;
  void (*DepthMask)(GLboolean) = nullptr;
  void (*CullFace)(GLenum) = nullptr;
  void (*FrontFace)(GLenum) = nullptr;
  void (*BlendFunc)(GLenum, GLenum) = nullptr;
  void (*PixelStorei)(GLenum, GLint) = nullptr;
  void (*GetIntegerv)(GLenum, GLint*) = nullptr;
  GLenum (*GetError)() = nullptr;
  const GLubyte* (*GetString)(GLenum) = nullptr;
  void (*Finish)() = nullptr;

  GLuint (*CreateShader)(GLenum) = nullptr;
  void (*ShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*) = nullptr;
  void (*CompileShader)(GLuint) = nullptr;
  void (*GetShaderiv)(GLuint, GLenum, GLint*) = nullptr;
  void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*) = nullptr;
  void (*DeleteShader)(GLuint) = nullptr;
  GLuint (*CreateProgram)() = nullptr;
  void (*AttachShader)(GLuint, GLuint) = nullptr;
  void (*LinkProgram)(GLuint) = nullptr;
  void (*GetProgramiv)(GLuint, GLenum, GLint*) = nullptr;
  void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*) = nullptr;
  void (*UseProgram)(GLuint) = nullptr;
  void (*DeleteProgram)(GLuint) = nullptr;
  GLint (*GetUniformLocation)(GLuint, const GLchar*) = nullptr;
  void (*Uniform1i)(GLint, GLint) = nullptr;
  void (*Uniform1f)(GLint, GLfloat) = nullptr;
  void (*Uniform2f)(GLint, GLfloat, GLfloat) = nullptr;
  void (*Uniform3f)(GLint, GLfloat, GLfloat, GLfloat) = nullptr;
  void (*Uniform4f)(GLint, GLfloat, GLfloat, GLfloat, GLfloat) = nullptr;
  void (*Uniform1fv)(GLint, GLsizei, const GLfloat*) = nullptr;
  void (*Uniform1iv)(GLint, GLsizei, const GLint*) = nullptr;
  void (*Uniform3fv)(GLint, GLsizei, const GLfloat*) = nullptr;
  void (*Uniform4fv)(GLint, GLsizei, const GLfloat*) = nullptr;
  void (*UniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat*) = nullptr;
  void (*UniformMatrix3fv)(GLint, GLsizei, GLboolean, const GLfloat*) = nullptr;

  void (*GenVertexArrays)(GLsizei, GLuint*) = nullptr;
  void (*BindVertexArray)(GLuint) = nullptr;
  void (*DeleteVertexArrays)(GLsizei, const GLuint*) = nullptr;
  void (*GenBuffers)(GLsizei, GLuint*) = nullptr;
  void (*BindBuffer)(GLenum, GLuint) = nullptr;
  void (*BufferData)(GLenum, GLsizeiptr, const void*, GLenum) = nullptr;
  void (*DeleteBuffers)(GLsizei, const GLuint*) = nullptr;
  void (*EnableVertexAttribArray)(GLuint) = nullptr;
  void (*VertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*) = nullptr;
  void (*VertexAttribIPointer)(GLuint, GLint, GLenum, GLsizei, const void*) = nullptr;
  void (*DrawArrays)(GLenum, GLint, GLsizei) = nullptr;
  void (*DrawElements)(GLenum, GLsizei, GLenum, const void*) = nullptr;

  void (*GenTextures)(GLsizei, GLuint*) = nullptr;
  void (*BindTexture)(GLenum, GLuint) = nullptr;
  void (*ActiveTexture)(GLenum) = nullptr;
  void (*TexParameteri)(GLenum, GLenum, GLint) = nullptr;
  void (*TexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum,
                     const void*) = nullptr;
  void (*GenerateMipmap)(GLenum) = nullptr;
  void (*DeleteTextures)(GLsizei, const GLuint*) = nullptr;

  void (*GenFramebuffers)(GLsizei, GLuint*) = nullptr;
  void (*BindFramebuffer)(GLenum, GLuint) = nullptr;
  void (*DeleteFramebuffers)(GLsizei, const GLuint*) = nullptr;
  void (*FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint) = nullptr;
  void (*FramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint) = nullptr;
  GLenum (*CheckFramebufferStatus)(GLenum) = nullptr;
  void (*DrawBuffers)(GLsizei, const GLenum*) = nullptr;
  void (*GenRenderbuffers)(GLsizei, GLuint*) = nullptr;
  void (*BindRenderbuffer)(GLenum, GLuint) = nullptr;
  void (*RenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei) = nullptr;
  void (*DeleteRenderbuffers)(GLsizei, const GLuint*) = nullptr;

  void (*ReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*) = nullptr;
};

extern Api gl;

bool load_api(std::string& missingEntryPoint);

}  // namespace gl

