#include "gl_renderer.hpp"

#include "gl_api.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace terrain {
namespace {

constexpr int kIndicesPerPatch = kPatchQuads * kPatchQuads * 6;
constexpr int kMaxEdgeSegments = 8;
constexpr int kEdges = 4;

struct EdgeSegmentGpu {
    std::array<float, 4> range{};
    std::array<float, 4> low{};
};

struct NodeGpu {
    std::array<float, 4> rect{};
    std::array<float, 4> bounds{};
    std::array<EdgeSegmentGpu, kEdges * kMaxEdgeSegments> edges{};
};

static_assert(sizeof(EdgeSegmentGpu) == 32);
static_assert(sizeof(NodeGpu) == 16 + 16 + 32 * kEdges * kMaxEdgeSegments);

const char* kTerrainVertexShader = R"GLSL(
#version 460 core

struct EdgeSegment {
    vec4 range;
    vec4 low;
};

struct NodeData {
    vec4 rect;
    vec4 bounds;
    EdgeSegment edges[4][8];
};

layout(std430, binding = 0) readonly buffer NodeBuffer {
    NodeData nodes[];
};

uniform sampler2D uHeightMap;
uniform float uExtent;
uniform int uGrid;
uniform int uNodeOffset;
uniform mat4 uViewProjection;

flat out int vDepth;
out vec3 vWorld;
out float vHeight;

float sampleHeight(vec2 position) {
    float maxCoordinate = uExtent;
    vec2 gridPosition = clamp(position / maxCoordinate, 0.0, 1.0) *
                        float(uGrid - 1);
    ivec2 i0 = ivec2(floor(gridPosition));
    ivec2 maxIndex = ivec2(uGrid - 1);
    ivec2 i1 = min(i0 + ivec2(1), maxIndex);
    vec2 fraction = gridPosition - vec2(i0);
    float h00 = texelFetch(uHeightMap, i0, 0).r;
    float h10 = texelFetch(uHeightMap, i1.x == i0.x ? i0 : ivec2(i1.x, i0.y), 0).r;
    float h01 = texelFetch(uHeightMap, i1.y == i0.y ? i0 : ivec2(i0.x, i1.y), 0).r;
    float h11 = texelFetch(uHeightMap, i1, 0).r;
    float hx0 = mix(h00, h10, fraction.x);
    float hx1 = mix(h01, h11, fraction.x);
    return mix(hx0, hx1, fraction.y);
}

void main() {
    int localIndex = gl_VertexID;
    int localX = localIndex % 17;
    int localZ = localIndex / 17;
    NodeData node = nodes[uNodeOffset + gl_InstanceID];

    float span = node.rect.z;
    float x = node.rect.x + span * float(localX) / 16.0;
    float z = node.rect.y + span * float(localZ) / 16.0;
    float height = sampleHeight(vec2(x, z));

    for (int edge = 0; edge < 4; ++edge) {
        bool onEdge = false;
        float t = 0.0;
        if (edge == 0) {
            onEdge = localZ == 0;
            t = float(localX) / 16.0;
        } else if (edge == 1) {
            onEdge = localX == 16;
            t = float(localZ) / 16.0;
        } else if (edge == 2) {
            onEdge = localZ == 16;
            t = float(localX) / 16.0;
        } else if (edge == 3) {
            onEdge = localX == 0;
            t = float(localZ) / 16.0;
        }
        if (!onEdge) {
            continue;
        }

        for (int segmentIndex = 0; segmentIndex < 8; ++segmentIndex) {
            EdgeSegment segment = node.edges[edge][segmentIndex];
            if (segment.range.z < 0.5) {
                continue;
            }
            if (t + 1.0e-6 < segment.range.x ||
                t - 1.0e-6 > segment.range.y) {
                continue;
            }
            float lowMinAlong = segment.low.x;
            float lowSpan = max(segment.low.y, 1.0e-6);
            float along = edge < 2 ? x : z;
            float edgeCoordinate = edge < 2 ? z : x;
            float local = clamp((along - lowMinAlong) / lowSpan * 16.0, 0.0, 16.0);
            float j0 = min(floor(local), 15.0);
            float fraction = local - j0;
            float a = lowMinAlong + j0 * lowSpan / 16.0;
            float b = lowMinAlong + (j0 + 1.0) * lowSpan / 16.0;
            float h0 = edge < 2
                           ? sampleHeight(vec2(a, edgeCoordinate))
                           : sampleHeight(vec2(edgeCoordinate, a));
            float h1 = edge < 2
                           ? sampleHeight(vec2(b, edgeCoordinate))
                           : sampleHeight(vec2(edgeCoordinate, b));
            height = mix(h0, h1, fraction);
        }
    }

    vWorld = vec3(x, height, z);
    vHeight = height;
    vDepth = int(node.rect.w + 0.5);
    gl_Position = uViewProjection * vec4(vWorld, 1.0);
}
)GLSL";

const char* kTerrainFragmentShader = R"GLSL(
#version 460 core

flat in int vDepth;
in vec3 vWorld;
in float vHeight;
uniform int uViewMode;
uniform float uAmplitude;
out vec4 fragColor;

vec3 palette(int depth, float shade) {
    vec3 colors[6] = vec3[6](
        vec3(0.98, 0.85, 0.25),
        vec3(0.98, 0.56, 0.18),
        vec3(0.90, 0.24, 0.22),
        vec3(0.68, 0.23, 0.73),
        vec3(0.20, 0.47, 0.93),
        vec3(0.15, 0.85, 0.84)
    );
    return colors[clamp(depth, 0, 5)] * shade;
}

void main() {
    vec3 normal = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (normal.y < 0.0) {
        normal = -normal;
    }

    if (uViewMode == 2) {
        fragColor = vec4(palette(vDepth, 0.70 + 0.30 * max(normal.y, 0.0)), 1.0);
        return;
    }

    float normalizedHeight =
        clamp(vHeight / max(uAmplitude, 1.0) * 0.75 + 0.5, 0.0, 1.0);
    vec3 base;
    if (normalizedHeight < 0.52) {
        base = vec3(0.18, 0.36, 0.19);
    } else if (normalizedHeight < 0.72) {
        base = vec3(0.44, 0.40, 0.27);
    } else if (normalizedHeight < 0.88) {
        base = vec3(0.48, 0.47, 0.45);
    } else {
        base = vec3(0.82, 0.86, 0.90);
    }

    vec3 lightDirection = normalize(vec3(-0.45, 0.78, -0.38));
    float diffuse = clamp(dot(normal, lightDirection), 0.0, 1.0);
    float illumination = 0.33 + 0.52 * diffuse + 0.15 * max(normal.y, 0.0);
    fragColor = vec4(base * illumination, 1.0);
}
)GLSL";

const char* kLineVertexShader = R"GLSL(
#version 460 core
layout(location = 0) in vec3 aPosition;
uniform mat4 uViewProjection;
uniform float uDepthBias;
void main() {
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
    gl_Position.z -= uDepthBias * gl_Position.w;
}
)GLSL";

const char* kLineFragmentShader = R"GLSL(
#version 460 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.09, 0.12, 1.0);
}
)GLSL";

unsigned int compileShader(terrain::gl::GLenum type, const char* source, std::string& error) {
    using namespace terrain::gl;
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    int status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == 0) {
        int length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(std::max(length, 1)));
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        error = "shader compilation failed: ";
        error.append(log.data());
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

unsigned int createProgram(
    const char* vertexSource,
    const char* fragmentSource,
    std::string& error) {
    using namespace terrain::gl;
    unsigned int vertex = compileShader(GL_VERTEX_SHADER, vertexSource, error);
    if (vertex == 0) {
        return 0;
    }
    unsigned int fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource, error);
    if (fragment == 0) {
        glDeleteShader(vertex);
        return 0;
    }
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    int status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == 0) {
        int length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(std::max(length, 1)));
        glGetProgramInfoLog(program, length, nullptr, log.data());
        error = "shader link failed: ";
        error.append(log.data());
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

}  // namespace

GlRenderer::GlRenderer(const Terrain& terrain, int width, int height)
    : terrain_(terrain),
      width_(std::max(width, 1)),
      height_(std::max(height, 1)),
      terrainVersion_(terrain.version()) {}

GlRenderer::~GlRenderer() {
    shutdown();
}

bool GlRenderer::initialize(std::string& error) {
    if (initialized_) {
        return true;
    }
    if (terrain_.version() != terrainVersion_) {
        error = "terrain version changed after renderer construction";
        return false;
    }

    std::string loaderError;
    if (!gl::loadFunctions(loaderError)) {
        error = loaderError;
        return false;
    }

    const auto* versionString =
        reinterpret_cast<const char*>(gl::glGetString(gl::GL_VERSION));
    int major = 0;
    int minor = 0;
    if (versionString == nullptr ||
        std::sscanf(versionString, "%d.%d", &major, &minor) != 2 ||
        major < 4 ||
        (major == 4 && minor < 6)) {
        error = "OpenGL 4.6 context is required";
        return false;
    }

    initialized_ = true;
    if (!createShaders(error) ||
        !createStaticBuffers(error) ||
        !createFramebuffer(error)) {
        shutdown();
        return false;
    }
    return true;
}

bool GlRenderer::createShaders(std::string& error) {
    terrainProgram_ = createProgram(
        kTerrainVertexShader,
        kTerrainFragmentShader,
        error);
    if (terrainProgram_ == 0) {
        return false;
    }
    lineProgram_ = createProgram(
        kLineVertexShader,
        kLineFragmentShader,
        error);
    if (lineProgram_ == 0) {
        return false;
    }

    viewProjectionLocation_ =
        gl::glGetUniformLocation(terrainProgram_, "uViewProjection");
    heightMapLocation_ = gl::glGetUniformLocation(terrainProgram_, "uHeightMap");
    extentLocation_ = gl::glGetUniformLocation(terrainProgram_, "uExtent");
    gridLocation_ = gl::glGetUniformLocation(terrainProgram_, "uGrid");
    nodeOffsetLocation_ = gl::glGetUniformLocation(terrainProgram_, "uNodeOffset");
    viewModeLocation_ = gl::glGetUniformLocation(terrainProgram_, "uViewMode");
    amplitudeLocation_ = gl::glGetUniformLocation(terrainProgram_, "uAmplitude");
    lineViewProjectionLocation_ =
        gl::glGetUniformLocation(lineProgram_, "uViewProjection");
    lineBiasLocation_ = gl::glGetUniformLocation(lineProgram_, "uDepthBias");
    return true;
}

bool GlRenderer::createStaticBuffers(std::string& error) {
    using namespace gl;
    glGenTextures(1, &heightTexture_);
    glBindTexture(GL_TEXTURE_2D, heightTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R32F,
        terrain_.gridSize(),
        terrain_.gridSize(),
        0,
        GL_RED,
        GL_FLOAT,
        terrain_.heights().data());

    std::array<unsigned int, kIndicesPerPatch> indices{};
    std::size_t write = 0;
    for (int z = 0; z < kPatchQuads; ++z) {
        for (int x = 0; x < kPatchQuads; ++x) {
            const unsigned int i00 =
                static_cast<unsigned int>(z * kPatchVertexCount + x);
            const unsigned int i10 = i00 + 1;
            const unsigned int i01 = i00 + kPatchVertexCount;
            const unsigned int i11 = i01 + 1;
            indices[write++] = i00;
            indices[write++] = i10;
            indices[write++] = i11;
            indices[write++] = i00;
            indices[write++] = i11;
            indices[write++] = i01;
        }
    }

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glGenBuffers(1, &indexBuffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer_);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(sizeof(indices)),
        indices.data(),
        GL_STATIC_DRAW);

    glGenBuffers(1, &nodeBuffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nodeBuffer_);
    const std::size_t maxLeaves =
        static_cast<std::size_t>(1) << (2 * terrain_.maxDepth());
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(maxLeaves * sizeof(NodeGpu)),
        nullptr,
        GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nodeBuffer_);

    glGenBuffers(1, &seamVertexBuffer_);
    glBindBuffer(GL_ARRAY_BUFFER, seamVertexBuffer_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(2 * 1024 * 1024),
        nullptr,
        GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    if (glGetError() != GL_NO_ERROR) {
        error = "failed to create static GPU buffers";
        return false;
    }
    return true;
}

bool GlRenderer::createFramebuffer(std::string& error) {
    using namespace gl;
    glGenTextures(1, &colorTexture_);
    glBindTexture(GL_TEXTURE_2D, colorTexture_);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        width_,
        height_,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenRenderbuffers(1, &depthRenderbuffer_);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer_);
    glRenderbufferStorage(
        GL_RENDERBUFFER,
        GL_DEPTH_COMPONENT24,
        width_,
        height_);

    glGenFramebuffers(1, &framebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        colorTexture_,
        0);
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER,
        GL_DEPTH_ATTACHMENT,
        GL_RENDERBUFFER,
        depthRenderbuffer_);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        error = "offscreen framebuffer is incomplete";
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void GlRenderer::buildGpuMetadata(const SceneFrame& frame, DebugView view) {
    (void)view;
    std::vector<NodeGpu> nodes;
    nodes.reserve(frame.selection.leaves.size());
    batches_.clear();

    for (int depth = 0; depth <= terrain_.maxDepth(); ++depth) {
        const int start = static_cast<int>(nodes.size());
        for (std::size_t leafIndex = 0;
             leafIndex < frame.selection.leaves.size();
             ++leafIndex) {
            const LeafNode& leaf = frame.selection.leaves[leafIndex];
            if (leaf.key.depth != depth) {
                continue;
            }
            NodeGpu node{};
            node.rect = {
                leaf.x0,
                leaf.z0,
                leaf.x1 - leaf.x0,
                static_cast<float>(leaf.key.depth),
            };
            node.bounds = {
                leaf.minHeight,
                leaf.maxHeight,
                0.0f,
                0.0f,
            };
            const SeamResult& seam = frame.seams[leafIndex];
            for (int edge = 0; edge < kEdges; ++edge) {
                const auto& segments =
                    seam.edgeSegments[static_cast<std::size_t>(edge)];
                for (int segmentIndex = 0;
                     segmentIndex < kMaxEdgeSegments &&
                     static_cast<std::size_t>(segmentIndex) < segments.size();
                     ++segmentIndex) {
                    const StitchSegment& segment =
                        segments[static_cast<std::size_t>(segmentIndex)];
                    EdgeSegmentGpu& target =
                        node.edges[static_cast<std::size_t>(
                            edge * kMaxEdgeSegments + segmentIndex)];
                    target.range = {segment.t0, segment.t1, 1.0f, 0.0f};
                    const float lowAlong =
                        edge < 2
                            ? segment.neighborMinX
                            : segment.neighborMinZ;
                    target.low = {lowAlong, segment.neighborSpan, 0.0f, 0.0f};
                }
            }
            nodes.push_back(node);
        }
        const int count = static_cast<int>(nodes.size()) - start;
        if (count > 0) {
            batches_.push_back({start, count});
        }
    }

    gpuNodesWords_.resize(nodes.size() * sizeof(NodeGpu) / sizeof(unsigned int));
    if (!nodes.empty()) {
        std::memcpy(
            gpuNodesWords_.data(),
            nodes.data(),
            nodes.size() * sizeof(NodeGpu));
    }
}

void GlRenderer::issueMeshDraws(const SceneFrame& frame, DebugView view) {
    using namespace gl;
    glViewport(0, 0, width_, height_);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);
    glPolygonMode(
        GL_FRONT_AND_BACK,
        view == DebugView::Wireframe ? GL_LINE : GL_FILL);

    if (view == DebugView::Wireframe) {
        glClearColor(0.025f, 0.055f, 0.09f, 1.0f);
    } else {
        glClearColor(0.52f, 0.68f, 0.82f, 1.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(terrainProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, heightTexture_);
    glUniform1i(heightMapLocation_, 0);
    glUniform1f(extentLocation_, terrain_.extent());
    glUniform1i(gridLocation_, terrain_.gridSize());
    glUniform1i(
        viewModeLocation_,
        view == DebugView::Lod ? 2 : 0);
    glUniform1f(amplitudeLocation_, terrain_.parameters().amplitude);
    const float aspect =
        static_cast<float>(width_) / static_cast<float>(height_);
    const Mat4 viewProjection = frame.camera.viewProjectionMatrix(aspect);
    glUniformMatrix4fv(
        viewProjectionLocation_,
        1,
        GL_FALSE,
        viewProjection.m.data());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nodeBuffer_);
    glBindVertexArray(vao_);

    for (const Batch& batch : batches_) {
        glUniform1i(nodeOffsetLocation_, batch.nodeOffset);
        glDrawElementsInstanced(
            GL_TRIANGLES,
            kIndicesPerPatch,
            GL_UNSIGNED_INT,
            nullptr,
            batch.nodeCount);
    }
    glBindVertexArray(0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glUseProgram(0);
}

void GlRenderer::issueSeamOverlay(const SceneFrame& frame) {
    if (frame.seams.empty()) {
        return;
    }
    using namespace gl;
    seamVertices_.clear();
    for (const SeamResult& result : frame.seams) {
        for (const SeamLineVertex& vertex : result.seamLines) {
            seamVertices_.push_back(vertex.position.x);
            seamVertices_.push_back(vertex.position.y);
            seamVertices_.push_back(vertex.position.z);
        }
    }
    if (seamVertices_.empty()) {
        return;
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glUseProgram(lineProgram_);
    glBindBuffer(GL_ARRAY_BUFFER, seamVertexBuffer_);
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        static_cast<GLsizeiptr>(seamVertices_.size() * sizeof(float)),
        seamVertices_.data());
    glBindVertexArray(vao_);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(3 * sizeof(float)),
        reinterpret_cast<const void*>(0));
    glUniform1f(lineBiasLocation_, 0.0005f);
    const float aspect =
        static_cast<float>(width_) / static_cast<float>(height_);
    const Mat4 viewProjection = frame.camera.viewProjectionMatrix(aspect);
    glUniformMatrix4fv(
        lineViewProjectionLocation_,
        1,
        GL_FALSE,
        viewProjection.m.data());
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    glLineWidth(1.0f);
    glDrawArrays(
        GL_LINES,
        0,
        static_cast<GLsizei>(seamVertices_.size() / 3));
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDepthMask(GL_TRUE);
    glUseProgram(0);
}

bool GlRenderer::render(
    const SceneFrame& frame,
    DebugView view,
    std::string& error) {
    if (!initialized_) {
        error = "renderer is not initialized";
        return false;
    }
    if (terrain_.version() != terrainVersion_) {
        error = "terrain version changed; rebuild GPU resources";
        return false;
    }

    buildGpuMetadata(frame, view);
    if (!gpuNodesWords_.empty()) {
        gl::glBindBuffer(gl::GL_SHADER_STORAGE_BUFFER, nodeBuffer_);
        gl::glBufferSubData(
            gl::GL_SHADER_STORAGE_BUFFER,
            0,
            static_cast<gl::GLsizeiptr>(
                gpuNodesWords_.size() * sizeof(unsigned int)),
            gpuNodesWords_.data());
    }

    issueMeshDraws(frame, view);
    if (view == DebugView::Seam) {
        issueSeamOverlay(frame);
    }
    gl::glFinish();
    if (gl::glGetError() != gl::GL_NO_ERROR) {
        error = "OpenGL error while rendering frame";
        return false;
    }
    return true;
}

bool GlRenderer::capturePpm(
    const std::string& filename,
    std::string& error) {
    if (!initialized_) {
        error = "renderer is not initialized";
        return false;
    }
    using namespace gl;
    readback_.resize(
        static_cast<std::size_t>(width_) *
        static_cast<std::size_t>(height_) * 4u);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(
        0,
        0,
        width_,
        height_,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        readback_.data());

    ImageRgb image;
    image.width = width_;
    image.height = height_;
    image.pixels.resize(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 3u);
    for (int y = 0; y < height_; ++y) {
        const int sourceY = height_ - 1 - y;
        for (int x = 0; x < width_; ++x) {
            const std::size_t source =
                (static_cast<std::size_t>(sourceY) * width_ +
                 static_cast<std::size_t>(x)) *
                4u;
            const std::size_t destination =
                (static_cast<std::size_t>(y) * width_ +
                 static_cast<std::size_t>(x)) *
                3u;
            image.pixels[destination] = readback_[source];
            image.pixels[destination + 1] = readback_[source + 1];
            image.pixels[destination + 2] = readback_[source + 2];
        }
    }
    return writePpm(filename, image, error);
}

void GlRenderer::presentToDefaultFramebuffer() const {
    if (!initialized_) {
        return;
    }
    using namespace gl;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer_);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(
        0,
        0,
        width_,
        height_,
        0,
        0,
        width_,
        height_,
        GL_COLOR_BUFFER_BIT,
        GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GlRenderer::shutdown() {
    if (!initialized_) {
        return;
    }
    using namespace gl;
    glDeleteBuffers(1, &seamVertexBuffer_);
    glDeleteBuffers(1, &nodeBuffer_);
    glDeleteBuffers(1, &indexBuffer_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteTextures(1, &heightTexture_);
    glDeleteTextures(1, &colorTexture_);
    glDeleteRenderbuffers(1, &depthRenderbuffer_);
    glDeleteFramebuffers(1, &framebuffer_);
    glDeleteProgram(terrainProgram_);
    glDeleteProgram(lineProgram_);
    initialized_ = false;
}

}  // namespace terrain

