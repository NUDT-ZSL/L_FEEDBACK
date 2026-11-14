#include "text_overlay.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "shaders.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace {
constexpr int kMaxQuads = 8192;
constexpr int kTextBufferBytes = 1024 * 1024;
}

TextOverlay::~TextOverlay() {
    shutdown();
}

void TextOverlay::initialize() {
    shader_.build(kOverlayVertexShader, kOverlayFragmentShader, "text_overlay");
    textBuffer_.resize(kTextBufferBytes);

    glGenVertexArrays(1, &textVao_);
    glGenBuffers(1, &textVbo_);
    glGenBuffers(1, &textEbo_);

    glBindVertexArray(textVao_);
    glBindBuffer(GL_ARRAY_BUFFER, textVbo_);
    glBufferData(GL_ARRAY_BUFFER, kMaxQuads * 4 * 16, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, textEbo_);

    std::vector<uint32_t> indices(static_cast<size_t>(kMaxQuads) * 6);
    for (int i = 0; i < kMaxQuads; ++i) {
        const uint32_t base = static_cast<uint32_t>(i * 4);
        const size_t offset = static_cast<size_t>(i) * 6;
        indices[offset + 0] = base + 0;
        indices[offset + 1] = base + 1;
        indices[offset + 2] = base + 2;
        indices[offset + 3] = base + 0;
        indices[offset + 4] = base + 2;
        indices[offset + 5] = base + 3;
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(indices.size() * sizeof(uint32_t)),
                 indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16,
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 16,
                          reinterpret_cast<void*>(12));

    glGenVertexArrays(1, &panelVao_);
    glGenBuffers(1, &panelVbo_);
    glBindVertexArray(panelVao_);
    glBindBuffer(GL_ARRAY_BUFFER, panelVbo_);
    glBufferData(GL_ARRAY_BUFFER, 6 * 16, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16,
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 16,
                          reinterpret_cast<void*>(12));

    glBindVertexArray(0);
    checkGLError("TextOverlay::initialize");
}

void TextOverlay::shutdown() {
    shader_.destroy();
    if (textEbo_ != 0) {
        glDeleteBuffers(1, &textEbo_);
        textEbo_ = 0;
    }
    if (textVbo_ != 0) {
        glDeleteBuffers(1, &textVbo_);
        textVbo_ = 0;
    }
    if (textVao_ != 0) {
        glDeleteVertexArrays(1, &textVao_);
        textVao_ = 0;
    }
    if (panelVbo_ != 0) {
        glDeleteBuffers(1, &panelVbo_);
        panelVbo_ = 0;
    }
    if (panelVao_ != 0) {
        glDeleteVertexArrays(1, &panelVao_);
        panelVao_ = 0;
    }
}

void TextOverlay::draw(const std::string& text,
                       int viewportWidth,
                       int viewportHeight,
                       float scale) {
    if (shader_.id() == 0 || text.empty()) {
        return;
    }

    const int lineCount =
        static_cast<int>(std::count(text.begin(), text.end(), '\n')) + 1;
    const float panelWidth = std::min(static_cast<float>(viewportWidth) - 20.0f,
                                      650.0f);
    const float panelHeight = static_cast<float>(lineCount) * 12.0f * scale + 18.0f;

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    shader_.use();
    shader_.setVec2("uViewport", glm::vec2(viewportWidth, viewportHeight));
    shader_.setFloat("uScale", scale);
    drawPanel(10.0f, 10.0f, panelWidth, panelHeight,
              glm::vec4(0.018f, 0.024f, 0.032f, 0.79f),
              viewportWidth, viewportHeight);

    std::vector<char> mutableText(text.begin(), text.end());
    mutableText.push_back('\0');
    unsigned char color[4] = {224, 234, 240, 255};
    const int quadCount = stb_easy_font_print(6.0f, 5.0f,
                                              mutableText.data(), color,
                                              textBuffer_.data(),
                                              static_cast<int>(textBuffer_.size()));
    const int quads = std::min(quadCount, kMaxQuads);
    if (quads > 0) {
        glBindVertexArray(textVao_);
        glBindBuffer(GL_ARRAY_BUFFER, textVbo_);
        const int byteCount = quads * 4 * 16;
        if (byteCount <= static_cast<int>(textBuffer_.size())) {
            glBufferSubData(GL_ARRAY_BUFFER, 0, byteCount, textBuffer_.data());
            glDrawElements(GL_TRIANGLES, quads * 6, GL_UNSIGNED_INT, nullptr);
        }
    }

    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

void TextOverlay::drawPanel(float x, float y, float width, float height,
                            const glm::vec4& color, int viewportWidth,
                            int viewportHeight) {
    struct PanelVertex {
        float x;
        float y;
        float z;
        unsigned char color[4];
    };
    const unsigned char c[4] = {
        static_cast<unsigned char>(color.r * 255.0f),
        static_cast<unsigned char>(color.g * 255.0f),
        static_cast<unsigned char>(color.b * 255.0f),
        static_cast<unsigned char>(color.a * 255.0f)
    };
    PanelVertex vertices[6] = {
        {x, y, 0, {c[0], c[1], c[2], c[3]}},
        {x + width, y, 0, {c[0], c[1], c[2], c[3]}},
        {x + width, y + height, 0, {c[0], c[1], c[2], c[3]}},
        {x, y, 0, {c[0], c[1], c[2], c[3]}},
        {x + width, y + height, 0, {c[0], c[1], c[2], c[3]}},
        {x, y + height, 0, {c[0], c[1], c[2], c[3]}}
    };
    (void)viewportWidth;
    (void)viewportHeight;
    glBindVertexArray(panelVao_);
    glBindBuffer(GL_ARRAY_BUFFER, panelVbo_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

