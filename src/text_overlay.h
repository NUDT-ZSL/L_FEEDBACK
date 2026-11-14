#pragma once

#include <glad/gl.h>

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "gl_utils.h"

class TextOverlay {
public:
    TextOverlay() = default;
    ~TextOverlay();

    TextOverlay(const TextOverlay&) = delete;
    TextOverlay& operator=(const TextOverlay&) = delete;

    void initialize();
    void shutdown();
    void draw(const std::string& text, int viewportWidth, int viewportHeight, float scale);

private:
    void drawPanel(float x, float y, float width, float height,
                   const glm::vec4& color, int viewportWidth, int viewportHeight);

    ShaderProgram shader_;
    GLuint textVao_ = 0;
    GLuint textVbo_ = 0;
    GLuint textEbo_ = 0;
    GLuint panelVao_ = 0;
    GLuint panelVbo_ = 0;
    std::vector<char> textBuffer_;
};

