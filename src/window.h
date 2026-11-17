#pragma once

namespace ct {

class WindowMessageLoop {
public:
    WindowMessageLoop();
    ~WindowMessageLoop();

    WindowMessageLoop(const WindowMessageLoop&) = delete;
    WindowMessageLoop& operator=(const WindowMessageLoop&) = delete;

    bool pump();
    bool shouldQuit() const { return quit_; }

private:
    bool quit_ = false;
};

}  // namespace ct
