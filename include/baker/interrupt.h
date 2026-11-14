#pragma once

#include <atomic>

#include "baker/error.h"

namespace baker {

inline std::atomic<int> g_signal_number{0};

void install_signal_handlers();
void request_interrupt_for_tests(int signal_number);
[[nodiscard]] bool interrupt_requested() noexcept;
[[nodiscard]] int interrupt_signal() noexcept;
void throw_if_interrupted();

}  // namespace baker
