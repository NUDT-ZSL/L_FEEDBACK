#include "baker/interrupt.h"

#include <csignal>
#include <string>

namespace baker {
namespace {

extern "C" void signal_handler(int signal_number) {
    g_signal_number.store(signal_number, std::memory_order_relaxed);
}

}  // namespace

void install_signal_handlers() {
    std::signal(SIGINT, signal_handler);
#ifdef SIGTERM
    std::signal(SIGTERM, signal_handler);
#endif
#ifdef SIGBREAK
    std::signal(SIGBREAK, signal_handler);
#endif
}

void request_interrupt_for_tests(int signal_number) {
    g_signal_number.store(signal_number, std::memory_order_relaxed);
}

bool interrupt_requested() noexcept {
    return g_signal_number.load(std::memory_order_relaxed) != 0;
}

int interrupt_signal() noexcept {
    return g_signal_number.load(std::memory_order_relaxed);
}

void throw_if_interrupted() {
    const int signal_number = interrupt_signal();
    if (signal_number != 0) {
        throw Interrupted(signal_number,
                          "interrupted by signal " + std::to_string(signal_number));
    }
}

}  // namespace baker
