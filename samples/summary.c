#include <string.h>

char summary_destination[64];

__declspec(noinline)
void sink_memcpy(char *input)
{
    memcpy(summary_destination, input, 16);
}

__declspec(noinline)
void build_buffer(char *input)
{
    sink_memcpy(input);
}

__declspec(noinline)
void relay(char *input)
{
    build_buffer(input);
}

__declspec(noinline)
void source_a(char *input)
{
    relay(input);
}

__declspec(noinline)
void source_b(char *input)
{
    sink_memcpy(input);
}

__declspec(noinline)
void f1(int depth);

__declspec(noinline)
void f2(int depth)
{
    if (depth > 0) {
        f1(depth - 1);
    }
}

__declspec(noinline)
void f1(int depth)
{
    if (depth > 0) {
        f2(depth - 1);
    }
}

int main(void)
{
    source_a(summary_destination);
    source_b(summary_destination);
    f1(2);
    return 0;
}
