#include <windows.h>
#include <string.h>

char g_dest[64];

__declspec(noinline)
void sink_memcpy(char *input)
{
    memcpy(g_dest, input, 8);
}

__declspec(noinline)
void relay(char *input)
{
    sink_memcpy(input);
}

__declspec(noinline)
void source_a(int unused, char *input)
{
    (void)unused;
    relay(input);
}

int main(void)
{
    source_a(0, "probe");
    return (int)g_dest[0];
}
