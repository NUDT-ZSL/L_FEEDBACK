#include <windows.h>

__declspec(noinline)
void sink_win_exec(char *input)
{
    WinExec((LPCSTR)input, SW_SHOW);
}

__declspec(noinline)
void wrapper_a(char *input)
{
    sink_win_exec(input);
}

__declspec(noinline)
void wrapper_b(char *input)
{
    wrapper_a(input);
}

__declspec(noinline)
void source_a(char *input)
{
    wrapper_b(input);
}

__declspec(noinline)
void source_b(char *input)
{
    sink_win_exec(input);
}

int main(void)
{
    source_a(GetCommandLineA());
    source_b(GetCommandLineA());
    return 0;
}
