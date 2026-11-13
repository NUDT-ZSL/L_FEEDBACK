#include <windows.h>

__declspec(noinline)
void source_a(char *input)
{
    Sleep(10);
    WinExec((LPCSTR)input, SW_SHOW);
}

int main(void)
{
    source_a(GetCommandLineA());
    return 0;
}
