#include <windows.h>

__declspec(noinline)
char *sanitize(char *input)
{
    (void)input;
    return (char *)0;
}

__declspec(noinline)
void safe_copy(char *input)
{
    char destination[16];
    int index;
    if (input == (char *)0) {
        return;
    }
    for (index = 0; index < 15 && input[index] != '\0'; ++index) {
        destination[index] = input[index];
    }
    destination[index] = '\0';
}

__declspec(noinline)
void sink_network(char *input)
{
    WinExec((LPCSTR)input, SW_SHOW);
}

__declspec(noinline)
void sink_win_exec(char *input)
{
    WinExec((LPCSTR)input, SW_SHOW);
}

__declspec(noinline)
void sink_create_file(char *input)
{
    CreateFileW((LPCWSTR)input, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
}

__declspec(noinline)
void relay(char *input)
{
    sink_network(input);
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
void only_clean(char *input)
{
    char *clean = sanitize(input);
    sink_create_file(clean);
}

__declspec(noinline)
void source_a(int unused, char *input)
{
    (void)unused;
    safe_copy(input);
    wrapper_b(input);
    relay(input);
    only_clean(input);
}

__declspec(noinline)
void direct_window(char *input)
{
    safe_copy(input);
}

int main(void)
{
    source_a(0, GetCommandLineA());
    direct_window(GetCommandLineA());
    return 0;
}
