namespace ReplayForge.Utilities;

public sealed class ReplayException : Exception
{
    public ReplayException(string code, string message, object? details = null)
        : base(message)
    {
        Code = code;
        Details = details;
    }

    public string Code { get; }

    public object? Details { get; }
}
