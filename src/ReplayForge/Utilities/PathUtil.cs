namespace ReplayForge.Utilities;

public static class PathUtil
{
    public static string Resolve(string baseDirectory, string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return "";
        }

        return Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(baseDirectory, path));
    }

    public static string ToDisplay(string baseDirectory, string path)
    {
        var relative = Path.GetRelativePath(baseDirectory, path);
        return relative.StartsWith("..", StringComparison.Ordinal) ? path : relative;
    }
}
