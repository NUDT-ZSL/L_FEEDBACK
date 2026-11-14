using System.Text.Json;
using System.Text.Json.Serialization;

namespace ReplayForge.Utilities;

public static class JsonIO
{
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        NumberHandling = JsonNumberHandling.AllowReadingFromString,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
    };

    public static T DeserializeFile<T>(string path)
    {
        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<T>(json, Options)
                ?? throw new ReplayException("E_MANIFEST_INVALID", $"JSON document is empty: {path}");
        }
        catch (ReplayException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new ReplayException("E_MANIFEST_INVALID", $"Cannot parse JSON file '{path}': {ex.Message}");
        }
    }

    public static string Serialize(object value) => JsonSerializer.Serialize(value, Options);

    public static void WriteAtomic(string path, object value)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath)!;
        Directory.CreateDirectory(directory);
        var temporary = fullPath + ".tmp-" + Environment.ProcessId + "-" + Guid.NewGuid().ToString("N");
        try
        {
            using (var stream = new FileStream(
                       temporary,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       64 * 1024,
                       FileOptions.WriteThrough))
            using (var writer = new StreamWriter(stream))
            {
                writer.Write(Serialize(value));
                writer.Flush();
                stream.Flush(flushToDisk: true);
            }

            File.Move(temporary, fullPath, overwrite: true);
        }
        finally
        {
            TryDelete(temporary);
        }
    }

    public static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Best effort cleanup only.
        }
    }
}
