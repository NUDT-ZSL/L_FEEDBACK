using System.Buffers.Binary;
using System.Text;

namespace ReplayForge.Media;

internal static class Bits
{
    public static ushort U16(ReadOnlySpan<byte> data, int offset)
        => BinaryPrimitives.ReadUInt16BigEndian(data.Slice(offset, 2));

    public static uint U24(ReadOnlySpan<byte> data, int offset)
        => (uint)(data[offset] << 16 | data[offset + 1] << 8 | data[offset + 2]);

    public static uint U32(ReadOnlySpan<byte> data, int offset)
        => BinaryPrimitives.ReadUInt32BigEndian(data.Slice(offset, 4));

    public static ulong U64(ReadOnlySpan<byte> data, int offset)
        => BinaryPrimitives.ReadUInt64BigEndian(data.Slice(offset, 8));

    public static string FourCc(ReadOnlySpan<byte> data, int offset)
        => Encoding.ASCII.GetString(data.Slice(offset, 4));
}
