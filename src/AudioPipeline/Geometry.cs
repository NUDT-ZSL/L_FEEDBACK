using System.Text.Json.Serialization;

namespace AudioPipeline;

public struct Vec3 : IEquatable<Vec3>
{
    [JsonPropertyName("x")]
    public double X { get; set; }

    [JsonPropertyName("y")]
    public double Y { get; set; }

    [JsonPropertyName("z")]
    public double Z { get; set; }

    public Vec3(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public static Vec3 operator +(Vec3 a, Vec3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

    public static Vec3 operator -(Vec3 a, Vec3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

    public static Vec3 operator *(Vec3 a, double scalar) => new(a.X * scalar, a.Y * scalar, a.Z * scalar);

    public static double Dot(Vec3 a, Vec3 b) => (a.X * b.X) + (a.Y * b.Y) + (a.Z * b.Z);

    [JsonIgnore]
    public double LengthSquared => Dot(this, this);

    [JsonIgnore]
    public double Length => Math.Sqrt(LengthSquared);

    [JsonIgnore]
    public bool IsFinite => double.IsFinite(X) && double.IsFinite(Y) && double.IsFinite(Z);

    public bool Equals(Vec3 other) => X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);

    public override bool Equals(object? obj) => obj is Vec3 other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(X, Y, Z);

    public static bool operator ==(Vec3 left, Vec3 right) => left.Equals(right);

    public static bool operator !=(Vec3 left, Vec3 right) => !left.Equals(right);

    public override string ToString() => $"({X}, {Y}, {Z})";
}

public struct Aabb
{
    [JsonPropertyName("min")]
    public Vec3 Min { get; set; }

    [JsonPropertyName("max")]
    public Vec3 Max { get; set; }

    public Aabb(Vec3 min, Vec3 max)
    {
        Min = min;
        Max = max;
    }

    [JsonIgnore]
    public bool IsValid =>
        Min.IsFinite &&
        Max.IsFinite &&
        Min.X <= Max.X &&
        Min.Y <= Max.Y &&
        Min.Z <= Max.Z;

    public bool Contains(Vec3 point) =>
        point.X >= Min.X && point.X <= Max.X &&
        point.Y >= Min.Y && point.Y <= Max.Y &&
        point.Z >= Min.Z && point.Z <= Max.Z;
}

public static class Geometry
{
    private const double ParallelEpsilon = 1e-12;
    private const double IntersectionEpsilon = 1e-12;

    public static Vec3 Lerp(Vec3 start, Vec3 end, double amount) =>
        start + ((end - start) * amount);

    public static bool SegmentIntersectsAabb(Vec3 start, Vec3 end, Aabb box)
    {
        var direction = end - start;
        var tMin = 0.0;
        var tMax = 1.0;

        if (!IntersectAxis(start.X, direction.X, box.Min.X, box.Max.X, ref tMin, ref tMax) ||
            !IntersectAxis(start.Y, direction.Y, box.Min.Y, box.Max.Y, ref tMin, ref tMax) ||
            !IntersectAxis(start.Z, direction.Z, box.Min.Z, box.Max.Z, ref tMin, ref tMax))
        {
            return false;
        }

        return tMax >= -IntersectionEpsilon && tMin <= 1.0 + IntersectionEpsilon && tMin <= tMax;
    }

    private static bool IntersectAxis(
        double origin,
        double direction,
        double min,
        double max,
        ref double tMin,
        ref double tMax)
    {
        if (Math.Abs(direction) < ParallelEpsilon)
        {
            return origin >= min - IntersectionEpsilon && origin <= max + IntersectionEpsilon;
        }

        var first = (min - origin) / direction;
        var second = (max - origin) / direction;
        if (first > second)
        {
            (first, second) = (second, first);
        }

        tMin = Math.Max(tMin, first);
        tMax = Math.Min(tMax, second);
        return tMin <= tMax + IntersectionEpsilon;
    }
}
