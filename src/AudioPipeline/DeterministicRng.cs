namespace AudioPipeline;

internal struct DeterministicRng
{
    private ulong _state;

    public DeterministicRng(ulong seed)
    {
        _state = seed == 0 ? 0x9E3779B97F4A7C15UL : seed;
    }

    public ulong NextUInt64()
    {
        var z = _state + 0x9E3779B97F4A7C15UL;
        _state = z;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / (1UL << 53));

    public double NextSigned() => (NextDouble() * 2.0) - 1.0;

    public int NextInt(int exclusiveMax) => (int)(NextUInt64() % (ulong)exclusiveMax);
}
