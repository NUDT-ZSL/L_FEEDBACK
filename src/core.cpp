#include "core.h"

Vec3 transformPoint(const Mat4& m, const Vec3& p) {
    const Vec4 r = m * Vec4(p.x, p.y, p.z, 1.0);
    return {r.x, r.y, r.z};
}

Vec3 transformVector(const Mat4& m, const Vec3& v) {
    const Vec4 r = m * Vec4(v.x, v.y, v.z, 0.0);
    return {r.x, r.y, r.z};
}

Mat4 makeTRS(const Vec3& translation, const Vec3& rotationDeg, const Vec3& scale) {
    const double kPi = 3.141592653589793238462643383279502884;
    const double rx = rotationDeg.x * kPi / 180.0;
    const double ry = rotationDeg.y * kPi / 180.0;
    const double rz = rotationDeg.z * kPi / 180.0;
    const double cx = std::cos(rx), sx = std::sin(rx);
    const double cy = std::cos(ry), sy = std::sin(ry);
    const double cz = std::cos(rz), sz = std::sin(rz);

    // R = Rz * Ry * Rx. Translation is applied last, scale first.
    Mat4 r;
    r.at(0, 0) = cz * cy;
    r.at(0, 1) = cz * sy * sx - sz * cx;
    r.at(0, 2) = cz * sy * cx + sz * sx;
    r.at(1, 0) = sz * cy;
    r.at(1, 1) = sz * sy * sx + cz * cx;
    r.at(1, 2) = sz * sy * cx - cz * sx;
    r.at(2, 0) = -sy;
    r.at(2, 1) = cy * sx;
    r.at(2, 2) = cy * cx;
    r.at(3, 3) = 1.0;

    for (int row = 0; row < 3; ++row) {
        r.at(row, 0) *= scale.x;
        r.at(row, 1) *= scale.y;
        r.at(row, 2) *= scale.z;
    }
    r.at(0, 3) = translation.x;
    r.at(1, 3) = translation.y;
    r.at(2, 3) = translation.z;
    return r;
}

uint64_t fnv1a64(const void* data, size_t size, uint64_t seed) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::string formatDouble(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

Framebuffer::Framebuffer(int width_, int height_, int tileSize_)
    : width(width_), height(height_), tileSize(tileSize_) {
    tilesX = (width + tileSize - 1) / tileSize;
    tilesY = (height + tileSize - 1) / tileSize;
    color.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 3U, 0U);
    depth.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0U);
}

void Framebuffer::tileBounds(int index, int& x0, int& y0, int& x1, int& y1) const {
    const int tx = index % tilesX;
    const int ty = index / tilesX;
    x0 = tx * tileSize;
    y0 = ty * tileSize;
    x1 = std::min(width, x0 + tileSize);
    y1 = std::min(height, y0 + tileSize);
}

uint64_t Framebuffer::colorHash() const {
    return fnv1a64(color.data(), color.size());
}

uint64_t Framebuffer::depthHash() const {
    return fnv1a64(depth.data(), depth.size() * sizeof(uint32_t));
}

uint64_t Framebuffer::tileColorHash(int index) const {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    tileBounds(index, x0, y0, x1, y1);
    uint64_t h = 1469598103934665603ULL;
    for (int y = y0; y < y1; ++y) {
        const size_t row = static_cast<size_t>((y * width + x0) * 3);
        const size_t bytes = static_cast<size_t>((x1 - x0) * 3);
        h = fnv1a64(color.data() + row, bytes, h);
    }
    return h;
}
