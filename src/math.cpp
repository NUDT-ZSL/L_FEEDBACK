#include "math.h"

#include <cstring>

namespace cloud {

Mat4 Mat4::identity() {
    Mat4 result;
    result.m = {1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f};
    return result;
}

Mat4 Mat4::lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 forward = normalize(target - eye);
    const Vec3 side = normalize(cross(forward, up));
    const Vec3 correctedUp = cross(side, forward);

    Mat4 result = identity();
    result(0, 0) = side.x;
    result(0, 1) = side.y;
    result(0, 2) = side.z;
    result(0, 3) = -dot(side, eye);
    result(1, 0) = correctedUp.x;
    result(1, 1) = correctedUp.y;
    result(1, 2) = correctedUp.z;
    result(1, 3) = -dot(correctedUp, eye);
    result(2, 0) = -forward.x;
    result(2, 1) = -forward.y;
    result(2, 2) = -forward.z;
    result(2, 3) = dot(forward, eye);
    return result;
}

Mat4 Mat4::perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    Mat4 result;
    result.m.fill(0.0f);
    result(0, 0) = f / aspect;
    result(1, 1) = f;
    result(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
    result(2, 3) = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    result(3, 2) = -1.0f;
    return result;
}

Mat4 Mat4::operator*(const Mat4& other) const {
    Mat4 result;
    result.m.fill(0.0f);
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += (*this)(row, k) * other(k, column);
            }
            result(row, column) = sum;
        }
    }
    return result;
}

Vec4 Mat4::operator*(const Vec4& value) const {
    return {
        m[0] * value.x + m[1] * value.y + m[2] * value.z + m[3] * value.w,
        m[4] * value.x + m[5] * value.y + m[6] * value.z + m[7] * value.w,
        m[8] * value.x + m[9] * value.y + m[10] * value.z + m[11] * value.w,
        m[12] * value.x + m[13] * value.y + m[14] * value.z + m[15] * value.w,
    };
}

Mat4 Mat4::inverse() const {
    double augmented[4][8]{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            augmented[row][column] = static_cast<double>((*this)(row, column));
            augmented[row][column + 4] = row == column ? 1.0 : 0.0;
        }
    }

    for (int column = 0; column < 4; ++column) {
        int pivot = column;
        double pivotMagnitude = std::abs(augmented[pivot][column]);
        for (int row = column + 1; row < 4; ++row) {
            const double magnitude = std::abs(augmented[row][column]);
            if (magnitude > pivotMagnitude) {
                pivot = row;
                pivotMagnitude = magnitude;
            }
        }
        if (pivotMagnitude < 1.0e-20) {
            throw std::runtime_error("singular matrix inversion");
        }
        if (pivot != column) {
            for (int k = 0; k < 8; ++k) {
                std::swap(augmented[pivot][k], augmented[column][k]);
            }
        }
        const double divisor = augmented[column][column];
        for (int k = 0; k < 8; ++k) {
            augmented[column][k] /= divisor;
        }
        for (int row = 0; row < 4; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = augmented[row][column];
            for (int k = 0; k < 8; ++k) {
                augmented[row][k] -= factor * augmented[column][k];
            }
        }
    }

    Mat4 result;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result(row, column) = static_cast<float>(augmented[row][column + 4]);
        }
    }
    return result;
}

Vec3 Mat4::transformPoint(const Vec3& point) const {
    const Vec4 result = (*this) * Vec4{point.x, point.y, point.z, 1.0f};
    if (std::abs(result.w) > 1.0e-20f) {
        return {result.x / result.w, result.y / result.w, result.z / result.w};
    }
    return {result.x, result.y, result.z};
}

Vec3 Mat4::transformDirection(const Vec3& direction) const {
    return {
        m[0] * direction.x + m[1] * direction.y + m[2] * direction.z,
        m[4] * direction.x + m[5] * direction.y + m[6] * direction.z,
        m[8] * direction.x + m[9] * direction.y + m[10] * direction.z,
    };
}

Camera Camera::fromLookAt(const Vec3& position, const Vec3& target, const Vec3& up, float fovDegrees,
                          float aspect, float nearPlane, float farPlane) {
    Camera camera;
    camera.position = position;
    camera.view = Mat4::lookAt(position, target, up);
    camera.projection = Mat4::perspective(fovDegrees * kPi / 180.0f, aspect, nearPlane, farPlane);
    camera.inverseViewProjection = (camera.projection * camera.view).inverse();
    return camera;
}

Camera Camera::fromMatrices(const Mat4& view, const Mat4& projection) {
    Camera camera;
    camera.view = view;
    camera.projection = projection;
    camera.inverseViewProjection = (projection * view).inverse();
    const Vec3 origin = view.inverse().transformPoint(Vec3{0.0f, 0.0f, 0.0f});
    camera.position = origin;
    return camera;
}

Ray Camera::rayForPixel(float pixelX, float pixelY, int width, int height) const {
    const float ndcX = (2.0f * (pixelX + 0.5f) / static_cast<float>(width)) - 1.0f;
    const float ndcY = 1.0f - (2.0f * (pixelY + 0.5f) / static_cast<float>(height));
    const Vec4 nearClip = inverseViewProjection * Vec4{ndcX, ndcY, -1.0f, 1.0f};
    const Vec4 farClip = inverseViewProjection * Vec4{ndcX, ndcY, 1.0f, 1.0f};
    const Vec3 nearPoint{nearClip.x / nearClip.w, nearClip.y / nearClip.w, nearClip.z / nearClip.w};
    const Vec3 farPoint{farClip.x / farClip.w, farClip.y / farClip.w, farClip.z / farClip.w};
    return {position, normalize(farPoint - nearPoint)};
}

}  // namespace cloud

