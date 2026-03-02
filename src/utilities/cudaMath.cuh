#pragma once
#include <math.h>
#include <limits>
#include <stdio.h>
#include <cuda_runtime.h>
#include "../../extern/cuda_helpers/helper_math.h"

#define deg_to_rad(deg) (((deg) / 360.f) * 2 * M_PI)
#define rad_to_deg(rad) (((rad) / 2 / M_PI) * 360.f)

// Useful functions
template <class T>
T hcuFract(const T& x) {
    return x - floor(x);
}
template <class T>
T hcuRound(const T& x) {
    return floor(x + T(0.5));
}
template <class TA, class TB, class TC>
TA hcuClamp(const TA& x, const TB& minv, const TC& maxv) {
    if (x < (TA)minv)
        return (TA)minv;
    else if (x > (TA)maxv)
        return (TA)maxv;
    return x;
}
template <class T>
T hcuSqr(const T& x) {
    return x * x;
}
template <class T>
T hcuCube(const T& x) {
    return x * x * x;
}
template <class T>
T hcuSign(const T& x) {
    if (x > 0)
        return T(1);
    else if (x < 0)
        return T(-1);
    return T(0);
}
template <class T>
T hcuMin(const T& a, const T& b) {
    return (a < b ? a : b);
}
template <class T>
T hcuMin(const T& a, const T& b, const T& c) {
    return min(min(a, b), c);
};
template <class T>
T hcuMin(const T& a, const T& b, const T& c, const T& d) {
    return min(min(a, b), min(c, d));
};
template <class T>
T hcuMax(const T& a, const T& b) {
    return (a > b ? a : b);
}
template <class T>
T hcuMax(const T& a, const T& b, const T& c) {
    return max(max(a, b), c);
}
template <class T>
T hcuMax(const T& a, const T& b, const T& c, const T& d) {
    return max(max(a, b), max(c, d));
}
template <class T>
T hcuSmoothStep(const T& l, const T& u, const T& x) {
    T t = clamp((x - l) / (u - l), (T)0, (T)1);
    return t * t * (3 - 2 * t);
}
template <class T>
T hcuLerp(const T& a, const T& b, const float& s) {
    T t = b * s + (1 - s) * a;
    return t;
}
template <class T>
void hcuLoop(T& a, const T& low, const T& high, const T& inc = 1.0) {
    if (a >= high)
        a = low;
    else
        a += inc;
}
template <class T>
__host__ __device__ T hcuDeg2rad(const T& x) {
    return x * T(3.1415926536f / 180.f);
}
template <class T>
__host__ __device__ T hcuRad2deg(const T& x) {
    return x * T(180.f / 3.1415926536f);
}
template <class T>
T hcuRand(const T& low, const T& high) {
    return rand() / (static_cast<float>(RAND_MAX) + 1.0) * (high - low) + low;
}
__host__ __device__ float hcuAngle(const float3& a, const float3& b);

template <typename Assignable1, typename Assignable2>
__host__ __device__ inline void hcuSwap(Assignable1& a, Assignable2& b) {
    Assignable1 temp = a;
    a = b;
    b = temp;
}

// Helper Classes
class hcuMat3 {
   public:
    // Constructors
    __host__ __device__ hcuMat3();
    __host__ __device__ hcuMat3(float m11, float m12, float m13, float m21, float m22, float m23, float m31, float m32, float m33);
    __host__ __device__ hcuMat3(const float3& a, const float3& b);
    __host__ __device__ hcuMat3(const float3& a);
    __host__ __device__ static hcuMat3 Identity();
    __host__ __device__ static hcuMat3 Zero();

    // Common Methods
    __host__ __device__ friend hcuMat3 operator*(const float& r, const hcuMat3& a);
    __host__ __device__ friend hcuMat3 operator*(const hcuMat3& a, const hcuMat3& b);
    __host__ __device__ friend hcuMat3 operator+(const hcuMat3& a, const hcuMat3& b);
    __host__ __device__ friend hcuMat3 operator-(const hcuMat3& a, const hcuMat3& b);
    __host__ __device__ friend float3 operator*(const hcuMat3& a, const float3& b);
    __host__ __device__ void Data(float* vec) const;
    __host__ __device__ void Print();
    __host__ __device__ float Det();
    __host__ __device__ hcuMat3 Inverse();
    __host__ __device__ float NormInfty();

    // Getters/Setters
    __host__ __device__ float GetEntry(int opt);

   private:
    // Matrix Entries
    float M11, M12, M13;
    float M21, M22, M23;
    float M31, M32, M33;
};

class hcuMat4 {
   public:
    // Constructors
    __host__ __device__ hcuMat4();
    __host__ __device__ hcuMat4(float m11, float m12, float m13, float m14, float m21, float m22, float m23, float m24, float m31, float m32,
                                float m33, float m34, float m41, float m42, float m43, float m44);
    __host__ __device__ static hcuMat4 Identity();
    __host__ __device__ static hcuMat4 Zero();
    __host__ __device__ static hcuMat4 Scale(const float3& s);
    __host__ __device__ static hcuMat4 Translate(const float3& s);
    __host__ __device__ static hcuMat4 RotateX(const float& theta);
    __host__ __device__ static hcuMat4 RotateY(const float& theta);
    __host__ __device__ static hcuMat4 RotateZ(const float& theta);
    __host__ __device__ static hcuMat4 Rotate(const float& theta, const float3& axis);
    __host__ __device__ static hcuMat4 Transpose(const hcuMat4& A);

    // Common Methods
    __host__ __device__ friend hcuMat4 operator*(const hcuMat4& a, const hcuMat4& b);
    __host__ __device__ friend hcuMat4 operator*(const float& r, const hcuMat4& a);
    __host__ __device__ friend float4 operator*(const hcuMat4& a, const float4& b);
    __host__ __device__ friend float3 operator*(const hcuMat4& a, const float3& b);
    __host__ __device__ void Data(float* vec) const;
    __host__ __device__ void Print();
    __host__ __device__ float Det();
    __host__ __device__ hcuMat4 Inverse();

    // Getters/Setters
    __host__ __device__ float GetEntry(int opt);

   private:
    // Matrix Entries
    float M11, M12, M13, M14;
    float M21, M22, M23, M24;
    float M31, M32, M33, M34;
    float M41, M42, M43, M44;
};
