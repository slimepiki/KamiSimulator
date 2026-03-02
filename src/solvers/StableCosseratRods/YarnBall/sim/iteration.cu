#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"

namespace YarnBall {
// Converts velocity to initial guess
// x <- x^t + h v^t + h^2a from the Algorithm 2 in the paper.
__global__ void initItr(MetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= data->numVerts) return;

    const float h = data->h;
    auto verts = data->d_verts;
    auto lastVels = data->d_lastVels;

    const vec3 g = data->gravity;
    const vec3 vel = data->d_vels[tid];

    const float ypos = verts[tid].pos.y;
    const float zpos = verts[tid].pos.z;

    const float windYVal = ypos > 0 ? cos(2 * M_PI * (data->WindYFreq * ypos + data->windTimeFreq * data->time)) : 0;
    const float windZVal = cos(2 * M_PI * (data->WindZFreq * abs(zpos) + data->windTimeFreq * data->time));

    float windx = data->windPeak * max(0.f, pow(windYVal * windZVal, data->windSharpness) * sign(windYVal * windZVal));

    const vec3 wind = vec3(windx, 0, 0);

    vec3 ext = (g + wind) * data->ExternalForceMultiplier;

    // if (!tid) printf("minseglen: %f, maxseglen: %f\n", data->minSegLen, data->maxSegLen);

    vec3 dx = h * vel;
    // if (tid == 124) printf("dx: %.15f\n", glm::length(dx));
    vec3 lastVel = lastVels[tid];
    lastVels[tid] = vel;
    [[maybe_unused]]
    float stepLimit = INFINITY;

    if (verts[tid].invMass != 0) {
        // Compute y (inertial + accel position)
        // Store it in vel (The actual vel is no longer needed)
        data->d_vels[tid] = dx + (h * h) * ext;

        // Compute initial guess described at Vertex Block Descent(2024) Sec. 3.7 (d)
        float ext2 = length2(ext);
        // if the external force is exerted
        if (ext2 > 0) {
            vec3 a = (vel - lastVel) / data->lastH;
            float s = clamp(dot(a, ext) / ext2, 0.f, 1.f);
            dx += (h * h * s) * ext;
        }
    }
    data->d_dx[tid] = dx;

    // Transfer segment data
    vec3 pos = verts[tid].pos;
    data->d_lastPos[tid] = pos;
}

// Converts dx back to velocity and advects
__global__ void endItr(MetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= data->numVerts) return;

    const float h = data->h;
    const float invH = 1 / h;
    auto verts = data->d_verts;

    // Linear velocity
    vec3 dx = data->d_dx[tid];
    if (verts[tid].invMass != 0) data->d_vels[tid] = dx * invH * (1 - data->velocityDecay * h);
    verts[tid].pos += dx;
}
}  // namespace YarnBall