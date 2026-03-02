
// Jerry Hsu, jerry.hsu.research@gmail.com, 2025

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"
#include "../../KittenEngine/includes/modules/SymMat.h"

// reference:Vertex Block Descent(VBD)( https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf )
// reference:Stable Cosserat Rods(SCR)( https://jerryhsu.io/wp-content/uploads/2025/05/Sig25__Stable_Cosserat_Rods.pdf )

// const static double r2d = 57.2957795130823208767981548141051703324;
// const static double myPi = 3.141592653589793238462643;
// const static double my2Pi = 6.2831853071795864769252867665590057683943;
// const static double myPi2 = 1.5707963267948966192313216916397;

namespace YarnBall {
using Kit::hess3;

// The main Cosserat iteration is split into sectors because we are compute bound when computing collisions
// The whole block is divided into THREADS_PER_VERTEX sectors.
// Each sector computes a portion of the collision energy and this is summed up into sector 0
// Sector 0 then performs the actual update
__device__ float GetPreciseLambda(vec4 b, vec3 v, uint32_t itr = 0);
__global__ void cosseratItr(MetaData* data) {
    // Sector id (1D Grid) ∈{0, ..., THREADS_PER_VERTEX}
    // sid: 0, 0, 0, 0, ..., 1, 1, 1
    const int sid = threadIdx.x / VERTEX_PER_BLOCK;
    // Local thread id (1D Block)
    // ltid: 0, 1, 2, ..., VERTEX_PER_BLOCK - 1, 0, 1, 2, ..., VERTEX_PER_BLOCK - 1
    const int ltid = threadIdx.x - sid * VERTEX_PER_BLOCK;
    // Global vertex id (1D Block)
    // vid (block 0): -1, 0, 1, ..., VERTEX_PER_BLOCK - 2,
    //     (block 1): VERTEX_PER_BLOCK - 2, VERTEX_PER_BLOCK - 1, ..., 2 * VERTEX_PER_BLOCK - 3
    //     (block 2): 2 * VERTEX_PER_BLOCK - 3, 2 * VERTEX_PER_BLOCK - 2, ...,  3 * VERTEX_PER_BLOCK - 4
    // ...
    const int vid = (int)(blockIdx.x * (VERTEX_PER_BLOCK - 1) + ltid) - 1;

    const int numVerts = data->numVerts;
    if (vid >= numVerts || vid < 0) return;

    const float h = data->h;  // time step, denoted as h in the paper

    const float damping = data->damping / h;  // Damping forces
    const auto lastPos = data->d_lastPos;     // Last vertex positions. Temp storage to speed up memory access.
    const auto dxs = data->d_dx;              // Temporary delta position iterants. Stored as deltas for precision.

    Vertex vi = data->d_verts[vid];  // current vertex
    vec3 vi_dx = dxs[vid];           // the suggested displacement

    // Quaternion modification from PaO Eq. (37)
    // if (sid == 0 && vi.flags & (uint32_t)VertexFlags::hasNext) {
    //     const float eps = 1.0e-6;
    //     auto vi1 = data->d_verts[vid + 1];
    //     auto qs = data->d_qs;
    //     auto qi = qs[vid];

    //    // Vector3r d3;                                                                  // third director d3 = q0 * e_3 * q0_conjugate
    //    // d3[0] = static_cast<Real>(2.0) * (q0.x() * q0.z() + q0.w() * q0.y());
    //    // d3[1] = static_cast<Real>(2.0) * (q0.y() * q0.z() - q0.w() * q0.x());
    //    // d3[2] = q0.w() * q0.w() - q0.x() * q0.x() - q0.y() * q0.y() + q0.z() * q0.z();
    //    // ↓
    //    auto D3 = qi.getD3();

    //    // Vector3r gamma = (p1 - p0) / restLength - d3;
    //    // ↓
    //    auto gamma = (vi1.pos - vi.pos) / vi.lRest - D3;

    //    float invMassq0 = 1.0 / (vi.lRest * (data->radius) * (data->radius) * myPi * data->density * 0.5f);

    //    // gamma /= (invMass1 + invMass0) / restLength + invMassq0 * static_cast<Real>(4.0) * restLength + eps;
    //    // ↓
    //    gamma /= (vi1.invMass + vi.invMass) / vi.lRest + invMassq0 * 4.0f * vi.lRest + eps;

    //    // if (std::abs(stretchingAndShearingKs[0] - stretchingAndShearingKs[1]) < eps &&
    //    //     std::abs(stretchingAndShearingKs[0] - stretchingAndShearingKs[2]) < eps)  // all Ks are approx. equal
    //    // ↓
    //    if (std::abs(vi1.kStretch - vi.kStretch) < eps) {
    //        // for (int i = 0; i < 3; i++) gamma[i] *= stretchingAndShearingKs[i];
    //        // std::abs(stretchingAndShearingKs[0] - stretchingAndShearingKs[2]) < eps)  // all Ks are approx. equal
    //        // ↓
    //        gamma *= vi.kStretch;
    //    } else {
    //        // diffenent stretching and shearing Ks. Transform diag(Ks[0], Ks[1], Ks[2]) into world space using Ks_w = R(q0) * diag(Ks[0],
    //        // Ks[1],Ks[2]) * R^T(q0) and multiply it with gamma

    //        // Matrix3r R = q0.toRotationMatrix();
    //        // ↓
    //        auto R = qi.matrix();

    //        // gamma = (R.transpose() * gamma).eval();
    //        // for (int i = 0; i < 3; i++) gamma[i] *= stretchingAndShearingKs[i];
    //        // gamma = (R * gamma).eval();
    //        // ↓
    //        gamma = glm::transpose(R) * gamma;
    //        gamma *= vi.kStretch;
    //        gamma = R * gamma;
    //    }

    //    // Quaternionr q_e_3_bar(q0.z(), -q0.y(), q0.x(), -q0.w());  // compute q*e_3.conjugate (cheaper than quaternion product)
    //    // corrq0 = Quaternionr(0.0, gamma.x(), gamma.y(), gamma.z()) * q_e_3_bar;
    //    // corrq0.coeffs() *= static_cast<Real>(2.0) * invMassq0 * restLength;
    //    // ↓
    //    Kit::Rotor q_e_3_bar(-qi.s, qi.z, -qi.y, qi.x);
    //    auto corrq0 = Kit::Rotor(gamma.x, gamma.y, gamma.z, 0.0) * q_e_3_bar;
    //    corrq0 = corrq0 * 2.0f * invMassq0 * vi.lRest;

    //    qs[vid] += corrq0;
    //    qs[vid] = Kit::Rotor(glm::normalize(qs[vid].v));
    //}

    hess3 H(0);  // Hessian
    vec3 f(0);   // force

    if (!sid) {  // if in the sector 0

        H = hess3(1 / (vi.invMass * h * h));  // Hessian H see VBD Eq. (9)

        // see VBD Eq. (8)
        // vel has been overwritten to contain y - pos
        f = 1 / (h * h * vi.invMass) * (data->d_vels[vid] - vi_dx);

        // Special connections energy (distance constraint energy?)
        if (vi.connectionIndex >= 0) {
            constexpr float stiffness = 4e1;
            vec3 vj_pos0 = lastPos[vi.connectionIndex];  // last pos of connected vert
            vec3 vj_dx = dxs[vi.connectionIndex];        // connected vert's dx

            f -= stiffness * ((vi.pos - vj_pos0) + (vi_dx - vj_dx) + damping * vi_dx);
            H.data.diag += stiffness * (1 + damping);
        }
    }

    // We need to store absolute position and position updates seperatly for floating point precision
    // If we added these together, the update could be small enough to be rounded out, causing stability issues

    float stepLimit = INFINITY;
    vec3 f2(0);
    hess3 H2(0);

    if (vi.flags & (uint32_t)VertexFlags::hasNext) {
        vec3 vj_p0 = lastPos[vid + 1];  // last pos. of connected vert
        vec3 vj_dx = dxs[vid + 1];      // connected vert's dx

        // Cosserat stretching energy: see SCR Eq. (28) and Eq. (29)
        if (!sid) {  // if in the sector 0
            stepLimit = data->d_maxStepSize[vid];

            float invl = 1 / vi.lRest;
            // vec3 c = ((vj_p0 + vj_dx) - (vi.pos + vi_dx)) * invl - data->d_qs[vid] * vec3(1, 0, 0);  // vector part of SCR Eq. (28)
            vec3 c = ((vj_p0 - vi.pos) + (vj_dx - vi_dx)) * invl - data->d_qs[vid] * glm::vec3(1, 0, 0);
            // printf("Eq. (28): vert %d\t c %f, invl %f\n", vid, length(c), length(invl));

            float k = vi.kStretch * invl;
            float d = k * invl;
            f += k * c - (damping * d) * vi_dx;    // the force for the v_i
            f2 += -k * c - (damping * d) * vj_dx;  // the force for the v_j
            d *= 1 + damping;
            H.data.diag += d;   // Hessian for the v_i: see Eq. (29)
            H2.data.diag += d;  // Hessian for the v_j: see Eq. (29)
        }

        const float fricK = data->kFriction;
        const float invb = 1 / data->barrierThickness;
        const float radius = 2 * data->radius;
        const float fricMu = data->frictionCoeff;
        const auto collisions = data->d_collisions;
        const float kCol = data->kCollision * invb;

        // Collision energy of this segment: see VBD Sec. 4
        const int numCols = data->d_numCols[vid];
        for (int i = sid; i < numCols; i += THREADS_PER_VERTEX) {
            int colID = collisions[vid + i * numVerts];

            vec3 b0 = lastPos[colID];
            vec3 b1 = lastPos[colID + 1];
            vec3 db0 = dxs[colID];
            vec3 db1 = dxs[colID + 1];

            // Compute collision UV and normal
            vec2 uv =
                Kit::segmentClosestPoints(vec3(0), (vj_p0 - vi.pos) + (vj_dx - vi_dx), (b0 - vi.pos) + (db0 - vi_dx), (b1 - vi.pos) + (db1 - vi_dx));

            vec3 dpos = mix(vi.pos, vj_p0, uv.x) - mix(b0, b1, uv.y);
            vec3 ddpos = mix(vi_dx, vj_dx, uv.x) - mix(db0, db1, uv.y);
            vec3 normal = dpos + ddpos;
            float d = length(normal);
            normal *= (1 / d);

            uv.y = uv.x;
            uv.x = 1 - uv.x;

            // Compute penetration
            d = d - radius;
            d *= invb;
            if (d > 1) continue;  // Not touching
            d = max(d, 1e-3f);    // Clamp to some small value. This is a ratio of the barrier thickness.

            // IPC barrier energy
            float invd = 1 / d;
            float logd = log(d);

            float dH = (-3 + (2 + invd) * invd - 2 * logd) * kCol * invb;
            float ff = -(1 - d) * (d - 1 + 2 * d * logd) * invd * kCol;
            f += (ff * uv.x - damping * dH * uv.x * uv.x * dot(normal, vi_dx)) * normal;
            f2 += (ff * uv.y - damping * dH * uv.y * uv.y * dot(normal, vj_dx)) * normal;

            dH *= 1 + damping;
            hess3 op = hess3::outer(normal);
            H += op * (dH * uv.x * uv.x);
            H2 += op * (dH * uv.y * uv.y);

            // Friction
            vec3 u = ddpos - dot(normal, ddpos) * normal;
            float ul = length(u);
            if (ul > 0) {
                float f1 = glm::min(fricK, fricMu * ff / ul);

                op.data.diag -= 1;

                f -= f1 * uv.x * u;
                H -= op * (Kit::pow2(uv.x) * f1);

                f2 -= f1 * uv.y * u;
                H2 -= op * (Kit::pow2(uv.y) * f1);
            }
        }
    }

    __shared__ float sharedData[18 * VERTEX_PER_BLOCK];

    // Reduce forces to the lower threads
    vec3* f0s = (vec3*)sharedData;
    vec3* f1s = (vec3*)(sharedData + 3 * VERTEX_PER_BLOCK);
    hess3* h0s = (hess3*)(sharedData + 6 * VERTEX_PER_BLOCK);
    hess3* h1s = (hess3*)(sharedData + 12 * VERTEX_PER_BLOCK);

    if (sid) {  // if in the sector 1
        f0s[ltid] = f;
        f1s[ltid] = f2;
        h0s[ltid] = H;
        h1s[ltid] = H2;
    }
    __syncthreads();

    if (!sid) {  // if in the sector 0
        f += f0s[ltid];
        f2 += f1s[ltid];
        H += h0s[ltid];
        H2 += h1s[ltid];
    }
    __syncthreads();

    // Sum forces across the yarn segments
    if (!sid) {  // if in the sector 0

        // Reuse of the shared memory
        vec4* forces = (vec4*)sharedData;
        hess3* hessians = (hess3*)(sharedData + 4 * VERTEX_PER_BLOCK);

        forces[threadIdx.x] = vec4(f2, stepLimit);
        hessians[threadIdx.x] = H2;

        __syncthreads();

        // No reason to keep thread 0 going anymore
        if (!threadIdx.x) return;

        if (vi.flags & (uint32_t)VertexFlags::hasPrev) {
            vec4 v = forces[threadIdx.x - 1];
            stepLimit = min(stepLimit, v.w);
            f += vec3(v);
            H += hessians[threadIdx.x - 1];
        }

        if (vi.invMass != 0) {
            // Local solve
            vec3 delta = data->accelerationRatio * (inverse((mat3)H) * f);
            vi_dx += delta;

            float l = length(vi_dx);
            if (l > stepLimit && l > 0) vi_dx *= stepLimit / l;

            // Apply update
            dxs[vid] = vi_dx;
        }
    }
}

__global__ void quaternionLambdaItr(MetaData* data) {
    // vertex id
    const int vid = threadIdx.x + blockIdx.x * blockDim.x;
    const int numVerts = data->numVerts;
    if (vid >= numVerts || vid < 0) return;

    const auto verts = data->d_verts;
    const auto dxs = data->d_dx;

    Vertex vi = verts[vid];  // current vertex

    // Update segment orientation
    // This is done assuming some very very large invMoment (i.e. no inertia so static equilibrium)
    if (!(bool)(vi.flags & (uint32_t)VertexFlags::fixOrientation) != 0 && (vi.flags & (uint32_t)VertexFlags::hasNext)) {
        vec3 vi_dx = dxs[vid];             // the suggested displacement
        vec3 vj_pos = verts[vid + 1].pos;  // connected vert's position
        vec3 vj_dx = dxs[vid + 1];         // connected vert's dx

        // from SCR Eq. (15)
        // vi.pos = ((vj_pos + vj_dx) - (vi.pos + dx)) / vi.lRest;
        vi.pos = ((vj_pos - vi.pos) + (vj_dx - vi_dx)) / vi.lRest;
        vi.pos *= -2 * vi.kStretch;

        vec4 b(0);
        auto qs = data->d_qs;
        auto qRests = data->d_qRests;
        auto vi_q = qs[vid];
        if (vi.flags & (uint32_t)VertexFlags::hasPrev) {
            auto qRest = Kit::Rotor(qRests[vid - 1]);
            auto vj_q = qs[vid - 1];                                           // v_{i-1}
            float phi = dot((vj_q.inverse() * vi_q).v, qRest.v) > 0 ? 1 : -1;  // from SCR Eq. (4)
            b += phi * (vj_q * qRest).v;                                       // from SCR Eq. (16)
        }

        if (vi.flags & (uint32_t)VertexFlags::hasNextOrientation) {
            auto qRest = Kit::Rotor(qRests[vid]);
            auto vj_q = qs[vid + 1];  // v_{i+1}

            float phi = dot((vi_q.inverse() * vj_q).v, qRest.v) > 0 ? 1 : -1;  // from SCR Eq. (4)
            b += phi * (vj_q * qRest.inverse()).v;                             // from SCR Eq. (16)
        }

        float lambda = length(vi.pos) + length(b);  // Eq. (22)
        // Eq. (18)
        vi_q = Kit::Rotor(normalize((Kit::Rotor(vi.pos) * Kit::Rotor(b) * Kit::Rotor(glm::vec3(1, 0, 0))).v + lambda * b));
        qs[vid] = vi_q;
    }
}

[[maybe_unused]]
__device__ float GetPreciseLambda(vec4 b, vec3 v, uint32_t itr) {
    auto lambda = length(v) + length(b);
    auto gamma = (lambda - length(v)) / length(b);

    for (uint32_t i = 0; i < itr; ++i) {
        lambda = length(v) + clamp(gamma, 1e-3f, 1.f) * length(b);
        lambda = sqrt(length((Kit::Rotor(v) * Kit::Rotor(b) * Kit::Rotor(glm::vec3(1, 0, 0))).v) + length2(v));
        gamma = (lambda - length(v)) / length(b);
    }
    return lambda;
}

// from Position and Orientation Based Cosserat Rods(PaO) ( https://animation.rwth-aachen.de/media/papers/2016-SCA-Cosserat-Rods.pdf )
__global__ void PBDQuat(MetaData* data) {
    const int vid = threadIdx.x + blockIdx.x * blockDim.x;
    const int numVerts = data->numVerts;
    if (vid >= numVerts || vid < 0) return;
    const auto verts = data->d_verts;
    auto vi = verts[vid];  // current vertex

    if (vi.flags & (uint32_t)VertexFlags::hasNext) {
        auto qs = data->d_qs;
        auto qRests = data->d_qRests;

        auto vi1 = verts[vid + 1];

        auto vi_q = qs[vid];       // current quaternion
        auto vi1_q = qs[vid + 1];  // next quaternion
        auto qRest = Kit::Rotor(qRests[vid]);

        auto bendk = glm::length(qRest.v);
        auto omega = (vi_q.inverse() * vi1_q).v;
        float s = dot(omega, qRest.v) > 0 ? 1 : -1;  // from PaO Eq.(33) or SCR Eq.(4)

        // from PaO Eq.(40)
        auto delta = (omega - s * qRest.v) * bendk / (vi.invMass + vi1.invMass + (float)1.0e-6);
        auto deltaq = Kit::Rotor(delta);

        qs[vid] += (vi.invMass) * vi1_q * deltaq;
        qs[vid + 1] += (-vi1.invMass) * vi_q * deltaq;

        qs[vid] = Kit::Rotor(glm::normalize(qs[vid].v));
        qs[vid + 1] = Kit::Rotor(glm::normalize(qs[vid + 1].v));
    }
}

}  // namespace YarnBall