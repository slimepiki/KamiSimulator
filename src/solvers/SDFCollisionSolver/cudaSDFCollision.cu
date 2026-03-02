#include "cudaSDFCollision.cuh"

__constant__ SDFSimulationParams Params;

static __device__ float4 TriInterpSDF(const int3& idx, const float3& pos, const SDFDeviceBuffers& buffers);
static __device__ float3 SampleNablaSdf(const int3& idx, const SDFDeviceBuffers& buffers);
static __device__ float SampleSDF(const int3& idx, const SDFDeviceBuffers& buffers);
static __device__ float4 SampleTotalSdf(const int3& idx, const SDFDeviceBuffers& buffers);
// static __device__ float3 SampleSDFVel(const int3& idx, const SDFDeviceBuffers& buffers);
// static __device__ float3 TriInterSDFVel(const int3& idx, const float3& pos, const SDFDeviceBuffers& buffers);
static __device__ int3 GetHeadSDFGridIndexFromGlobalPos(const float3& globalPos);
static __device__ float3 GetHeadLocalPosFromGlobalPos(const float3& globalPos);

// Trilinear interpolation
// https://handwiki.org/wiki/Trilinear_interpolation
template <typename T>
static __inline__ __device__ T TriInterpolate(const float3& ud, const T& c000, const T& c001, const T& c010, const T& c011, const T& c100,
                                              const T& c101, const T& c110, const T& c111) {
    T c00 = c000 * (1.f - ud.x) + c100 * ud.x;
    T c01 = c001 * (1.f - ud.x) + c101 * ud.x;
    T c10 = c010 * (1.f - ud.x) + c110 * ud.x;
    T c11 = c011 * (1.f - ud.x) + c111 * ud.x;

    T c0 = c00 * (1.f - ud.y) + c10 * ud.y;
    T c1 = c01 * (1.f - ud.y) + c11 * ud.y;

    T c = c0 * (1.f - ud.z) + c1 * ud.z;

    return c;
}
static __inline__ __device__ int Idx1DSDF(const int3& idx) { return idx.x + idx.y * Params.SDFDim.x + idx.z * Params.SDFDim.x * Params.SDFDim.y; }

static __inline__ __device__ float3 rotate(const float4& q, const float3& v) {  // rotate v by quat q
    // Calculate v * ab
    float3 qim = make_float3(q.x, q.y, q.z);  // imagenary part of q
    float3 a = q.w * v + cross(qim, v);       // The vector
    float c = dot(v, qim);                    // The trivector

    // Calculate (w - q) * (a + c). Ignoring the scaler-trivector parts
    return q.w * a          // The scaler-vector product
           + cross(qim, a)  // The bivector-vector product
           + c * qim;       // The bivector-trivector product
}

static __inline__ __device__ bool CheckInsideHeadGrid(int3 idx) {
    if (idx.x < 0 || idx.x >= Params.SDFDim.x) return false;
    if (idx.y < 0 || idx.y >= Params.SDFDim.y) return false;
    if (idx.z < 0 || idx.z >= Params.SDFDim.z) return false;
    return true;
}

static __global__ void InitHeadVel(SDFDeviceBuffers buffers) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx < 0 || cellIdx > Params.NumSdfCells - 1) return;

    buffers.HeadVelX[cellIdx] = 0.f;
    buffers.HeadVelY[cellIdx] = 0.f;
    buffers.HeadVelZ[cellIdx] = 0.f;
}

void KamiLaunchKernelInitHeadSDFVel(const int2& blockThread, SDFDeviceBuffers buffers) { InitHeadVel<<<blockThread.x, blockThread.y>>>(buffers); }

// static __device__ float3 TriInterSDFVel(const int3& idx, const float3& pos, const SDFDeviceBuffers& buffers) {
//     // Values at surrounding points
//     float3 c000 = SampleSDFVel(idx + make_int3(0, 0, 0), buffers);
//     float3 c001 = SampleSDFVel(idx + make_int3(0, 0, 1), buffers);
//     float3 c010 = SampleSDFVel(idx + make_int3(0, 1, 0), buffers);
//     float3 c011 = SampleSDFVel(idx + make_int3(0, 1, 1), buffers);
//     float3 c100 = SampleSDFVel(idx + make_int3(1, 0, 0), buffers);
//     float3 c101 = SampleSDFVel(idx + make_int3(1, 0, 1), buffers);
//     float3 c110 = SampleSDFVel(idx + make_int3(1, 1, 0), buffers);
//     float3 c111 = SampleSDFVel(idx + make_int3(1, 1, 1), buffers);

//    // Normal coordinates (unit cube)
//    float3 ud;
//    ud.x = Params.SDFInvDs.x * (pos.x - (idx.x - 0.5) * Params.SdfDs.x);
//    ud.y = Params.SDFInvDs.y * (pos.y - (idx.y - 0.5) * Params.SdfDs.y);
//    ud.z = Params.SDFInvDs.z * (pos.z - (idx.z - 0.5) * Params.SdfDs.z);

//    // printf("ud: (%f, %f, %f)\n", ud.x, ud.y, ud.z);

//    // Trilinear interpolation
//    float3 res = TriInterpolate<float3>(ud, c000, c001, c010, c011, c100, c101, c110, c111);

//    return res;
//}

static __device__ int3 GetHeadSDFGridIndexFromGlobalPos(const float3& globalPos) {
    float3 localPos = GetHeadLocalPosFromGlobalPos(globalPos);

    int cellX = (int)(localPos.x * Params.SDFInvDs.x + 0.5);
    int cellY = (int)(localPos.y * Params.SDFInvDs.y + 0.5);
    int cellZ = (int)(localPos.z * Params.SDFInvDs.z + 0.5);

    return make_int3(cellX, cellY, cellZ);
}

inline static __device__ float3 GetHeadLocalPosFromGlobalPos(const float3& globalPos) {
    float4 invq = make_float4(-Params.rotation.x, -Params.rotation.y, -Params.rotation.z, Params.rotation.w);
    return rotate(invq, globalPos - Params.translate) - Params.HeadMin;
}
__device__ float3 SampleNablaSdf(const int3& idx, const SDFDeviceBuffers& buffers) {
    // We asume zero neumann conditions : d\phi/dn = 0
    if (!CheckInsideHeadGrid(idx)) return make_float3(0.f);

    return buffers.NablaSDF[Idx1DSDF(idx)];
}

__device__ float SampleSDF(const int3& idx, const SDFDeviceBuffers& buffers) {
    // We asume zero neumann conditions : d\phi/dn = 0 -> \phi_n - \phi_{n-1} = 0
    // -> \phi_n = \phi_{n-1}

    int x = max(0, min(idx.x, Params.SDFDim.x - 1));
    int y = max(0, min(idx.y, Params.SDFDim.y - 1));
    int z = max(0, min(idx.z, Params.SDFDim.z - 1));

    return buffers.SDF[Idx1DSDF(make_int3(x, y, z))];
}

__device__ float4 SampleTotalSdf(const int3& idx, const SDFDeviceBuffers& buffers) {
    float3 normal = SampleNablaSdf(idx, buffers);
    float phi = SampleSDF(idx, buffers);
    return make_float4(normal.x, normal.y, normal.z, phi);
}

// This function is used if the mesh moves.
__device__ float4 TriInterpSDF(const int3& idx, const float3& pos, const SDFDeviceBuffers& buffers) {
    // Values at surrounding points
    float4 c000 = SampleTotalSdf(idx + make_int3(0, 0, 0), buffers);
    float4 c001 = SampleTotalSdf(idx + make_int3(0, 0, 1), buffers);
    float4 c010 = SampleTotalSdf(idx + make_int3(0, 1, 0), buffers);
    float4 c011 = SampleTotalSdf(idx + make_int3(0, 1, 1), buffers);
    float4 c100 = SampleTotalSdf(idx + make_int3(1, 0, 0), buffers);
    float4 c101 = SampleTotalSdf(idx + make_int3(1, 0, 1), buffers);
    float4 c110 = SampleTotalSdf(idx + make_int3(1, 1, 0), buffers);
    float4 c111 = SampleTotalSdf(idx + make_int3(1, 1, 1), buffers);

    // Normal coordinates (unit cube)
    float3 ud;
    ud.x = Params.SDFInvDs.x * (pos.x - (idx.x - 0.5) * Params.SdfDs.x);
    ud.y = Params.SDFInvDs.y * (pos.y - (idx.y - 0.5) * Params.SdfDs.y);
    ud.z = Params.SDFInvDs.z * (pos.z - (idx.z - 0.5) * Params.SdfDs.z);

    // printf("ud: (%f, %f, %f)\n", ud.x, ud.y, ud.z);
    // Normalize the gradient
    float4 res = TriInterpolate<float4>(ud, c000, c001, c010, c011, c100, c101, c110, c111);
    float3 normal = normalize(make_float3(res.x, res.y, res.z));
    res.x = normal.x;
    res.y = normal.y;
    res.z = normal.z;

    res.w -= Params.ThreshSdf;

    return res;
}

//// This function is used if the mesh moves.
// static __device__ float3 SampleSDFVel(const int3& idx, const SDFDeviceBuffers& buffers) {
//     // We asume zero neumann conditions : d\phi/dn = 0 -> \phi_n - \phi_{n-1} = 0
//     // -> \phi_n = \phi_{n-1}
//     if (!CheckInsideHeadGrid(idx)) return make_float3(0, 0, 0);

//    int x = max(0, min(idx.x, Params.SDFDim.x - 1));
//    int y = max(0, min(idx.y, Params.SDFDim.y - 1));
//    int z = max(0, min(idx.z, Params.SDFDim.z - 1));

//    float vx = buffers.HeadVelX[Idx1DSDF(make_int3(x, y, z))];
//    float vy = buffers.HeadVelY[Idx1DSDF(make_int3(x, y, z))];
//    float vz = buffers.HeadVelZ[Idx1DSDF(make_int3(x, y, z))];

//    return make_float3(vx, vy, vz);
//}

void KamiCopySDFSimParamsToDevice(SDFSimulationParams* hostParams) { cudaMemcpyToSymbol(Params, hostParams, sizeof(SDFSimulationParams)); }

__global__ void StrainLimiting(SDFDeviceBuffers buffers) {
    int rIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rIdx == 0) printf("strainLimiting!\n");
    if (rIdx < 0 || rIdx > Params.NumStrands - 1) return;

    // Parallel over roots/strands
    ParticleForSDFColl& p = buffers.Particles[buffers.rootIdx[rIdx]];
    int rootIdx = p.GlobalIdx;

    // Marches from the root
    for (int i = 0; i < p.StrandLength - 1; i++) {
        // Geometric info
        ParticleForSDFColl& pCurrent = buffers.Particles[rootIdx + i];
        ParticleForSDFColl& pNext = buffers.Particles[rootIdx + i + 1];
        float& l0 = buffers.restLenghts[pCurrent.EdgeRestIdx.y];
        float l = length(pNext.Position - pCurrent.Position);

        // Distance constraint
        if (abs(l - l0) > Params.StrainError * l0) {
            float3 dir = normalize(pCurrent.Position - pNext.Position);
            pNext.Position += (l - l0) * dir;
        }
    }
}

void KamiLaunchKernelStrainLimitingAfterSDFCollision(const int2& blockThread, SDFDeviceBuffers buffers) {
    StrainLimiting<<<blockThread.x, blockThread.y>>>(buffers);
}

static __global__ void SDFHeadCollision(SDFDeviceBuffers buffers) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx < 0 || pIdx > Params.NumParticles - 1) return;

    // if (pIdx == 0) {
    //     // AABBmin: (-0.116027, 1.548379, -0.120367), AABBmax: (0.113609, 1.887230, 0.136588)
    //     int3 lcidxmn = GetNearestHeadSDFGridIndexFromGlobalPos(make_float3(-0.116027, 1.548379, -0.120367));
    //     int3 lcidxmx = GetNearestHeadSDFGridIndexFromGlobalPos(make_float3(0.113609, 1.887230, 0.136588));

    //    printf("(%d, %d, %d)\n", lcidxmn.x, lcidxmn.y, lcidxmn.z);
    //}

    //// Mesh vel., particle vel., normal, and sdf
    // localPos = GetHeadLocalPosFromGlobalPos(pRef.Position);

    // headGridIdx = GetHeadSDFGridIndexFromGlobalPos(pRef.Position);
    //  float3 meshVel = TriInterSDFVel(headGridIdx, localPos, buffers);
    //  float3 globalPVel = pRef.Velocity;

    //// Decompose the velocity into normal and tangential direction
    // float3 vMesh_n = dot(meshVel, normal) * normal;
    // float3 vMesh_t = meshVel - vMesh_n;
    // float3 vPart_n = dot(globalPVel, normal) * normal;
    // float3 vPart_t = globalPVel - vPart_n;

    //// Vel correction
    // float rel = length(vPart_n - vMesh_n) / length(vPart_t - vMesh_t);
    // float3 velNew = meshVel + fmax(0.f, 1 - Params.Friction * rel) * (vPart_t - vMesh_t);

    //// Updates
    // pRef.Velocity = velNew;
    // pRef.Position += Params.Dt * pRef.Velocity;

    // Second stage
    // Check particle's index within the box
    ParticleForSDFColl& pRef = buffers.Particles[pIdx];  // a particle ref
    if (pRef.LocalIdx == 0) return;                      // root of the strand

    int3 headGridIdx = GetHeadSDFGridIndexFromGlobalPos(pRef.Position);
    float3 localPos = GetHeadLocalPosFromGlobalPos(pRef.Position);

    if (!CheckInsideHeadGrid(headGridIdx)) return;

    // Otherwise, gets (via trilinear interp.) SDF and normal vector
    TriInterpSDF(headGridIdx, localPos, buffers);

    // if the particle is outside box, do nothing
    if (CheckInsideHeadGrid(headGridIdx)) {
        // Otherwise, gets (via trilinear interp.) SDF and normal vector
        float4 sdf = TriInterpSDF(headGridIdx, localPos, buffers);
        float3 normal = make_float3(sdf.x, sdf.y, sdf.z);
        float phi = sdf.w;

        //   Positive SDF -> Do nothing
        if (phi < 0) {
            pRef.Position -= phi * normal;
            if (length(pRef.Velocity) >= 1e-6) {
                float3 pVelNorm = pRef.Velocity / length(pRef.Velocity);
                pRef.Velocity -= dot(pVelNorm, normal) * normal;
            }
            return;
        }
    }
}

void KamiLaunchKernelSDFHeadCollision(const int2& blockThread, SDFDeviceBuffers buffers) {
    SDFHeadCollision<<<blockThread.x, blockThread.y>>>(buffers);
}