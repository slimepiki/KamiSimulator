#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../../utilities/cudaMath.cuh"

struct ParticleForSDFColl {
    ParticleForSDFColl() {}

    // Dynamic info
    float3 Position = make_float3(0.f);  // params at t+dt
    float3 Velocity = make_float3(0.f);

    int2 EdgeRestIdx = {-1, -1};
    int2 BendRestIdx = {-1, -1};
    int2 TorsRestIdx = {-1, -1};

    // Geometry info
    int StrandLength = 0;
    int LocalIdx = 0;
    int GlobalIdx = 0;
};

struct SDFSimulationParams {
    float Dt;        // time step (s)
    float InvDt;     // inverse time step (s^{-1})
    float Friction;  // friction coef. of head

    // System dimensions
    size_t NumStrands;       // number of hair strands
    size_t MaxStrandLength;  // used for uniform exporting
    size_t NumParticles;     // number of particles in hair discretization

    // Sdf Grid
    int3 SDFDim;         // grid dimensions (nx, ny, nz)
    float3 SdfDs;        // grid step sizes (dx, dy, dz)
    float3 SDFInvDs;     // inverse step sizes (1/dx, 1/dy, 1/dz)
    float3 SdfInvSqDs;   // inverse squared step sizes (1/dx^2,1/dy^2,1/dz^2)
    size_t NumSdfCells;  // number of grid voxels for sdf
    float ThreshSdf;     // small threshold for head collision
    // float3 HeadAxis[3], HeadAxis0[3];   // grid orientation (e_x, e_y, e_z)
    float3 HeadMin, HeadMax;  // grid global (Xmin,Ymin,Zmin) position

    float4 rotation;  // rotation quaternion (x,y,z,w)
    float3 translate;

    float StrainError;
    bool UseStrainLimiting;
};

struct SDFDeviceBuffers {
    hcuMat4* RigidMotion;

    // Hair
    ParticleForSDFColl* Particles;
    float* restLenghts;
    int* rootIdx;

    float* SDF;
    float3* NablaSDF;

    // These buffers is used if the mesh moves.
    float* HeadVelX;
    float* HeadVelY;
    float* HeadVelZ;
    float* HeadVelWeight;
};

void KamiCopySDFSimParamsToDevice(SDFSimulationParams* hostParams);
void KamiLaunchKernelInitHeadSDFVel(const int2& blockThread, SDFDeviceBuffers buffers);
void KamiLaunchKernelStrainLimitingAfterSDFCollision(const int2& blockThread, SDFDeviceBuffers buffers);
void KamiLaunchKernelSDFHeadCollision(const int2& blockThread, SDFDeviceBuffers buffers);