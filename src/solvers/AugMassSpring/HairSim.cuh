#pragma once

#include <string>
#include <cuda_runtime.h>
#include "SimpleTriangle.cuh"

// Experiment Option (select between experiments)
// enum ExpOpt {
//    CANTI_E,
//    CANTI_L,
//    SAG_WIND,
//    SAG_HEAD,
//    SANDBOX,
//    WIG,
//    ROLLER,
//    SALON,
//    KATE,
//    CURLY,
//    SPACE,
//    FACIAL,
//    MODERN,
//    CHENGAN,
//    SICTION,
//    ANIMATION,
//    NONE
//};

enum AnimType { SEQUENCE, WIND };

// Type of cell for fluid solver
// Only FLUID is available.
enum CellType { FLUID, AIR, SOLID };

// Wrapers for loading assets
struct HairInfo {
    // Basic information
    std::string FilePath;
    float4 Move;
    // ↓I think this parameter should be managed by collision solver.
    // int3 EulerDim;
    int ID = -1;
};

struct HeadInfo {
    // Basic information
    std::string FilePath;
    float4 Move;
    int3 SDFDim;
    int ID = -1;

    // Animation data
    int NumFrames = 0;
    bool Animate = false;
};

struct InterParticle {
    int2 VboIdx;
    int2 InterIdx;
    float Lambda;
};

struct Particle {
    Particle(float3 position, int strandLength, int localIdx) : Position(position), StrandLength(strandLength), LocalIdx(localIdx) {}
    Particle() {}

    // Dynamic info
    float3 Position = make_float3(0.f);  // params at t+dt
    float3 Velocity = make_float3(0.f);

    float3 InitialPosition = make_float3(0.f);  // position before dynamics
    float3 Position0 = make_float3(0.f);        // position at t

    // Geometry info
    int StrandLength = 0;
    int LocalIdx = 0;
    int GlobalIdx = 0;

    // Indices for finding rest length of one-phase springs
    // (left and right, negative value means not connected)
    int2 EdgeRestIdx = {-1, -1};
    int2 BendRestIdx = {-1, -1};
    int2 TorsRestIdx = {-1, -1};

    // Biphasic Interaction
    float3 GravityPos = make_float3(0.f);
    float3 GravityPos0 = make_float3(0.f);
    float3 Angular = make_float3(0.f);

    // Cut Interaction
    bool Cut = false;
    int CutParent = -1;

    // Grab Interaction
    // bool BeingPulled = false;
    // float GrabL0 = 0.f;

    // Indices for mapping into OpenGL's VBO
    int2 VboIdx = {-1, -1};
};

struct SimulationParams {
    // Rigid head
    float3 RigidPos;
    float3 RigidAngle;

    // int CutMin;

    // System dimensions
    size_t MaxStrandLength;  // used for uniform exporting
    size_t NumVertices;      // number of vertices in main mesh
    size_t NumTriangles;     // number of triangles in main mesh
    size_t NumParticles;     // number of particles in hair discretization
    size_t NumInter;         // number of iterpolated particles
    size_t NumSprings;       // number of springs
    size_t NumRoots;         // number of hair strands/roots
    size_t NumGridCells;     // number of Eulerian grid voxels
    size_t NumGridCellsU;    // number of grid voxels for staggered Eulerian grids
    size_t NumGridCellsV;
    size_t NumGridCellsW;

    float LengthIncreaseGrid;  // percentage of grid increase w.r.t. loaded obj

    // Eulerian hair
    int3 HairDim;                      // grid dimensions (nx,ny,nz)
    float3 HairDs;                     // grid step sizes (dx,dy,dz)
    float3 HairInvDs;                  // inverse step sizes (1/dx, 1/dy, 1/dz)
    float3 HairInvSqDs;                // inverse squared step sizes (1/dx^2,1/dy^2,1/dz^2)
    float3 HairAxis[3], HairAxis0[3];  // grid orientation (e_x, e_y, e_z)
    float3 HairMin, HairMin0;          // grid global (Xmin,Ymin,Zmin) position
    float3 HairCenter, HairCenter0;    // grid center

    // Sdf Grid
    int3 SDFDim;                       // grid dimensions (nx, ny, nz)
    float3 SdfDs;                      // grid step sizes (dx, dy, dz)
    float3 SDFInvDs;                   // inverse step sizes (1/dx, 1/dy, 1/dz)
    float3 SdfInvSqDs;                 // inverse squared step sizes (1/dx^2,1/dy^2,1/dz^2)
    size_t NumSdfCells;                // number of grid voxels for sdf
    float ThreshSdf;                   // small threshold for head collision
    float3 HeadAxis[3], HeadAxis0[3];  // grid orientation (e_x, e_y, e_z)
    float3 HeadMin, HeadMin0;          // grid global (Xmin,Ymin,Zmin) position
    float3 HeadCenter, HeadCenter0;    // grid center

    // PIC/FLIP parameters
    int NumGridNeighbors;  // number of neighbors to look for in particle2grid routine
    int NumStepsJacobi;    // number of iterations for Jacobi pressure solver
    int Parity;            // Sdf parity (to avoid buggs if mesh was inverted)
    float MaxWeight;       // maximum node weight
    float JacobiWeight;    // (0,1] weight in Jacobi solver
    float FlipWeight;      // (0,1) control of FLIP-PIC grid2particle weight

    // Physical Parameters
    float WindPeak;
    float3 WindSpeed;  // wind intensity (m/s)
    float Dt;          // time step (s)
    float DtN;         // nested time step (s)
    float InvDt;       // inverse time step (s^{-1})
    float Gravity;     // gravity constant (m/s^2)
    float EdgeK;       // spring constants (N/m)
    float BendK;
    float TorsionK;
    float AngularK;  // (N/rad)
    float GravityK;  // (N/m)
    float Damping;   // particle damping
    float HairMass;  // mass (kg)
    float Mu;        // inverse hair mass (kg^{-1})
    float Friction;  // friction coef. of head

    // User Interaction
    // bool ToogleCutHair;
    // bool ToogleGrabHair;
    float CutRadius;  // radious of sphere for cutting hair
    // float GrabRadius; // radius of cylinder for grabbing hair
    // float GrabK; // spring constant for grabbing
    // float3 GrabPos;

    // AMSSolver Parameters
    // int StrainSteps;
    float StrainError;
    size_t NestedSteps;  // This parameter should be separated from SimulationParams.

    // Export Options
    // bool SaveSelection;
    // bool LoadSelection;

    // Preprocessing options
    // bool TooglePreProcess;
    // bool RecomputeSdf;

    // Experiments manuscript
    // ExpOpt Experiment;

    //// Import/Export
    // int ExportStep;  // to export at every i-th frame

    // Animation
    bool MeshSequence;   // tracks if there is a loaded mesh sequence
    float SimEndTime;    // stop animation here
    float SimTime;       // 'inside' simulation time
    float AnimDuration;  // time duration of animation
    int NumFrames;       // number of frames for mesh sequence
    float3 RootMove;

    // Toogle Options
    AnimType AnimationType;
    bool Animate;  // start prescribed animatiomn
    // bool PauseSim;  // stop iterating simulation

    // Toogle export
    bool FixedHairLength;  // export strands of equal size
    // bool ExportVDB;        // export SDF in volumetric format

    // Toogle Debug
    bool SolidCollision;
    bool HairCollision;
    bool DrawHead;
    bool DrawHeadWF;
};

struct GenerationParams {
    // Interpolation
    int NumInterpolated;
    bool Interpolate;

    // Procedural Parameters
    // float StepLength;
    // int StepsMin;
    // int StepsMax;
    // int HairsPerTri;
    // float HairThickness;
    // float GravityInfluence;
    // float DirNoise;
    // float GravNoise;
    // float GravityDotInfluence;
    // float SpiralRad;
    // float FreqMult;
    // float SpiralAmount;
    // float SpiralY;
    // float SpiralImpact;
    // float PartingImpact;
    // float PartingStrengthX;
    // float PartingStrengthY;

    // Loader options
    size_t MaxLoadStrandSize;
    size_t MaxLoadNumStrands;
};

struct DeviceBuffers {  // This struct holds device pointers for the array in cuda. The sizes of the array are stored at SimulationParams.
    // Interpolation
    InterParticle* InterParticles;

    // Other
    hcuMat4* RigidMotion;

    // Hair
    Particle* Particles;
    float* RestLenghts;
    int* RootIdx;

    // AMSSolver
    hcuMat3* StrandA;
    hcuMat3* StrandL;
    hcuMat3* StrandU;
    float3* StrandV;
    float3* StrandB;

    // Eulerian
    CellType* VoxelType;
    float2* HairPressure;
    float* HairVelU;
    float* HairVelV;
    float* HairVelW;
    float* HairPicU;
    float* HairPicV;
    float* HairPicW;
    float* HairDiv;
    float* HairWeightU;
    float* HairWeightV;
    float* HairWeightW;

    // Head
    SimpleTriangle* HeadTriangles;
    SimpleVertex* HeadVertices;
    float* Sdf;
    float3* NablaSdf;
    float* HeadVelX;
    float* HeadVelY;
    float* HeadVelZ;
    float* HeadVelWeight;

    // Animation
    MiniVertex* AnimVertices;
    float3* RootBary;
    int* RootTri;
};

// Init/set variables
void launchKernelResetParticles(const int2& blockThread, DeviceBuffers buffers);
void launchKernelInitParticles(const int2& blockThread, DeviceBuffers buffers);

// Integration
void launchKernelFillMatrices(const int2& blockThread, DeviceBuffers buffers);
void launchKernelSolveVelocity(const int2& blockThread, DeviceBuffers buffers);
void launchKernelPositionUpdate(const int2& blockThread, DeviceBuffers buffers);

// Eulerian solver
void launchKernelSegmentToEulerianGrid(const int2& blockThreadParticles, const int2& blockThreadGrid, const int2& blockThreadU,
                                       const int2& blockThreadV, const int2& blockThreadW, DeviceBuffers buffers);
void launchKernelEulerianGridToParticle(const int2& blockThread, DeviceBuffers buffers);
void launchKernelProjectVelocity(const int2& blockThreadGrid, const int2& blockThreadU, const int2& blockThreadV, const int2& blockThreadW,
                                 const int& numIter, DeviceBuffers buffers);

// Hair-solir solver
void launchKernelSDFHeadCollision(const int2& blockThread, DeviceBuffers buffers);
void launchKernelNablaSDF(const int2& blockThread, DeviceBuffers buffers);
void launchKernelInitHeadVel(const int2& blockThread, DeviceBuffers buffers);
void launchKernelInitHeadVerticesVel(const int2& blockThread, DeviceBuffers buffers);
void launchKernelUpdateVelocitySdf(const int2& blockThreadSdf, const int2& blockThreadVertices, DeviceBuffers buffers);

// Additional dynamics
void launchKernelStrainLimiting(const int2& blockThread, DeviceBuffers buffers);
void launchKernelSwapPositions(const int2& blockThread, DeviceBuffers buffers);
void launchKernelMoveRoots(const int2& blockThread, DeviceBuffers buffers);
void launchKernelUpdateMesh(const int2& blockThread, DeviceBuffers buffers);

// Grooming
void launchKernelCutHair(const int2& blockThread, DeviceBuffers buffers);
// void launchKernelCutSelect(const int2& blockthread, const int2& blockthreadRoot, float3 pos, float3 dir, DeviceBuffers buffers);

// Animation
// void launchKernelUpdateRootsSeq(const int2& blockthread, DeviceBuffers buffers);
void launchKernelUpdateAnimSeq(const int2& blockthread, const int& a, const int& b, const float& lambda, DeviceBuffers buffers);

// Host helpers
// void launchKernelCudaToGLInter(const int2& blockThread, float* hairVbo, DeviceBuffers buffers);
// void launchKernelCudaToGLHair(const int2& blockThread, float* hairVbo, DeviceBuffers buffers);
// void launchKernelCudaToGLMesh(const int2& blockThread, float* meshVbo, DeviceBuffers buffers);
void copySimParamsToDevice(SimulationParams* hostParams);

__host__ __device__ float3 BaryCoordinates(const float3& p, const float3& a, const float3& b, const float3& c);