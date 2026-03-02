#pragma once

#include <fstream>
#include <chrono>
#include <memory>
#include <Eigen/Sparse>

#include "../../utilities/cubuffer.h"
#include "HairSim.cuh"

#include "../../../extern/glm/glm/glm.hpp"

class Hair;
class Body;
class KamiSimulator;
class HairGenerator;
class Particle;
class InterParticle;
class SFieldCPU;
class SimpleObject;
class SimpleTriangle;
class SimpleVertex;
class MiniVertex;
template <class T>
class CuBuffer;

using std::weak_ptr;
using namespace std;

class AMSSolver {
   public:
    struct ModParams {
        float windPeak = 3.f;
        float SDFThreshold = 0.013f;
    };
    // Constructor
    AMSSolver(const HairInfo& hair, const HeadInfo& head);
    AMSSolver(weak_ptr<Hair> _hair, weak_ptr<Body> _body);
    ~AMSSolver();

    // void Clear();
    void InitParams();
    void ModifyParams(ModParams mp);

    // Init methods
    void InitBuffers(vector<shared_ptr<SimpleObject>>& objs, shared_ptr<HairGenerator> hairGen);
    void InitParticles();

    // Updaters
    // void ComputeSDF();
    void ComputeSDFCPU(weak_ptr<Body> _body);
    void ComputeBary();
    void UpdateSimulation(weak_ptr<Hair> _hair, weak_ptr<Body> _body);
    void UpdateProcessor(weak_ptr<Hair> _hair, weak_ptr<Body> _body);
    void Reset();
    void UseSdfOrNot(bool yn);

    // Getters/setters
    SimulationParams& GetParamsSimRef() { return SimParams; }
    HeadInfo HeadInformation;

    void UploadHair(weak_ptr<Hair> _hair, weak_ptr<Body> _body);
    void DownloadHair(weak_ptr<Hair> _hair, weak_ptr<Body> _body);

   private:
    static const int EULER_DIM = 64;
    // Parameters
    SimulationParams SimParams;

    // Import/Export
    // Copy from GPU to CPU
    void WritebackData(weak_ptr<Hair> _hair, weak_ptr<Body> _body);
    // Copy from GPU to CPU
    void WritebackKamiHair(weak_ptr<Hair> hair);
    // Copy from CPU to GPU
    void SendData(weak_ptr<Hair> _hair, weak_ptr<Body> _body);
    // Copy from CPU to GPU
    void SendKamiHair(weak_ptr<Hair> hair);
    // void WriteHair(const string& fileName);
    // void WriteObj(const string& fileName);
    // void WriteData();
    // void ExportData();

    // Step Update
    void UpdateAnimation(weak_ptr<Body> _body);
    void UpdateDynamics();
    void UpdateRoots();
    void UpdateMesh();

    // Preprocessing
    void FixRootPositions();

    // Custom Animations (for experiments)
    void WindBlowing();

    // Rigid head-motion
    shared_ptr<CuBuffer<hcuMat4>> RigidMotion;

    // Interpolation
    shared_ptr<CuBuffer<InterParticle>> InterParticles;

    // Hair Data
    shared_ptr<CuBuffer<Particle>> Particles;
    shared_ptr<CuBuffer<float>> RestLengths;
    shared_ptr<CuBuffer<int>> RootIdx;

    // Heptadiagional sparse LU solver
    shared_ptr<CuBuffer<hcuMat3>> StrandA;
    shared_ptr<CuBuffer<hcuMat3>> StrandL;
    shared_ptr<CuBuffer<hcuMat3>> StrandU;
    shared_ptr<CuBuffer<float3>> StrandV;
    shared_ptr<CuBuffer<float3>> StrandB;

    // Hair-hair interactions
    shared_ptr<CuBuffer<float>> HairWeightU;  // hair-"fluid" weight U
    shared_ptr<CuBuffer<float>> HairWeightV;  // hair-"fluid" weight V
    shared_ptr<CuBuffer<float>> HairWeightW;  // hair-"fluid" weight W

    shared_ptr<CuBuffer<float>> HairVelU;  // hair-"fluid" velocity U
    shared_ptr<CuBuffer<float>> HairVelV;  // hair-"fluid" velocity V
    shared_ptr<CuBuffer<float>> HairVelW;  // hair-"fluid" velocity W

    shared_ptr<CuBuffer<float2>> HairPressure;  // hair-"fluid" pressure
    shared_ptr<CuBuffer<CellType>> VoxelType;   // to categorize all voxels
    shared_ptr<CuBuffer<float>> HairPicU;       // incompressible velocity fields
    shared_ptr<CuBuffer<float>> HairPicV;
    shared_ptr<CuBuffer<float>> HairPicW;
    shared_ptr<CuBuffer<float>> HairDiv;  // divergence hair fluid field

    // Hair-solid interactions
    shared_ptr<CuBuffer<float3>> RootBary;  // barycentric coordinates of roots
    shared_ptr<CuBuffer<int>> RootTri;      // triangles to wich roots are attached
    shared_ptr<CuBuffer<float>> HeadVelX;   // head velocity field
    shared_ptr<CuBuffer<float>> HeadVelY;
    shared_ptr<CuBuffer<float>> HeadVelZ;
    shared_ptr<CuBuffer<float>> HeadVelWeight;
    shared_ptr<CuBuffer<float>> Sdf;  // head sdf
    shared_ptr<CuBuffer<float3>> NablaSdf;

    // Solid Object Data
    shared_ptr<CuBuffer<SimpleTriangle>> HeadTriangles;  // head triangle GPU
    shared_ptr<CuBuffer<SimpleVertex>> HeadVertices;     // head nodes GPU
    shared_ptr<CuBuffer<MiniVertex>> AnimVertices;       // sequence of vertices GPU
    vector<shared_ptr<SimpleObject>> Meshes;             // head & others
    shared_ptr<SFieldCPU> SdfCPU;                        // cpu sdf computation

    // Misc
    DeviceBuffers DBuffers;

    int3 EulerDim = make_int3(EULER_DIM);

    // CUDA Kernel Dims
    int2 BlockThreadGridU = make_int2(0, 256);  // Number of blocks and threads for U-grid in CUDA
    int2 BlockThreadGridV = make_int2(0, 256);  // %% for V-grid in CUDA
    int2 BlockThreadGridW = make_int2(0, 256);  // %% for W-grid in CUDA
    int2 BlockThreadHair = make_int2(0, 256);   // %% for hair in CUDA
    int2 BlockThreadRoot = make_int2(0, 256);   // %% for roots in CUDA
    int2 BlockThreadMesh = make_int2(0, 256);   // % for triangles in CUDA
    int2 BlockThreadVert = make_int2(0, 256);   // % for vertices in CUDA
    int2 BlockThreadGrid = make_int2(0, 256);   // % for grid in CUDA
    int2 BlockThreadSdf = make_int2(0, 256);    // % for sdf grid in CUDA
    int2 BlockThreadInter = make_int2(0, 256);  // % for hair interpolation in CUDA

    // Helpers
    // float DynamicQuant(vector<float> times, vector<float> values, float t);
    void UpdateAnimGridBox(const int& a, const int& b, const float& lambda);
    void AnimTimeToFrame(int& a, int& b, float& lambda);
    int DivUp(const int& a, const int& b);  // Computes necesary number of blocks given number of threads
    inline Eigen::Vector3f Float2eigen(const float3& u) { return Eigen::Vector3f(u.x, u.y, u.z); }
    inline glm::vec3 Float2GLM(const float3& u) { return glm::vec3(u.x, u.y, u.z); }
    inline float3 GLM2Float(const glm::vec3& u) { return make_float3(u.x, u.y, u.z); };
    inline float3 Eigen2float(const Eigen::Vector3f& u) { return make_float3(u[0], u[1], u[2]); }
    void FreeBuffers();
    int SelectCudaDevice();
    vector<int> StrandLength;
    int ExportCounter = 0;
    int ExportIdx = 0;
};