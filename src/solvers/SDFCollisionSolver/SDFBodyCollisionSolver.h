#ifndef SDF_BODY_COLLISION_SOLVER_H_
#define SDF_BODY_COLLISION_SOLVER_H_

#include <thread>
#include <chrono>

#include "../../Kami.h"
#include "../Solver.h"
#include "cudaSDFCollision.cuh"
#include "../../utilities/cubuffer.h"

// Solve SDFcollision
class SDFBodyCollisionSolver : public Solver {
   public:
    SDFBodyCollisionSolver(wk_ptr<KamiSimulator> _kami) : Solver(_kami) {};
    void Construct(string settingJsonPath = "") override;
    void SolveNextStep(float dt) override;

    void EnableStrainLimiting(bool yn);
    void UploadHair() override;
    void DownloadHair() override;

   private:
    struct JsonParams {
        float SDFThreshold = 0.015f;
        float StrainError = 0.1f;
        bool UseStrainLimiting = false;
        JsonParams() {
            SDFThreshold = 0.015f;
            StrainError = 0.1f;
            UseStrainLimiting = false;
        }
    };

    // parameters
    SDFSimulationParams simParams;

    // block sizes for CUDA
    int2 BlockThreadHair = make_int2(0, 256);  // %% for hair in CUDA
    int2 BlockThreadRoot = make_int2(0, 256);  // %% for roots in CUDA
    int2 BlockThreadSdf = make_int2(0, 256);   // % for sdf grid in CUDA

    sh_ptr<float> sdfCreatedTime = make_shared<float>(0.f);

    // Initial hair data
    sh_ptr<CuBuffer<ParticleForSDFColl>> particlesBuffer;
    sh_ptr<CuBuffer<float>> restLengthsBuffer;
    sh_ptr<CuBuffer<int>> rootIdxBuffer;

    // buffers
    SDFDeviceBuffers deviceBuffers;

    sh_ptr<CuBuffer<float>> SDFBuffer;        // body's SDF
    sh_ptr<CuBuffer<float3>> nablaSDFBuffer;  // normals in the SDF

    // These buffers is used if the mesh moves.
    sh_ptr<CuBuffer<float>> headVelXBuffer;  // head velocity field
    sh_ptr<CuBuffer<float>> headVelYBuffer;
    sh_ptr<CuBuffer<float>> headVelZBuffer;

    void InitCudaDevice();
    JsonParams LoadSetting(string settingJsonPath = "");
    void InitParams(JsonParams jp);
    // initialize hair's rootIdx and restLengths
    void InitAdditionalHairData();
    void InitBuffers();

    void SendLatestSDFToCuda();

    // copy hairs' dynamic infomations (simParams and particles)
    void SendSimParamsToCuda(float dt);
    void RecieveHairFromCuda();
    int DivUp(const int& a, const int& b);  // Computes necesary number of blocks given number of threads
};

#endif /* SDF_BODY_COLLISION_SOLVER_H_ */
